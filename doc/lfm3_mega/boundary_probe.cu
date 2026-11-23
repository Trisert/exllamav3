// Cost of a dependency boundary on this card, per step, with the decode token's shape (S steps, full-grid launches):
//   A: CUDA graph of S dependent kernels (what the decode graph does today: 243 kernels per token)
//   B: one persistent kernel, S global barriers (sense-reversing atomic barrier, one CTA per SM x occ)
//   C: one persistent kernel, S cooperative-groups grid.sync()
// Each step does a small amount of per-thread work (W FMAs) so the boundary sits between real (tiny) work.
// nvcc -O3 -arch=sm_60 -rdc=true boundary_probe.cu -o boundary_probe   (sm_75 for the RTX 2060)
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void step_kernel(float* buf, int w)
{
    float v = buf[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < w; ++i) v = fmaf(v, 0.999f, 0.001f);
    buf[blockIdx.x * blockDim.x + threadIdx.x] = v;
}

__device__ __forceinline__ void atomic_barrier(unsigned int* count, volatile unsigned int* gen, unsigned int nblocks)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int g = *gen;
        __threadfence();
        if (atomicAdd(count, 1) == nblocks - 1)
        {
            *count = 0;
            __threadfence();
            atomicAdd((unsigned int*) gen, 1);
        }
        else
            while (*gen == g) { }
        __threadfence();
    }
    __syncthreads();
}

__global__ void persistent_atomic(float* buf, int w, int steps, unsigned int* count, unsigned int* gen)
{
    float v = buf[blockIdx.x * blockDim.x + threadIdx.x];
    for (int s = 0; s < steps; ++s)
    {
        for (int i = 0; i < w; ++i) v = fmaf(v, 0.999f, 0.001f);
        buf[blockIdx.x * blockDim.x + threadIdx.x] = v;
        atomic_barrier(count, gen, gridDim.x);
        v = buf[((blockIdx.x + 1) % gridDim.x) * blockDim.x + threadIdx.x];
    }
}

__global__ void persistent_coop(float* buf, int w, int steps)
{
    cg::grid_group grid = cg::this_grid();
    float v = buf[blockIdx.x * blockDim.x + threadIdx.x];
    for (int s = 0; s < steps; ++s)
    {
        for (int i = 0; i < w; ++i) v = fmaf(v, 0.999f, 0.001f);
        buf[blockIdx.x * blockDim.x + threadIdx.x] = v;
        grid.sync();
        v = buf[((blockIdx.x + 1) % gridDim.x) * blockDim.x + threadIdx.x];
    }
}

int main()
{
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    const int S = 243, threads = 256, reps = 200;
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, persistent_coop, threads, 0));
    printf("%s, %d SMs, occupancy %d blocks/SM at %d threads\n", p.name, p.multiProcessorCount, occ, threads);
    float* buf;
    unsigned int *count, *gen;
    CK(cudaMalloc(&buf, 64 * 1024 * 1024));
    CK(cudaMemset(buf, 0, 64 * 1024 * 1024));
    CK(cudaMalloc(&count, 4));
    CK(cudaMalloc(&gen, 4));
    CK(cudaMemset(count, 0, 4));
    CK(cudaMemset(gen, 0, 4));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    for (int w : {0, 64})
    for (int bpsm : {1, 2})
    {
        int nb = p.multiProcessorCount * bpsm;
        if (bpsm > occ) continue;
        // A: graph of S dependent kernels
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < S; ++s) step_kernel<<<nb, threads, 0, st>>>(buf, w);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int i = 0; i < 10; ++i) cudaGraphLaunch(ge, st);
        cudaEventRecord(e0, st);
        for (int i = 0; i < reps; ++i) cudaGraphLaunch(ge, st);
        cudaEventRecord(e1, st);
        CK(cudaEventSynchronize(e1));
        float ta;
        cudaEventElapsedTime(&ta, e0, e1);

        // B: persistent, atomic barrier
        for (int i = 0; i < 10; ++i) persistent_atomic<<<nb, threads, 0, st>>>(buf, w, S, count, gen);
        cudaEventRecord(e0, st);
        for (int i = 0; i < reps; ++i) persistent_atomic<<<nb, threads, 0, st>>>(buf, w, S, count, gen);
        cudaEventRecord(e1, st);
        CK(cudaEventSynchronize(e1));
        float tb;
        cudaEventElapsedTime(&tb, e0, e1);

        // C: persistent, cooperative grid.sync
        int steps = S;
        void* args[] = {&buf, &w, &steps};
        for (int i = 0; i < 10; ++i) CK(cudaLaunchCooperativeKernel((void*) persistent_coop, nb, threads, args, 0, st));
        cudaEventRecord(e0, st);
        for (int i = 0; i < reps; ++i) cudaLaunchCooperativeKernel((void*) persistent_coop, nb, threads, args, 0, st);
        cudaEventRecord(e1, st);
        CK(cudaEventSynchronize(e1));
        float tc;
        cudaEventElapsedTime(&tc, e0, e1);

        // floor: the same work with no boundaries (one kernel, S * w FMAs)
        float tf = 0.0f;
        {
            for (int i = 0; i < 10; ++i) step_kernel<<<nb, threads, 0, st>>>(buf, w * S);
            cudaEventRecord(e0, st);
            for (int i = 0; i < reps; ++i) step_kernel<<<nb, threads, 0, st>>>(buf, w * S);
            cudaEventRecord(e1, st);
            CK(cudaEventSynchronize(e1));
            cudaEventElapsedTime(&tf, e0, e1);
        }
        auto us = [&] (float t) { return t * 1000.0f / reps; };
        printf("work %2d FMA, %3d blocks: per token (%d steps) graph %7.1f us | persistent atomic %7.1f | coop %7.1f | "
               "no-boundary floor %6.1f  -> per boundary %.2f / %.2f / %.2f us\n",
               w, nb, S, us(ta), us(tb), us(tc), us(tf),
               (us(ta) - us(tf)) / S, (us(tb) - us(tf)) / S, (us(tc) - us(tf)) / S);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    return 0;
}
