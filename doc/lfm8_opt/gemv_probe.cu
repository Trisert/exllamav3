// Standalone phase probe of the m = 1 exl3 GEMV (8 bpw mul1, staged words) on the P100: per-block clock64 at the
// phase boundaries (EXL3_GEMV_BAL_DBG & 128) and in-graph duration over L2-cold weight copies (random trellis bits;
// the decode cost does not depend on the values).
// nvcc -arch=sm_60 -O3 -std=c++17 -DEXL3_GEMV_BAL_DBG=128 -I<ext> -o gemv_probe gemv_probe.cu
// usage: gemv_probe k n [cfg 0|1] [bal 0|1]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemv_kernel.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

template <int CFG, bool BAL>
static void* kern() { return (void*) exl3_gemv_kernel<8, false, 2, 0, CFG, true, false, BAL>; }

int main(int argc, char** argv)
{
    const int k = atoi(argv[1]), n = atoi(argv[2]);
    const int cfg = argc > 3 ? atoi(argv[3]) : 0;
    const int bal = argc > 4 ? atoi(argv[4]) : 0;
    void* kernel = cfg == 0 ? (bal ? kern<0, true>() : kern<0, false>()) : (bal ? kern<1, true>() : kern<1, false>());
    const int block = bal ? 256 : (cfg == 0 ? 512 : 256);
    const int cols = bal ? 64 : (cfg == 0 ? 32 : 64);
    int bps, sms;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, block, 0));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int grid = bal ? bps * sms : std::min(n / cols, bps * sms);

    const size_t wbytes = (size_t) k * n;   // 8 bpw
    const int copies = std::max(1, (int) ((64u << 20) / wbytes));
    std::vector<uint16_t*> B(copies);
    std::vector<uint32_t> host(wbytes / 4);
    for (auto& v : host) v = (uint32_t) rand() * 2654435761u;
    for (auto& b : B) { CK(cudaMalloc(&b, wbytes)); CK(cudaMemcpy(b, host.data(), wbytes, cudaMemcpyHostToDevice)); }
    half *A, *A_had, *C, *suh, *svh;
    CK(cudaMalloc(&A, k * 2)); CK(cudaMalloc(&A_had, k * 2)); CK(cudaMalloc(&C, n * 4));
    CK(cudaMalloc(&suh, k * 2)); CK(cudaMalloc(&svh, n * 2));
    std::vector<half> hv(std::max(k, n), __float2half(0.5f));
    CK(cudaMemcpy(A, hv.data(), k * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(suh, hv.data(), k * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(svh, hv.data(), n * 2, cudaMemcpyHostToDevice));
    int* locks;
    const size_t lock_ints = EXL3_GEMV_WS_OFFSET + EXL3_GEMV_WS_FLOATS;
    CK(cudaMalloc(&locks, lock_ints * 4)); CK(cudaMemset(locks, 0, lock_ints * 4));

    int m = 1;
    cudaStream_t st; CK(cudaStreamCreate(&st));
    auto launch = [&] (int i)
    {
        const uint16_t* b = B[i % copies];
        void* args[] = { &A, &b, &C, &m, (void*) &k, (void*) &n, &locks, &suh, &A_had, &svh };
        CK(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(block), args, 0, st));
    };
    for (int i = 0; i < 20; ++i) launch(i);
    CK(cudaStreamSynchronize(st));

    // In-graph duration: 200 launches captured once, replayed
    const int R = 200;
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < R; ++i) launch(i);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best = 1e9;
    for (int rep = 0; rep < 5; ++rep)
    {
        CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
    }
    const float us = best / R * 1000;

    // Phase probe of the last launch (block 0 .. grid-1), cycles at the SM clock
    std::vector<unsigned long long> pr(4096 * 8);
    CK(cudaMemcpyFromSymbol(pr.data(), g_bal_probe, grid * 8 * sizeof(unsigned long long)));
    double ph[5] = {};
    unsigned long long mx[5] = {};
    for (int b = 0; b < grid; ++b)
        for (int i = 0; i < 5; ++i) { ph[i] += pr[b * 8 + i]; mx[i] = std::max(mx[i], pr[b * 8 + i]); }
    int clk; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));   // kHz
    const double c2us = 1000.0 / clk;
    printf("k=%d n=%d cfg=%d bal=%d grid=%d block=%d copies=%d: %.1f us/call in graph, %.0f GB/s | "
           "phase avg/max us: had_in+sync %.1f/%.1f  main %.1f/%.1f  sync %.1f/%.1f  out %.1f/%.1f  total %.1f/%.1f\n",
           k, n, cfg, bal, grid, block, copies, us, wbytes / us / 1e3,
           ph[0] / grid * c2us, mx[0] * c2us, ph[1] / grid * c2us, mx[1] * c2us, ph[2] / grid * c2us, mx[2] * c2us,
           ph[3] / grid * c2us, mx[3] * c2us, ph[4] / grid * c2us, mx[4] * c2us);
    return 0;
}
