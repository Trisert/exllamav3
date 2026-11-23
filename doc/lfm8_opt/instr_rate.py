"""Per-SM issue throughput of the instructions in the 8 bpw GEMV inner loop (P100 / sm_60): 8 independent chains per
thread, 4096 iterations, full occupancy; reports warp-instructions per clock per SM (2.0 = full rate on sm_60, 64
lanes). usage: python instr_rate.py"""
import os, torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#define N_IT 4096
#define CH 8
template <int OP>
__global__ void k(uint32_t* out, uint32_t seed)
{
    uint32_t v[CH];
    #pragma unroll
    for (int c = 0; c < CH; ++c) v[c] = seed ^ (threadIdx.x * 0x9E3779B9u + c * 0x85EBCA6Bu);
    uint32_t z = seed >> 31;  // 0 at runtime, opaque to the compiler
    for (int i = 0; i < N_IT; ++i)
    {
        #pragma unroll
        for (int c = 0; c < CH; ++c)
        {
            uint32_t x = v[c], r;
            if constexpr (OP == 0) asm volatile ("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(z), "r"(x));
            if constexpr (OP == 1) asm volatile ("mul.lo.u16 %0, %1, 0x83DC;" : "=h"(*(unsigned short*)&r) : "h"((unsigned short) x));
            if constexpr (OP == 2) asm volatile ("mad.lo.u32 %0, %1, 65536, %2;" : "=r"(r) : "r"(x), "r"(z));
            if constexpr (OP == 3) asm volatile ("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(z), "r"(c + 8));
            if constexpr (OP == 4) asm volatile ("prmt.b32 %0, %1, %2, 0x5432;" : "=r"(r) : "r"(x), "r"(z));
            if constexpr (OP == 5) asm volatile ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(x), "r"(z), "r"(0x00ff00ffu));
            if constexpr (OP == 6) asm volatile ("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(0x3c003c00u), "r"(z));
            if constexpr (OP == 7) asm volatile ("add.u32 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(z));
            if constexpr (OP == 8) asm volatile ("mul.wide.u16 %0, %1, 0xD12D;" : "=r"(r) : "h"((unsigned short) x));
            if constexpr (OP == 9) asm volatile ("mul.lo.u32 %0, %1, 0x83DCD12D;" : "=r"(r) : "r"(x));
            v[c] = r;
        }
    }
    uint32_t s = 0;
    #pragma unroll
    for (int c = 0; c < CH; ++c) s ^= v[c];
    if (s == 0x12345678u) out[0] = s;
}
void run(int op, torch::Tensor out, int blocks, int threads)
{
    auto p = (uint32_t*) out.data_ptr();
    switch (op) {
        case 0: k<0><<<blocks, threads>>>(p, 7); break; case 1: k<1><<<blocks, threads>>>(p, 7); break;
        case 2: k<2><<<blocks, threads>>>(p, 7); break; case 3: k<3><<<blocks, threads>>>(p, 7); break;
        case 4: k<4><<<blocks, threads>>>(p, 7); break; case 5: k<5><<<blocks, threads>>>(p, 7); break;
        case 6: k<6><<<blocks, threads>>>(p, 7); break; case 7: k<7><<<blocks, threads>>>(p, 7); break;
        case 8: k<8><<<blocks, threads>>>(p, 7); break; case 9: k<9><<<blocks, threads>>>(p, 7); break;
    }
}
"""
cpp_src = "void run(int op, torch::Tensor out, int blocks, int threads);"
build = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "instr_rate_build")
os.makedirs(build, exist_ok = True)
m = load_inline("instr_rate", cpp_src, cuda_sources = cuda_src, functions = ["run"], build_directory = build,
                extra_cuda_cflags = ["-O3", "-gencode=arch=compute_60,code=sm_60", "-gencode=arch=compute_75,code=sm_75",
                                     "-ccbin", "/usr/bin/g++-14"], verbose = False)
props = torch.cuda.get_device_properties(0)
sms = props.multi_processor_count
out = torch.zeros(4, dtype = torch.int32, device = "cuda")
names = ["VABSDIFF4", "mul.lo.u16 (XMAD)", "mad.lo.u32 x65536 (XMAD.PSL)", "SHF (funnel)", "PRMT", "LOP3",
         "HFMA2", "IADD", "mul.wide.u16 (XMAD)", "mul.lo.u32 (IMAD emul)"]
threads, blocks = 1024, sms * 2
for op in range(10):
    m.run(op, out, blocks, threads); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    e0.record(); m.run(op, out, blocks, threads); e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)
    clk = props.clock_rate * 1e3 if hasattr(props, "clock_rate") else 1.328e9
    warp_instr = blocks * threads / 32 * 4096 * 8
    print(f"{names[op]:32s} {ms:7.3f} ms  {warp_instr / (ms * 1e-3) / sms / 1.328e9:5.2f} warp-instr/clk/SM (at 1.328 GHz)")
