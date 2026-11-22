// SASS probe: mul1 product idx * 0x83DCD12D of a 16-bit half register on sm_60, three PTX forms.
// nvcc -arch=sm_60 -cubin -O3 -o xmad_probe.cubin xmad_probe.cu && cuobjdump -sass xmad_probe.cubin
#include <cstdint>

#define PROD_ASM(BODY, HALF) \
    asm ("{ .reg .u16 xl, xh, h16; .reg .u32 lo, hi;\n mov.b32 {xl, xh}, %1;\n" BODY "}\n" : "=r"(r) : "r"(x))

template <int V, bool HI> __device__ __forceinline__ uint32_t prod(uint32_t x)
{
    uint32_t r;
    // V0: current (mul.lo.u16 high product, zero-extended)
    if constexpr (V == 0 && HI)  PROD_ASM(" mul.wide.u16 lo, xh, 0xD12D;\n mul.lo.u16 h16, xh, 0x83DC;\n cvt.u32.u16 hi, h16;\n mad.lo.u32 %0, hi, 65536, lo;\n", 1);
    if constexpr (V == 0 && !HI) PROD_ASM(" mul.wide.u16 lo, xl, 0xD12D;\n mul.lo.u16 h16, xl, 0x83DC;\n cvt.u32.u16 hi, h16;\n mad.lo.u32 %0, hi, 65536, lo;\n", 0);
    // V1: full 16x16 high product, the shift drops its top half
    if constexpr (V == 1 && HI)  PROD_ASM(" mul.wide.u16 lo, xh, 0xD12D;\n mul.wide.u16 hi, xh, 0x83DC;\n mad.lo.u32 %0, hi, 65536, lo;\n", 1);
    if constexpr (V == 1 && !HI) PROD_ASM(" mul.wide.u16 lo, xl, 0xD12D;\n mul.wide.u16 hi, xl, 0x83DC;\n mad.lo.u32 %0, hi, 65536, lo;\n", 0);
    // V2: shl + add
    if constexpr (V == 2 && HI)  PROD_ASM(" mul.wide.u16 lo, xh, 0xD12D;\n mul.wide.u16 hi, xh, 0x83DC;\n shl.b32 hi, hi, 16;\n add.u32 %0, hi, lo;\n", 1);
    if constexpr (V == 2 && !HI) PROD_ASM(" mul.wide.u16 lo, xl, 0xD12D;\n mul.wide.u16 hi, xl, 0x83DC;\n shl.b32 hi, hi, 16;\n add.u32 %0, hi, lo;\n", 0);
    // V3: 32-bit multiply of the zero-extended index by 0x83DC0000 (= the high product shifted)
    if constexpr (V == 3 && HI)  PROD_ASM(" mul.wide.u16 lo, xh, 0xD12D;\n cvt.u32.u16 hi, xh;\n mad.lo.u32 %0, hi, 0x83DC0000, lo;\n", 1);
    if constexpr (V == 3 && !HI) PROD_ASM(" mul.wide.u16 lo, xl, 0xD12D;\n cvt.u32.u16 hi, xl;\n mad.lo.u32 %0, hi, 0x83DC0000, lo;\n", 0);
    // V4: high product via mul.wide.u16 into a 32-bit value, then shifted multiply-add of 16-bit halves
    if constexpr (V == 4 && HI)  PROD_ASM(" mul.wide.u16 lo, xh, 0xD12D;\n mul.lo.u16 h16, xh, 0x83DC;\n mov.b32 hi, {0, h16};\n add.u32 %0, hi, lo;\n", 1);
    if constexpr (V == 4 && !HI) PROD_ASM(" mul.wide.u16 lo, xl, 0xD12D;\n mul.lo.u16 h16, xl, 0x83DC;\n mov.b32 hi, {0, h16};\n add.u32 %0, hi, lo;\n", 0);
    return r;
}

template <int V> __global__ void k(const uint32_t* in, uint32_t* out)
{
    uint32_t x = in[threadIdx.x];
    out[threadIdx.x] = prod<V, true>(x) ^ (prod<V, false>(x) << 1);
}
template __global__ void k<0>(const uint32_t*, uint32_t*);
template __global__ void k<1>(const uint32_t*, uint32_t*);
template __global__ void k<2>(const uint32_t*, uint32_t*);
template __global__ void k<3>(const uint32_t*, uint32_t*);
template __global__ void k<4>(const uint32_t*, uint32_t*);
