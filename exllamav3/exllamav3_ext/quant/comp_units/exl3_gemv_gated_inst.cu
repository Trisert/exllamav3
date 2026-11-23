#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../exl3_gemv_kernel.cuh"

// Gated-input m = 1 GEMV instances (exl3_gemv_gated_kernel): mul1 7 / 8 bpw, narrow / wide config, plain or balanced
// split (the SwiGLU down projections of the 8 bpw LFM3 quants)
void* exl3_gemv_select_kernel_gated(int bits, bool c_fp32, int cfg, bool bal)
{
    #define SEL(bits_, fp32_, cfg_, bal_) \
        if (bits == bits_ && c_fp32 == fp32_ && cfg == cfg_ && bal == bal_) \
            return (void*) exl3_gemv_gated_kernel<bits_, fp32_, cfg_, bal_>;
    #define SEL_BITS(bits_) \
        SEL(bits_, false, 0, false) SEL(bits_, false, 1, false) SEL(bits_, true, 0, false) SEL(bits_, true, 1, false) \
        SEL(bits_, false, 1, true) SEL(bits_, true, 1, true)
    SEL_BITS(7) SEL_BITS(8)
    #undef SEL_BITS
    #undef SEL
    return nullptr;
}
