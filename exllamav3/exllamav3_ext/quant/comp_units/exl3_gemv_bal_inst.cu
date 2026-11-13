#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../exl3_gemv_kernel.cuh"

// Balanced-split instances (BAL, m = 1) of the small-m GEMV kernel: 2..8 bpw, mul1 codebook, the default extraction
// style (shuffle below 5 bpw, staged tile words from 5 bpw)
void* exl3_gemv_select_kernel_bal(int bits, bool c_fp32, int cfg)
{
    #define SEL(bits_, sm_, fp32_, cfg_) \
        if (bits == bits_ && c_fp32 == fp32_ && cfg == cfg_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, 2, 0, cfg_, sm_, false, true>;
    #define SEL_GRID(bits_, sm_) \
        SEL(bits_, sm_, false, 0) SEL(bits_, sm_, false, 1) SEL(bits_, sm_, true, 0) SEL(bits_, sm_, true, 1)
    SEL_GRID(2, false) SEL_GRID(3, false) SEL_GRID(4, false)
    SEL_GRID(5, true) SEL_GRID(6, true) SEL_GRID(7, true) SEL_GRID(8, true)
    #undef SEL_GRID
    #undef SEL
    return nullptr;
}
