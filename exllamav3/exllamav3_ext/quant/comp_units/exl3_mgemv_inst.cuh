#pragma once
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../exl3_gemv_kernel.cuh"

// Multi-matrix small-m GEMV instances (exl3_mgemv_kernel) for one bitrate: mul1, staged tile words, the Pascal row
// modes (MMODE 0: m = 1, 2: m = 2, 4: m = 3..4, 1: m = 5..8, 5: m = 9..16), narrow / wide config, the balanced split
// at m = 1
#define EXL3_MGEMV_SEL(bits_, fp32_, mm_, cfg_, bal_) \
    if (c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && bal == bal_) \
        return (void*) exl3_mgemv_kernel<bits_, fp32_, mm_, cfg_, bal_>;
#define EXL3_MGEMV_SEL_MM(bits_, mm_) \
    EXL3_MGEMV_SEL(bits_, false, mm_, 0, false) EXL3_MGEMV_SEL(bits_, false, mm_, 1, false) \
    EXL3_MGEMV_SEL(bits_, true,  mm_, 0, false) EXL3_MGEMV_SEL(bits_, true,  mm_, 1, false)
#define EXL3_MGEMV_SEL_BITS(bits_) \
    EXL3_MGEMV_SEL_MM(bits_, 0) EXL3_MGEMV_SEL_MM(bits_, 1) EXL3_MGEMV_SEL_MM(bits_, 2) \
    EXL3_MGEMV_SEL_MM(bits_, 4) EXL3_MGEMV_SEL_MM(bits_, 5) \
    EXL3_MGEMV_SEL(bits_, false, 0, 1, true) EXL3_MGEMV_SEL(bits_, true, 0, 1, true)
