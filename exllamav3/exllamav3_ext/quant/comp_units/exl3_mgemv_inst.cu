#include "exl3_mgemv_inst.cuh"

void* exl3_mgemv_select_kernel_8(bool c_fp32, int mmode, int cfg, bool bal);

// Multi-matrix GEMV instances, 7 bpw here and 8 bpw in exl3_mgemv_inst_8.cu (the rates whose bundles Pascal otherwise
// unfuses for the single-matrix GEMV)
void* exl3_mgemv_select_kernel(int bits, bool c_fp32, int mmode, int cfg, bool bal)
{
    if (bits == 8) return exl3_mgemv_select_kernel_8(c_fp32, mmode, cfg, bal);
    if (bits != 7) return nullptr;
    EXL3_MGEMV_SEL_BITS(7)
    return nullptr;
}
