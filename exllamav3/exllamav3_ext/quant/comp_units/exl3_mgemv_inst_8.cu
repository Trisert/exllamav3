#include "exl3_mgemv_inst.cuh"

void* exl3_mgemv_select_kernel_8(bool c_fp32, int mmode, int cfg, bool bal)
{
    EXL3_MGEMV_SEL_BITS(8)
    return nullptr;
}
