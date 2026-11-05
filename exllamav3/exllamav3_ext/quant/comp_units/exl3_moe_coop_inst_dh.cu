#include <algorithm>
#include "exl3_moe_coop_instances.cuh"
#include "../exl3_moe_coop_kernel.cuh"

// Dual persistent kernel: routed experts at K + 0.5 bpw (mul1), shared expert at 4..6 bpw (mul1)
#define EXL3_MOE_COOP_D_SMEM_H(R, S) \
    max(max(smem_a_bytes<R, true>(Hi), smem_b_bytes<R, true>()), max(smem_a_bytes<S>(Hi), smem_b_bytes<S>()))

MoeCoopKernel exl3_moe_coop_kernel_d_h(int bits_r, int bits_s, int cb, int Hi)
{
    using namespace exl3_moe_coop_ns;
    using std::max;
    if (cb != 2) return { nullptr, 0 };
    #define DH(R, S) if (bits_r == R && bits_s == S) return { (void*) exl3_moe_coop_d_kernel<R, S, 2, true>, EXL3_MOE_COOP_D_SMEM_H(R, S) };
    DH(1, 4) DH(1, 5) DH(1, 6)
    DH(2, 4) DH(2, 5) DH(2, 6)
    #undef DH
    return { nullptr, 0 };
}
