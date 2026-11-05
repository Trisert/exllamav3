#pragma once

// Per-K compilation units of the fused decode MoE kernels (exl3_moe_coop_kernel.cuh): each
// exl3_moe_coop_inst_k{K}.cu instantiates the gate/up (A) and down (B) kernels for the three
// codebooks. Returns the kernel pointer for cudaLaunchKernel and the dynamic shared memory size
// (A: for the given Hi; B: fixed)

struct MoeCoopKernel
{
    void* kernel;
    int smem;
};

#define EXL3_MOE_COOP_INST_DECL(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_k##K(int cb, int Hi, bool wide); \
    MoeCoopKernel exl3_moe_coop_kernel_b_k##K(int cb, bool wide); \
    MoeCoopKernel exl3_moe_coop_kernel_f_k##K(int cb, int Hi);

EXL3_MOE_COOP_INST_DECL(1)
EXL3_MOE_COOP_INST_DECL(2)
EXL3_MOE_COOP_INST_DECL(3)
EXL3_MOE_COOP_INST_DECL(4)
EXL3_MOE_COOP_INST_DECL(5)
EXL3_MOE_COOP_INST_DECL(6)
EXL3_MOE_COOP_INST_DECL(7)
EXL3_MOE_COOP_INST_DECL(8)

#undef EXL3_MOE_COOP_INST_DECL

// Half-integer rates K + 0.5 (exl3_moe_coop_inst_h{K}.cu), mul1 codebook only
#define EXL3_MOE_COOP_INST_DECL_H(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_h##K(int Hi, bool wide); \
    MoeCoopKernel exl3_moe_coop_kernel_b_h##K(bool wide); \
    MoeCoopKernel exl3_moe_coop_kernel_f_h##K(int Hi);

EXL3_MOE_COOP_INST_DECL_H(1)
EXL3_MOE_COOP_INST_DECL_H(2)
EXL3_MOE_COOP_INST_DECL_H(3)

#undef EXL3_MOE_COOP_INST_DECL_H

#define EXL3_MOE_COOP_INST_DEF_H(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_h##K(int Hi, bool wide) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_a_bytes<K, true>(Hi); \
        if (wide) return { (void*) exl3_moe_coop_a_kernel<K, 2, true, true>, smem }; \
        return { (void*) exl3_moe_coop_a_kernel<K, 2, false, true>, smem }; \
    } \
    MoeCoopKernel exl3_moe_coop_kernel_b_h##K(bool wide) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_b_bytes<K, true>(); \
        if (wide) return { (void*) exl3_moe_coop_b_kernel<K, 2, true, true>, smem }; \
        return { (void*) exl3_moe_coop_b_kernel<K, 2, false, true>, smem }; \
    } \
    MoeCoopKernel exl3_moe_coop_kernel_f_h##K(int Hi) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_a_bytes<K, true>(Hi) > smem_b_bytes<K, true>() ? smem_a_bytes<K, true>(Hi) : smem_b_bytes<K, true>(); \
        return { (void*) exl3_moe_coop_f_kernel<K, 2, true>, smem }; \
    }

// Body of an instance file
#define EXL3_MOE_COOP_INST_DEF(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_k##K(int cb, int Hi, bool wide) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_a_bytes<K>(Hi); \
        if (wide) switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_a_kernel<K, 0, true>, smem }; \
            case 1: return { (void*) exl3_moe_coop_a_kernel<K, 1, true>, smem }; \
            default: return { (void*) exl3_moe_coop_a_kernel<K, 2, true>, smem }; \
        } \
        switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_a_kernel<K, 0, false>, smem }; \
            case 1: return { (void*) exl3_moe_coop_a_kernel<K, 1, false>, smem }; \
            default: return { (void*) exl3_moe_coop_a_kernel<K, 2, false>, smem }; \
        } \
    } \
    MoeCoopKernel exl3_moe_coop_kernel_b_k##K(int cb, bool wide) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_b_bytes<K>(); \
        if (wide) switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_b_kernel<K, 0, true>, smem }; \
            case 1: return { (void*) exl3_moe_coop_b_kernel<K, 1, true>, smem }; \
            default: return { (void*) exl3_moe_coop_b_kernel<K, 2, true>, smem }; \
        } \
        switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_b_kernel<K, 0, false>, smem }; \
            case 1: return { (void*) exl3_moe_coop_b_kernel<K, 1, false>, smem }; \
            default: return { (void*) exl3_moe_coop_b_kernel<K, 2, false>, smem }; \
        } \
    } \
    MoeCoopKernel exl3_moe_coop_kernel_f_k##K(int cb, int Hi) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_a_bytes<K>(Hi) > smem_b_bytes<K>() ? smem_a_bytes<K>(Hi) : smem_b_bytes<K>(); \
        switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_f_kernel<K, 0>, smem }; \
            case 1: return { (void*) exl3_moe_coop_f_kernel<K, 1>, smem }; \
            default: return { (void*) exl3_moe_coop_f_kernel<K, 2>, smem }; \
        } \
    }

// Dual persistent kernel (shared + routed experts, exl3_moe_coop_d_kernel), one unit per routed
// width R (exl3_moe_coop_inst_d{R}.cu): shared widths R, R + 1, R + 2, codebooks mcg and mul1, integer
// rates. Returns a null kernel for other combinations (the launcher then keeps separate launches)
MoeCoopKernel exl3_moe_coop_kernel_d_r2(int bits_s, int cb, int Hi);
MoeCoopKernel exl3_moe_coop_kernel_d_r3(int bits_s, int cb, int Hi);
MoeCoopKernel exl3_moe_coop_kernel_d_r4(int bits_s, int cb, int Hi);

#define EXL3_MOE_COOP_D_SMEM(R, S) \
    max(max(smem_a_bytes<R>(Hi), smem_b_bytes<R>()), max(smem_a_bytes<S>(Hi), smem_b_bytes<S>()))

#define EXL3_MOE_COOP_INST_DEF_D(R) \
    MoeCoopKernel exl3_moe_coop_kernel_d_r##R(int bits_s, int cb, int Hi) \
    { \
        using namespace exl3_moe_coop_ns; \
        using std::max; \
        if (cb != 1 && cb != 2) return { nullptr, 0 }; \
        if (bits_s == R) \
            return { cb == 1 ? (void*) exl3_moe_coop_d_kernel<R, R, 1> : (void*) exl3_moe_coop_d_kernel<R, R, 2>, EXL3_MOE_COOP_D_SMEM(R, R) }; \
        if (bits_s == R + 1) \
            return { cb == 1 ? (void*) exl3_moe_coop_d_kernel<R, R + 1, 1> : (void*) exl3_moe_coop_d_kernel<R, R + 1, 2>, EXL3_MOE_COOP_D_SMEM(R, R + 1) }; \
        if (bits_s == R + 2) \
            return { cb == 1 ? (void*) exl3_moe_coop_d_kernel<R, R + 2, 1> : (void*) exl3_moe_coop_d_kernel<R, R + 2, 2>, EXL3_MOE_COOP_D_SMEM(R, R + 2) }; \
        return { nullptr, 0 }; \
    }
