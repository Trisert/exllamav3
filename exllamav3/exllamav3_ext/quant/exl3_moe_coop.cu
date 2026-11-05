// Fused decode-shaped MoE path (bsz 1..MAX_BSZN): launcher and tensor front end. Kernels in
// exl3_moe_coop_kernel.cuh, instantiated per (K, codebook) in comp_units/exl3_moe_coop_inst_k*.cu

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <tuple>
#include <cmath>

#include "../util.h"
#include "../util.cuh"
#include "exl3_devctx.cuh"
#include "exl3_moe_coop.cuh"
#include "comp_units/exl3_moe_coop_instances.cuh"
#include "bits_k.cuh"
#define EXL3_MOE_COOP_DEFINE_ROT
#include "exl3_moe_coop_kernel.cuh"

// Output rows of a launch without routed slots (top-k 0): the empty sum, plus the shared-expert term
// when there is one. One warp per (row, 128-column chunk)
__global__ __launch_bounds__(32)
void exl3_moe_coop_empty_kernel(const MoeCoopParams p)
{
    exl3_moe_coop_ns::write_empty_row_chunk(p, blockIdx.x, blockIdx.y, threadIdx.x);
}

static MoeCoopKernel moe_coop_kernel_a(float K_, int cb, int Hi, bool wide)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_a_h1(Hi, wide);
            case 2: return exl3_moe_coop_kernel_a_h2(Hi, wide);
            default: return exl3_moe_coop_kernel_a_h3(Hi, wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_a_k1(cb, Hi, wide);
        case 2: return exl3_moe_coop_kernel_a_k2(cb, Hi, wide);
        case 3: return exl3_moe_coop_kernel_a_k3(cb, Hi, wide);
        case 4: return exl3_moe_coop_kernel_a_k4(cb, Hi, wide);
        case 5: return exl3_moe_coop_kernel_a_k5(cb, Hi, wide);
        case 6: return exl3_moe_coop_kernel_a_k6(cb, Hi, wide);
        case 7: return exl3_moe_coop_kernel_a_k7(cb, Hi, wide);
        default: return exl3_moe_coop_kernel_a_k8(cb, Hi, wide);
    }
}

static MoeCoopKernel moe_coop_kernel_b(float K_, int cb, bool wide)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_b_h1(wide);
            case 2: return exl3_moe_coop_kernel_b_h2(wide);
            default: return exl3_moe_coop_kernel_b_h3(wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_b_k1(cb, wide);
        case 2: return exl3_moe_coop_kernel_b_k2(cb, wide);
        case 3: return exl3_moe_coop_kernel_b_k3(cb, wide);
        case 4: return exl3_moe_coop_kernel_b_k4(cb, wide);
        case 5: return exl3_moe_coop_kernel_b_k5(cb, wide);
        case 6: return exl3_moe_coop_kernel_b_k6(cb, wide);
        case 7: return exl3_moe_coop_kernel_b_k7(cb, wide);
        default: return exl3_moe_coop_kernel_b_k8(cb, wide);
    }
}

// Persistent single-launch kernel (both stages, wide tile only): gate/up and down share one bit
// width (and half-rate flag)
static MoeCoopKernel moe_coop_kernel_f(float K_, int cb, int Hi)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_f_h1(Hi);
            case 2: return exl3_moe_coop_kernel_f_h2(Hi);
            default: return exl3_moe_coop_kernel_f_h3(Hi);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_f_k1(cb, Hi);
        case 2: return exl3_moe_coop_kernel_f_k2(cb, Hi);
        case 3: return exl3_moe_coop_kernel_f_k3(cb, Hi);
        case 4: return exl3_moe_coop_kernel_f_k4(cb, Hi);
        case 5: return exl3_moe_coop_kernel_f_k5(cb, Hi);
        case 6: return exl3_moe_coop_kernel_f_k6(cb, Hi);
        case 7: return exl3_moe_coop_kernel_f_k7(cb, Hi);
        default: return exl3_moe_coop_kernel_f_k8(cb, Hi);
    }
}

// EXL3_MOE_COOP_FUSED: 1 forces the persistent kernel wherever it applies (testing), 0 the two
// launches. Unset (2 = auto): persistent kernel on sm_60 and sm_75 (measured there) when the
// gate/up stage fills the device
static int moe_coop_fused_mode(int device)
{
    static int mode = -2;
    if (mode == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_FUSED");
        mode = env ? atoi(env) : -1;
    }
    if (mode >= 0) return mode ? 1 : 0;
    static int sm[MAX_DEVICES] = {};
    if (!sm[device])
    {
        int major = 0, minor = 0;
        cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        cuda_check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        sm[device] = major * 10 + minor;
    }
    return (sm[device] == 60 || sm[device] == 75) ? 2 : 0;
}

// Resident blocks per SM of a kernel (once per device, kernel and shared memory size: the host stub
// is shared by every device, the register count is not)
static int moe_coop_occupancy(void* kernel, int smem, int device)
{
    static std::map<std::tuple<int, void*, int>, int> cache;
    auto key = std::make_tuple(device, kernel, smem);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    int n = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, MOE_COOP_THREADS, smem));
    n = std::max(n, 1);
    cache[key] = n;
    return n;
}

// Split-k factor: blocks per column chunk, partial outputs summed by the last-arriving block.
// Measured ineffective.
// Split-k factor for one stage. The grid is slots x column tiles, which fills the device for a
// routed launch (bsz x top-k slots) but not for a single-slot launch such as the shared expert
// (one expert's tiles on a 100+ SM device): split k until there is a block per SM, keeping at
// least 64 k-slices (1024 columns) per split so the partials and their reduction stay cheap.
// EXL3_MOE_COOP_KSPLIT forces one factor for every stage (testing)
static int moe_coop_pick_ksplit(int base, int kslices, int max_split, int device)
{
    static int forced = -2;
    if (forced == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_KSPLIT");
        forced = env ? atoi(env) : 0;
    }
    if (forced > 0) return std::min(forced, max_split);
    const int want = CEIL_DIVIDE(DevCtx::instance().get_num_sms(device), std::max(base, 1));
    const int kcap = std::max(kslices / 64, 1);
    return std::max(1, std::min(std::min(want, kcap), max_split));
}

// Tile geometry per stage: the wide (128-column, 4-way k-split) tile for long k, the narrow one
// otherwise. EXL3_MOE_COOP_WIDE = 0 / 1 forces one geometry for every stage (testing)
static int moe_coop_wide_mode()
{
    static int mode = -2;
    if (mode == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_WIDE");
        mode = env ? atoi(env) : -1;
    }
    return mode;
}

static bool moe_coop_pick_wide(int kslices, int slots, int device)
{
    const int mode = moe_coop_wide_mode();
    if (mode >= 0) return mode != 0;
    if (DevCtx::instance().get_cc(device) < CC_BLACKWELL) return true;
    return kslices >= 256 || (kslices >= 128 && slots >= 32);
}

// Dynamic shared memory above the 48 KB default needs an opt-in per kernel (once per process)
static void moe_coop_smem_optin(void* kernel, int smem)
{
    if (smem <= 48 * 1024) return;
    static std::map<void*, int> done;
    auto it = done.find(kernel);
    if (it != done.end() && it->second >= smem) return;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    done[kernel] = smem;
}

// Geometry of one call (both paths): tile widths, split-k factors (into p) and the two-launch grids
struct MoeCoopPlan
{
    bool wide_a, wide_b;
    int grid_a, grid_b;
};

static MoeCoopPlan moe_coop_plan(MoeCoopParams& p, int device)
{
    { static int dbg = std::getenv("EXL3_MOE_COOP_DBG") ? atoi(std::getenv("EXL3_MOE_COOP_DBG")) : 0; p.dbg = dbg; }
    MoeCoopPlan pl;
    const int slots = p.bsz * p.topk;
    const int nproj = p.gated ? 2 : 1;
    p.a_global = p.bsz > 1;
    pl.wide_a = moe_coop_pick_wide(p.Hi / 16, slots, device);
    pl.wide_b = moe_coop_pick_wide(p.I / 16, slots, device);
    const int base_a = slots * nproj * (p.I / (pl.wide_a ? 128 : MOE_COOP_COLS));
    const int base_b = slots * (p.Ho / (pl.wide_b ? 128 : MOE_COOP_COLS));
    const int max_split = std::max(1, p.slots_max / slots);
    p.ksplit_a = moe_coop_pick_ksplit(base_a, p.Hi / 16, max_split, device);
    p.ksplit_b = moe_coop_pick_ksplit(base_b, p.I / 16, max_split, device);
    pl.grid_a = base_a * p.ksplit_a;
    pl.grid_b = base_b * p.ksplit_b;
    p.f_items_a = pl.grid_a;
    p.f_items_b = pl.grid_b;
    return pl;
}

static void moe_coop_rot(const MoeCoopParams& p, cudaStream_t stream)
{
    const int rot_items = p.bsz * p.topk * (p.Hi / 128) * (p.gated ? 2 : 1);
    const int rot_grid = CEIL_DIVIDE(rot_items, MOE_COOP_THREADS / 32);
    exl3_moe_coop_ns::exl3_moe_coop_rot_kernel<<<rot_grid, MOE_COOP_THREADS, 0, stream>>>(p);
}

void exl3_moe_coop_launch(const MoeCoopParams& p_in, float K_gu, float K_d, int cb, int device, cudaStream_t stream)
{
    MoeCoopParams p = p_in;
    const int slots = p.bsz * p.topk;
    if (!slots)
    {
        if (p.bsz && p.H_out)
        {
            exl3_moe_coop_empty_kernel<<<dim3(p.bsz, CEIL_DIVIDE(p.H_out, 128)), 32, 0, stream>>>(p);
            cuda_check(cudaPeekAtLastError());
        }
        return;
    }
    const MoeCoopPlan pl = moe_coop_plan(p, device);
    const bool wide_a = pl.wide_a, wide_b = pl.wide_b;
    const int grid_a = pl.grid_a, grid_b = pl.grid_b;
    MoeCoopKernel ka = moe_coop_kernel_a(K_gu, cb, p.Hi, wide_a);
    MoeCoopKernel kb = moe_coop_kernel_b(K_d, cb, wide_b);
    moe_coop_smem_optin(ka.kernel, ka.smem);
    moe_coop_smem_optin(kb.kernel, kb.smem);

    // The persistent kernel pays off when the gate/up stage alone fills the device, so its tail can
    // overlap down items (measured: the 16 + 16 block shared-expert launch gets slower)
    const int fmode = moe_coop_fused_mode(device);
    bool fused = p.f_ok && wide_a && wide_b && K_gu == K_d && fmode;
    MoeCoopKernel kf = {};
    int resident = 0;
    if (fused)
    {
        kf = moe_coop_kernel_f(K_gu, cb, p.Hi);
        moe_coop_smem_optin(kf.kernel, kf.smem);
        resident = moe_coop_occupancy(kf.kernel, kf.smem, device) * DevCtx::instance().get_num_sms(device);
        fused = fmode == 1 || grid_a >= resident;
    }

    void* args[] = { (void*) &p };
    if (p.a_global) moe_coop_rot(p, stream);
    if (fused)
    {
        // One block per resident slot: blocks loop over the queue, so more would only wait
        const int grid_f = std::min(resident, grid_a + grid_b);
        cuda_check(cudaLaunchKernel(kf.kernel, dim3(grid_f), dim3(MOE_COOP_THREADS), args, kf.smem, stream));
        cuda_check(cudaPeekAtLastError());
        return;
    }
    cuda_check(cudaLaunchKernel(ka.kernel, dim3(grid_a), dim3(MOE_COOP_THREADS), args, ka.smem, stream));
    cuda_check(cudaLaunchKernel(kb.kernel, dim3(grid_b), dim3(MOE_COOP_THREADS), args, kb.smem, stream));
    cuda_check(cudaPeekAtLastError());
}

// Dual kernel for (routed, shared) widths; null when not instantiated
static MoeCoopKernel moe_coop_kernel_d(float K_r, float K_s, int cb, int Hi)
{
    const BitsK br = bits_from_K(K_r), bs = bits_from_K(K_s);
    if (br.half || bs.half) return { nullptr, 0 };
    switch (br.bits)
    {
        case 2: return exl3_moe_coop_kernel_d_r2(bs.bits, cb, Hi);
        case 3: return exl3_moe_coop_kernel_d_r3(bs.bits, cb, Hi);
        case 4: return exl3_moe_coop_kernel_d_r4(bs.bits, cb, Hi);
        default: return { nullptr, 0 };
    }
}

void exl3_moe_coop_launch_pair
(
    const MoeCoopParams& p_in, float K_gu, float K_d, int cb,
    const MoeCoopParams& ps_in, float Ks_gu, float Ks_d, int cbs,
    int device, cudaStream_t stream
)
{
    MoeCoopParams p = p_in, ps = ps_in;
    const int fmode = moe_coop_fused_mode(device);
    bool dual = fmode && p.f_ok && ps.f_ok && p.bsz * p.topk > 0 && ps.bsz * ps.topk > 0 &&
                K_gu == K_d && Ks_gu == Ks_d && cb == cbs && p.Hi == ps.Hi && p.sh_out;
    MoeCoopKernel kd = {};
    int resident = 0;
    MoeCoopPlan pl = {}, pls = {};
    if (dual)
    {
        kd = moe_coop_kernel_d(K_gu, Ks_gu, cb, p.Hi);
        dual = kd.kernel != nullptr;
    }
    if (dual)
    {
        pl = moe_coop_plan(p, device);
        pls = moe_coop_plan(ps, device);
        dual = pl.wide_a && pl.wide_b && pls.wide_a && pls.wide_b;
    }
    if (dual)
    {
        moe_coop_smem_optin(kd.kernel, kd.smem);
        resident = moe_coop_occupancy(kd.kernel, kd.smem, device) * DevCtx::instance().get_num_sms(device);
        dual = fmode == 1 || pl.grid_a >= resident;
    }
    if (!dual)
    {
        exl3_moe_coop_launch(ps_in, Ks_gu, Ks_d, cbs, device, stream);
        exl3_moe_coop_launch(p_in, K_gu, K_d, cb, device, stream);
        return;
    }
    p.f_sh_done = ps.f_queue;
    p.f_sh_wait = pls.grid_b;
    const char* rp_env = std::getenv("EXL3_MOE_ROT_PAIR");   // per call: tests compare both forms in one process
    const bool rot_pair = !rp_env || std::atoi(rp_env) != 0;   // default on, EXL3_MOE_ROT_PAIR=0 disables
    if (rot_pair && ps.a_global && p.a_global)
    {
        // One launch for both rotation pre-kernels (bit-identical: same per-warp work)
        auto rgrid = [](const MoeCoopParams& q)
        {
            const int items = q.bsz * q.topk * (q.Hi / 128) * (q.gated ? 2 : 1);
            return CEIL_DIVIDE(items, MOE_COOP_THREADS / 32);
        };
        const int g0 = rgrid(ps), g1 = rgrid(p);
        exl3_moe_coop_ns::exl3_moe_coop_rot2_kernel<<<g0 + g1, MOE_COOP_THREADS, 0, stream>>>(ps, p, g0);
    }
    else
    {
        if (ps.a_global) moe_coop_rot(ps, stream);
        if (p.a_global) moe_coop_rot(p, stream);
    }
    void* args[] = { (void*) &p, (void*) &ps };
    const int grid = std::min(resident, pl.grid_a + pl.grid_b + pls.grid_a + pls.grid_b);
    cuda_check(cudaLaunchKernel(kd.kernel, dim3(grid), dim3(MOE_COOP_THREADS), args, kd.smem, stream));
    cuda_check(cudaPeekAtLastError());
}

MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    float& K_gu, float& K_d, int& cb
)
{
    TORCH_CHECK_DTYPE(ctr, kInt);
    TORCH_CHECK(ctr.is_contiguous(), "exl3_moe_coop: ctr must be contiguous");
    for (const at::Tensor* t : { &g_trellis, &g_suh, &g_svh, &u_trellis, &u_suh, &u_svh, &d_trellis, &d_suh, &d_svh })
        TORCH_CHECK(t->scalar_type() == at::kLong && t->is_contiguous(), "exl3_moe_coop: pointer tables must be contiguous int64");
    TORCH_CHECK(!(mcg && mul1), "exl3_moe_coop: specified both mcg and mul1");
    TORCH_CHECK(Kg >= 1 && Kg <= 8 && Ku >= 1 && Ku <= 8 && Kd >= 1 && Kd <= 8, "exl3_moe_coop: K out of range");
    TORCH_CHECK(!gated || Kg == Ku, "exl3_moe_coop: gate and up must share a bit width");
    TORCH_CHECK(act >= 0 && act <= 3, "exl3_moe_coop: unknown activation");

    MoeCoopParams p = {};
    p.Hi = Hi;
    p.I = (int) gu_u.size(-1);
    p.Ho = (int) d_out.size(-1);
    p.H_out = (int) out.size(-1);
    TORCH_CHECK(p.Hi % 128 == 0 && p.I % 128 == 0 && p.Ho % 128 == 0 && p.H_out <= p.Ho, "exl3_moe_coop: Hi/I/Ho shape");
    TORCH_CHECK(p.Hi > 0 && p.I > 0 && p.Ho > 0, "exl3_moe_coop: empty Hi/I/Ho dimension");

    TORCH_CHECK_DTYPE(had_g, kHalf);
    TORCH_CHECK_DTYPE(had_u, kHalf);
    TORCH_CHECK_DTYPE(act_out, kHalf);
    TORCH_CHECK_DTYPE(d_out, kFloat);
    TORCH_CHECK_DTYPE(out, kFloat);
    p.gu_f32 = gu_u.scalar_type() == at::kFloat;
    if (!p.gu_f32) TORCH_CHECK_DTYPE(gu_u, kHalf);
    TORCH_CHECK(gu_g.scalar_type() == gu_u.scalar_type(), "exl3_moe_coop: gate/up scratch dtype mismatch");
    // Scratch capacity in slots: the smallest of the per-slot buffers
    const int slots_max = (int) std::min({ had_g.numel() / p.Hi, had_u.numel() / p.Hi, gu_g.numel() / p.I,
                                           gu_u.numel() / p.I, act_out.numel() / p.I, d_out.numel() / p.Ho });
    auto check_scratch = [&] (const at::Tensor& t, int width)
    {
        TORCH_CHECK(t.is_contiguous() && t.size(-1) == width, "exl3_moe_coop: scratch shape");
    };
    check_scratch(had_g, p.Hi);
    check_scratch(had_u, p.Hi);
    check_scratch(gu_g, p.I);
    check_scratch(gu_u, p.I);
    check_scratch(act_out, p.I);
    check_scratch(d_out, p.Ho);
    TORCH_CHECK(out.dim() >= 2 && out.stride(-1) == 1, "exl3_moe_coop: out shape");
    TORCH_CHECK(slots_max >= 1 && slots_max <= 256, "exl3_moe_coop: scratch must hold 1..256 slots");

    // Counters: the gate/up stage's block covers every slot the scratch can hold, the down
    // stage's every token row the output can hold
    const int rows_max = (int) out.size(-2);
    p.slots_max = slots_max;
    p.ctr_a_len = slots_max * (p.I / 128);
    p.ctr_b_len = rows_max * (p.Ho / 128);
    TORCH_CHECK(ctr.numel() >= exl3_moe_coop_ctr_len(slots_max, rows_max, p.I, p.Ho), "exl3_moe_coop: counter scratch too small");
    p.ctr_a = (int*) ctr.data_ptr();
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.rows_max = rows_max;
    p.f_ok = ctr.numel() >= exl3_moe_coop_ctr_len_fused(slots_max, rows_max, p.I, p.Ho);
    p.f_queue = p.runs + 2 + (slots_max + 1) + slots_max;
    p.f_ready = p.f_queue + 2;

    p.g_trellis = (const int64_t*) g_trellis.data_ptr();
    p.g_suh = (const int64_t*) g_suh.data_ptr();
    p.g_svh = (const int64_t*) g_svh.data_ptr();
    p.u_trellis = (const int64_t*) u_trellis.data_ptr();
    p.u_suh = (const int64_t*) u_suh.data_ptr();
    p.u_svh = (const int64_t*) u_svh.data_ptr();
    p.d_trellis = (const int64_t*) d_trellis.data_ptr();
    p.d_suh = (const int64_t*) d_suh.data_ptr();
    p.d_svh = (const int64_t*) d_svh.data_ptr();
    p.g_bias = g_bias ? (const int64_t*) g_bias->data_ptr() : nullptr;
    p.u_bias = u_bias ? (const int64_t*) u_bias->data_ptr() : nullptr;
    p.d_bias = d_bias ? (const int64_t*) d_bias->data_ptr() : nullptr;
    p.n_local = (int) u_trellis.size(0);
    p.act = act;
    p.act_limit = act_limit;
    p.gated = gated;

    p.had_g = (half*) had_g.data_ptr();
    p.had_u = (half*) had_u.data_ptr();
    p.gu_g = gu_g.data_ptr();
    p.gu_u = gu_u.data_ptr();
    p.act_out = (half*) act_out.data_ptr();
    p.d_out = (float*) d_out.data_ptr();
    p.out = (float*) out.data_ptr();
    p.out_stride = (int) out.stride(-2);

    if (sh_gate_w)
    {
        TORCH_CHECK_DTYPE(sh_gate_w.value(), kHalf);
        TORCH_CHECK(sh_gate_w->is_contiguous(), "exl3_moe_coop: shared gate weight must be contiguous");
        p.sh_gate_w = (const half*) sh_gate_w->data_ptr();
        p.sh_gate_n = (int) sh_gate_w->numel();
    }

    K_gu = gated ? Kg : Ku;
    K_d = Kd;
    cb = mcg ? 1 : (mul1 ? 2 : 0);
    return p;
}

void exl3_moe_coop_run
(
    MoeCoopParams p, float K_gu, float K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();
    exl3_moe_coop_bind(p, x, sel, rw, sh_out);
    exl3_moe_coop_launch(p, K_gu, K_d, cb, device, stream);
}

void exl3_moe_coop_bind
(
    MoeCoopParams& p,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
)
{
    // Per-call checks kept to what can actually vary between calls
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "exl3_moe_coop: x must be (bsz, H) with unit column stride");
    p.x = (const half*) x.data_ptr();
    p.x_stride = (int) x.stride(0);
    p.bsz = (int) x.size(0);
    p.H = (int) x.size(1);
    p.topk = (int) sel.size(-1);
    p.sel = (const int64_t*) sel.data_ptr();
    p.rw = (const half*) rw.data_ptr();
    const int slots = p.bsz * p.topk;
    TORCH_CHECK(p.H % 4 == 0 && p.H <= p.Hi, "exl3_moe_coop: H/Hi shape");
    TORCH_CHECK(slots <= p.slots_max && p.bsz <= p.rows_max, "exl3_moe_coop: batch exceeds the scratch");
    TORCH_CHECK(p.min_expert < 0 || p.max_expert - p.min_expert <= p.n_local, "exl3_moe_coop: expert range exceeds tables");
    if (sh_out)
    {
        TORCH_CHECK(p.H_out == p.H, "exl3_moe_coop: shared expert output width must match the routed output");
        p.sh_out = (const float*) sh_out->data_ptr();
        if (p.sh_gate_w) TORCH_CHECK(p.sh_gate_n == p.H && p.H % 2 == 0, "exl3_moe_coop: shared gate width");
    }
    else
        p.sh_gate_w = nullptr;
}

void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(rw, kHalf);
    TORCH_CHECK(sel.is_contiguous() && rw.is_contiguous(), "exl3_moe_coop: sel/rw must be contiguous");
    TORCH_CHECK(sel.sizes() == rw.sizes() && sel.dim() == 2 && sel.size(0) == x.size(0), "exl3_moe_coop: sel/rw must be (bsz, topk)");
    if (sh_out)
    {
        TORCH_CHECK_DTYPE(sh_out.value(), kFloat);
        TORCH_CHECK(sh_out->is_contiguous() && sh_out->size(-1) == x.size(1) && sh_out->numel() >= x.size(0) * x.size(1),
                    "exl3_moe_coop: shared expert output shape");
    }
    float K_gu, K_d; int cb;
    MoeCoopParams p = exl3_moe_coop_prepare(Hi, g_trellis, g_suh, g_svh, u_trellis, u_suh, u_svh, d_trellis, d_suh, d_svh,
                                            g_bias, u_bias, d_bias, Kg, Ku, Kd, mcg, mul1, act, act_limit, gated,
                                            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out, sh_gate_w, K_gu, K_d, cb);
    p.min_expert = min_expert;
    p.max_expert = max_expert;
    exl3_moe_coop_run(p, K_gu, K_d, cb, x, sel, rw, sh_out);
}

void exl3_moe_coop_pair
(
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw, int Hi,
    const std::vector<at::Tensor>& tabs, float K, const std::vector<at::Tensor>& scratch,
    const std::vector<at::Tensor>& tabs_sh, float K_sh, const std::vector<at::Tensor>& scratch_sh,
    bool mcg, bool mul1,
    at::Tensor& out, at::Tensor& sh_buf, const at::Tensor& sh_gate_w
)
{
    TORCH_CHECK(tabs.size() == 9 && tabs_sh.size() == 9 && scratch.size() == 7 && scratch_sh.size() == 7, "exl3_moe_coop_pair: argument lists");
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();
    const int bsz = (int) x.size(0);
    auto prep = [&] (const std::vector<at::Tensor>& t, float K_, const std::vector<at::Tensor>& sc, at::Tensor& o,
                     const c10::optional<at::Tensor>& gw, float& Kgu, float& Kd, int& cb_)
    {
        at::Tensor hg = sc[0], hu = sc[1], gg = sc[2], gu = sc[3], ao = sc[4], dd = sc[5], ct = sc[6];
        return exl3_moe_coop_prepare(Hi, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8],
                                     c10::nullopt, c10::nullopt, c10::nullopt, K_, K_, K_, mcg, mul1,
                                     MOE_COOP_ACT_SILU, 0.0f, true, hg, hu, gg, gu, ao, dd, ct, o, gw, Kgu, Kd, cb_);
    };
    float Kgu, Kd, Ksgu, Ksd; int cb, cbs;
    MoeCoopParams p = prep(tabs, K, scratch, out, sh_gate_w, Kgu, Kd, cb);
    MoeCoopParams ps = prep(tabs_sh, K_sh, scratch_sh, sh_buf, c10::nullopt, Ksgu, Ksd, cbs);
    p.min_expert = p.max_expert = -1;
    ps.min_expert = ps.max_expert = -1;
    at::Tensor sh_sel = at::zeros({bsz, 1}, x.options().dtype(at::kLong));
    at::Tensor sh_rw = at::ones({bsz, 1}, x.options());
    at::Tensor sh_o = sh_buf.slice(0, 0, bsz);
    exl3_moe_coop_bind(ps, x, sh_sel, sh_rw, c10::nullopt);
    exl3_moe_coop_bind(p, x, sel, rw, sh_o);
    exl3_moe_coop_launch_pair(p, Kgu, Kd, cb, ps, Ksgu, Ksd, cbs, device, stream);
}
