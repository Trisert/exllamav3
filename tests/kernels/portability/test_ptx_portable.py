"""
The C++ fallbacks in exllamav3_ext/ptx_portable.cuh stand in for inline PTX on ROCm. On an NVIDIA device this
compiles a small kernel that evaluates every fallback next to the PTX instruction it replaces, on random and edge-
case inputs, and requires bit-identical results. On ROCm (no PTX) the fallbacks are checked against a NumPy
reference of the PTX semantics instead.
"""

import hashlib
import os
import shutil

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

import exllamav3

EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(exllamav3.__file__)), "exllamav3_ext")
HIP = torch.version.hip is not None

CUDA_SRC = r"""
#include <torch/extension.h>
#include "ptx_portable.cuh"

// Output columns, per input row: 0 lop3, 1 shf, 2 bfe16, 3 bfe64, 4 mul_lo, 5 mul_hi, 6 dp4a.
// cols 0..6 from the fallbacks, cols 7..13 from PTX (CUDA only, zero on ROCm)
__global__ void ptx_portable_kernel(const uint32_t* a, const uint32_t* b, const uint32_t* c, const int* p, const int* l,
                                    uint32_t* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = a[i], y = b[i], z = c[i];
    int pos = p[i], len = l[i];
    uint32_t* o = out + i * 14;
    uint64_t v64 = (static_cast<uint64_t>(y) << 32) | x;
    o[0] = exl3_lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
    o[1] = exl3_shf_r_wrap(x, y, pos & 31);
    o[2] = exl3_bfe_u32_16(x, pos & 15);
    o[3] = exl3_bfe_u64(v64, pos, len);
    o[4] = exl3_mul_lo_u32(x, y);
    o[5] = exl3_mul_hi_u32(x, y);
    o[6] = static_cast<uint32_t>(exl3_dp4a_us(x, y, static_cast<int>(z)));
#if !defined(USE_ROCM)
    uint32_t r; uint64_t r64; int d;
    r = x; asm ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(r)); o[7] = r;
    asm ("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(pos & 31)); o[8] = r;
    asm ("bfe.u32 %0, %1, %2, 16;" : "=r"(r) : "r"(x), "r"(pos & 15)); o[9] = r;
    asm ("bfe.u64 %0, %1, %2, %3;" : "=l"(r64) : "l"(v64), "r"(pos), "r"(len)); o[10] = static_cast<uint32_t>(r64);
    asm ("mul.lo.u32 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(y)); o[11] = r;
    asm ("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(y)); o[12] = r;
    asm ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(x), "r"(y), "r"(static_cast<int>(z))); o[13] = static_cast<uint32_t>(d);
#endif
}

torch::Tensor run(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor p, torch::Tensor l)
{
    int n = a.numel();
    auto out = torch::zeros({n, 14}, a.options());
    ptx_portable_kernel<<<(n + 255) / 256, 256>>>(
        (const uint32_t*) a.data_ptr(), (const uint32_t*) b.data_ptr(), (const uint32_t*) c.data_ptr(),
        p.data_ptr<int>(), l.data_ptr<int>(), (uint32_t*) out.data_ptr(), n);
    return out;
}
"""

CPP_SRC = "torch::Tensor run(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor p, torch::Tensor l);"

NAMES = ["lop3_6a", "shf_r_wrap", "bfe_u32_16", "bfe_u64", "mul_lo_u32", "mul_hi_u32", "dp4a_us"]


def reference(a, b, c, pos, length):
    """NumPy model of the PTX instructions (for the ranges the kernels use)"""
    a64, b64 = a.astype(np.uint64), b.astype(np.uint64)
    v = (b64 << np.uint64(32)) | a64
    ref = np.zeros((len(a), 7), dtype = np.uint64)
    ref[:, 0] = (a & np.uint32(0x8fff8fff)) ^ np.uint32(0x3b603b60)
    ref[:, 1] = (v >> (pos & 31).astype(np.uint64)) & np.uint64(0xffffffff)
    ref[:, 2] = (a >> (pos & 15).astype(np.uint32)) & np.uint32(0xffff)
    for i in range(len(a)):
        p, n = int(pos[i]), int(length[i])
        x = 0 if (n <= 0 or p >= 64) else (int(v[i]) >> p) & ((1 << n) - 1 if n < 64 else (1 << 64) - 1)
        ref[i, 3] = x & 0xffffffff
    ref[:, 4] = (a64 * b64) & np.uint64(0xffffffff)
    ref[:, 5] = (a64 * b64) >> np.uint64(32)
    ab = a.view(np.uint8).reshape(-1, 4).astype(np.int64)
    bb = b.view(np.int8).reshape(-1, 4).astype(np.int64)
    ref[:, 6] = ((ab * bb).sum(-1) + c.view(np.int32).astype(np.int64)) & 0xffffffff
    return ref.astype(np.uint32)


DP4A_SIGN_SRC = r"""
#include <torch/extension.h>
#include "ptx_portable.cuh"

// exl3_dp4a_us only: no __dp4a builtin and no dp4a inline PTX, so this compiles (and runs) on
// every arch including sm_60, where the dp4a instruction does not exist.
__global__ void dp4a_us_kernel(const uint32_t* a, const uint32_t* b, const int* c, uint32_t* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<uint32_t>(exl3_dp4a_us(a[i], b[i], c[i]));
}

torch::Tensor run_dp4a(torch::Tensor a, torch::Tensor b, torch::Tensor c)
{
    int n = a.numel();
    auto out = torch::zeros({n}, a.options());
    dp4a_us_kernel<<<(n + 255) / 256, 256>>>(
        (const uint32_t*) a.data_ptr(), (const uint32_t*) b.data_ptr(),
        c.data_ptr<int>(), (uint32_t*) out.data_ptr(), n);
    return out;
}
"""

DP4A_SIGN_CPP = "torch::Tensor run_dp4a(torch::Tensor a, torch::Tensor b, torch::Tensor c);"


def test_dp4a_us_sign_extension(device, tmp_path):
    """exl3_dp4a_us must implement dp4a.u32.s32: unsigned bytes of a times SIGNED bytes of b.

    Edge vectors: b = 0x01010101 with a bytes 0x80/0xFF (the mul1 byte-sum call pattern used by
    codebook.cuh and exl3_gemv_kernel.cuh), plus a general section where b itself carries
    0x80/0xFF bytes so an unsigned*unsigned model cannot pass. Compared against the NumPy model
    with bb viewed as int8 (the same model as reference() column 6).
    """
    header = os.path.join(EXT_DIR, "ptx_portable.cuh")
    with open(header, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    inc = EXT_DIR
    if HIP:
        shutil.copy(header, tmp_path)
        inc = str(tmp_path)
    ext = load_inline(
        name = "ptx_portable_dp4a_test", cpp_sources = DP4A_SIGN_CPP,
        cuda_sources = f"// ptx_portable.cuh {digest}\n" + DP4A_SIGN_SRC,
        functions = ["run_dp4a"],
        extra_include_paths = [inc], verbose = False,
    )
    a_vals, b_vals, c_vals = [], [], []
    # Section 1: the kernels' call pattern, b = 0x01010101, a over 0x80/0xFF edge bytes
    edge_words = [0x00000000, 0x01010101, 0x7f7f7f7f, 0x80808080, 0xffffffff,
                 0x807f00ff, 0xff800100, 0x80000000, 0x00000080, 0x000000ff, 0x7f0080ff]
    for w in edge_words:
        a_vals.append(w); b_vals.append(0x01010101); c_vals.append(0)
    for w in edge_words:
        a_vals.append(w); b_vals.append(0x01010101); c_vals.append(0x6400)
    for w in [0x80808080, 0xffffffff, 0x807f00ff]:
        a_vals.append(w); b_vals.append(0x01010101); c_vals.append(-7)
    # Section 2: b itself carries 0x80/0xFF bytes (locks the signedness of the second operand)
    for b in [0x00000000, 0x01010101, 0x7f7f7f7f, 0x80808080, 0xffffffff, 0xff7f0180, 0x80017fff]:
        for a in [0x00000000, 0x01010101, 0x80808080, 0xffffffff, 0x7f0080ff]:
            a_vals.append(a); b_vals.append(b); c_vals.append(0x6400)
    a = np.array(a_vals, dtype = np.uint32)
    b = np.array(b_vals, dtype = np.uint32)
    c = np.array(c_vals, dtype = np.int32)
    n = len(a)
    pos = np.zeros(n, dtype = np.int32)
    length = np.zeros(n, dtype = np.int32)
    ref = reference(a, b, c, pos, length)[:, 6]
    t = lambda x: torch.from_numpy(x.view(np.int32)).to(device)
    out = ext.run_dp4a(t(a), t(b), t(c)).cpu().numpy().view(np.uint32)
    bad = np.nonzero(out != ref)[0]
    assert not len(bad), (
        f"exl3_dp4a_us sign-extension mismatch at row {bad[0]}: "
        f"a={a[bad[0]]:#010x} b={b[bad[0]]:#010x} c={c[bad[0]]} "
        f"got={out[bad[0]]:#010x} want={ref[bad[0]]:#010x}"
    )


def test_ptx_portable(device, tmp_path):
    # On ROCm the header is compiled from a copy, since torch hipifies the include directories in place. On CUDA the
    # source tree is included directly, which keeps the include path (part of the build hash) stable between runs.
    # The build hash covers the sources but not included headers, so the header's digest goes into the source
    header = os.path.join(EXT_DIR, "ptx_portable.cuh")
    with open(header, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    inc = EXT_DIR
    if HIP:
        shutil.copy(header, tmp_path)
        inc = str(tmp_path)
    ext = load_inline(
        name = "ptx_portable_test", cpp_sources = CPP_SRC, cuda_sources = f"// ptx_portable.cuh {digest}\n" + CUDA_SRC,
        functions = ["run"],
        extra_include_paths = [inc], verbose = False,
    )
    rng = np.random.default_rng(0)
    n = 1 << 16
    a = rng.integers(0, 1 << 32, n, dtype = np.uint64).astype(np.uint32)
    b = rng.integers(0, 1 << 32, n, dtype = np.uint64).astype(np.uint32)
    c = rng.integers(0, 1 << 32, n, dtype = np.uint64).astype(np.uint32)
    pos = rng.integers(0, 64, n).astype(np.int32)
    length = rng.integers(0, 33, n).astype(np.int32)
    edges = np.array([0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0x8fff8fff, 0x00ff00ff], dtype = np.uint32)
    k = len(edges)
    a[:k * k] = np.repeat(edges, k); b[:k * k] = np.tile(edges, k); c[:k] = edges
    pos[:4] = [0, 31, 32, 63]; length[:4] = [0, 32, 1, 32]
    t = lambda x: torch.from_numpy(x.view(np.int32)).to(device)
    out = ext.run(t(a), t(b), t(c), t(pos), t(length)).cpu().numpy().view(np.uint32)
    ref = reference(a, b, c, pos, length)
    for j, name in enumerate(NAMES):
        bad = np.nonzero(out[:, j] != ref[:, j])[0]
        assert not len(bad), f"{name} fallback differs from the PTX model at row {bad[0]}"
        if not HIP:
            bad = np.nonzero(out[:, j] != out[:, 7 + j])[0]
            assert not len(bad), f"{name} fallback differs from the PTX instruction at row {bad[0]}"
