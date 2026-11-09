"""
Pascal (sm_60) m = 5..8 GEMV, MMODE 6 of exl3_gemv_kernel (quant/exl3_gemv_kernel.cuh): the MMODE 4 arithmetic
("one output per lane", row lane % 4, column lane / 4 per n8 fragment) extended to two row groups per lane (rows
lane % 4 and lane % 4 + 4), with the B columns of the lane's quad gathered once and reused by both groups.

Contract: bit-identical to MMODE 1 (EXL3_GEMV_M58=0, read per call) for every eligible shape it is enabled for -
the per-output HFMA2 chain, pair sum, fp16 run sum and fold cadence are the emulated MMA's - and within the GEMV
tolerance of the float64 reference (see test_exl3_gemv.py). MMODE 6 runs in the wide kernel configuration for
every eligible rate and in the narrow one for the staged 5 / 6 bpw paths; 4 bpw narrow is 5-10% slower at m = 8
and stays on MMODE 1 even with the knob on (exl3_gemv.cu). Other m, other architectures, half-integer rates and
codebooks other than mul1 do not take MMODE 6 either (the switch changes nothing).

The decode graphs of the model forward capture the GEMV launches at load time, so EXL3_GEMV_M58 has to be set
before the model is loaded for a model-level A/B; here it is set per call, which the kernel-level dispatch honors.
"""

import os

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.env import compute_capability, is_rocm
from testlib.exl3 import generator, rand_scale

# All wide-config (cfg1: n > 8192 or n > k) shapes: the only ones MMODE 6 is enabled for. K = 4 body class,
# 5 bpw lm_head class (staged words), 6 bpw MTP class, both output dtypes.
CASES = [
    # K, cb, m, k, n, fp32 out
    (4, "mul1", 8, 512, 8320, False),       # 4 bpw gate/up class (n > 8192)
    (4, "mul1", 5, 2048, 8320, True),       # 4 bpw, fp32 out
    (4, "mul1", 7, 1024, 8320, False),      # 4 bpw, m = 7
    (5, "mul1", 8, 512, 8320, False),       # 5 bpw lm_head class, staged words
    (5, "mul1", 6, 1024, 8320, True),       # 5 bpw, fp32 out
    (5, "mul1", 5, 256, 1024, False),       # 5 bpw, n <= 8192 but n > k (wide by the second rule)
    (6, "mul1", 5, 512, 1024, False),       # 6 bpw MTP head class
    (6, "mul1", 7, 256, 8320, True),        # 6 bpw, fp32 out
]

# cfg0 (narrow) shapes at K = 5 / 6: MMODE 6 is enabled there (2.6-6% faster, the staged-word paths), so these
# take the MMODE 6 / MMODE 1 bit-identity test too
NARROW_K56 = [
    (5, "mul1", 8, 5120, 4096, False),      # MTP head class
    (5, "mul1", 6, 6144, 5120, True),       # o_proj class, fp32 out
    (6, "mul1", 8, 5120, 4096, False),
    (6, "mul1", 7, 6144, 5120, True),
]

# cfg0 shapes at K = 4: the gate keeps them on MMODE 1, so the knob must not change the result
NARROW_K4 = [
    (4, "mul1", 8, 6144, 5120, False),      # o_proj class
    (4, "mul1", 8, 17408, 5120, True),      # down_proj class, fp32 out
    (4, "mul1", 5, 5120, 1024, False),      # k/v_proj class
]


def _supported(device):
    """MMODE 6 is sm_60 only (the emulated MMA and its one-output-per-lane form are Pascal-specific)"""
    if is_rocm():
        return False
    return compute_capability(device) == (6, 0)


def _set_m58(m58):
    old = os.environ.get("EXL3_GEMV_M58")
    os.environ["EXL3_GEMV_M58"] = "1" if m58 else "0"
    return old


def _restore_m58(old):
    if old is None: del os.environ["EXL3_GEMV_M58"]
    else: os.environ["EXL3_GEMV_M58"] = old


def _run(A, tr, suh, svh, n, cb, fp32, m58):
    m = A.shape[0]
    rows = 2 * n
    big = torch.full((m * n + 2 * rows,), 7.0, dtype = torch.float if fp32 else torch.half, device = A.device)
    C = big[rows : rows + m * n].view(m, n)
    A_had = torch.empty_like(A)
    old = _set_m58(m58)
    try:
        ext.exl3_gemv(A, tr, C, suh, A_had, svh, cb == "mcg", cb == "mul1")
    finally:
        _restore_m58(old)
    torch.cuda.synchronize(A.device)
    assert (big[:rows] == 7.0).all() and (big[-rows:] == 7.0).all(), "wrote outside C"
    return C


def _weights(K, k, n, seed):
    g = generator(seed)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)).reshape(k // 16, n // 16, -1)
    suh, svh = rand_scale(k, g), rand_scale(n, g)
    return p, suh, svh, g


@pytest.mark.parametrize("K, cb, m, k, n, fp32", CASES + NARROW_K56)
@torch.inference_mode()
def test_m58_bit_identical_to_mma_path(device, K, cb, m, k, n, fp32):
    if not _supported(device):
        pytest.skip("MMODE 6 is Pascal (sm_60) only")
    seed = m * 1000 + k + n + K
    p, suh, svh, g = _weights(K, k, n, seed)
    A = (torch.randn(m, k, generator = g) * 0.5).half()
    tr = torch.from_numpy(p.copy()).to(device)
    Ad, suhd, svhd = A.to(device), suh.to(device), svh.to(device)

    C_new = _run(Ad, tr, suhd, svhd, n, cb, fp32, True)
    C_old = _run(Ad, tr, suhd, svhd, n, cb, fp32, False)
    assert torch.equal(C_new, C_old),         f"MMODE 6 differs from MMODE 1: max |diff| {(C_new.float() - C_old.float()).abs().max().item():.3e}"
    assert torch.equal(C_new, _run(Ad, tr, suhd, svhd, n, cb, fp32, True)), "repeat launch differs"

    ref = tref.linear(A, p, suh, svh, K, cb)
    e = C_new.cpu().double() - ref
    rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    rel_max = (e.abs().max() / ref.abs().max()).item()
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"relative RMS {rel_rms:.2e}, max {rel_max:.2e}"


@pytest.mark.parametrize("m", [5, 6, 7, 8])
@torch.inference_mode()
def test_m58_rows_beyond_m_untouched(device, m):
    """Zero-padding A to 8 rows leaves the first m output rows unchanged (the rows past m load no A)"""
    if not _supported(device):
        pytest.skip("MMODE 6 is Pascal (sm_60) only")
    K, cb, k, n = 4, "mul1", 1024, 8320
    p, suh, svh, g = _weights(K, k, n, 7 + m)
    tr = torch.from_numpy(p.copy()).to(device)
    suhd, svhd = suh.to(device), svh.to(device)
    A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
    A8 = torch.zeros(8, k, dtype = torch.half, device = device)
    A8[:m] = A
    C = _run(A, tr, suhd, svhd, n, cb, False, True)
    C8 = _run(A8, tr, suhd, svhd, n, cb, False, True)
    assert torch.equal(C, C8[:m])


@pytest.mark.parametrize("K, cb, m, k, n, fp32", NARROW_K4)
@torch.inference_mode()
def test_m58_narrow_k4_unchanged(device, K, cb, m, k, n, fp32):
    """4 bpw cfg0 shapes keep MMODE 1 (5-10% slower with MMODE 6), so the knob cannot change them"""
    if not _supported(device):
        pytest.skip("MMODE 6 is Pascal (sm_60) only")
    p, suh, svh, g = _weights(K, k, n, 31 + K)
    A = (torch.randn(m, k, generator = g) * 0.5).half()
    tr = torch.from_numpy(p.copy()).to(device)
    Ad, suhd, svhd = A.to(device), suh.to(device), svh.to(device)
    C_on = _run(Ad, tr, suhd, svhd, n, cb, fp32, True)
    C_off = _run(Ad, tr, suhd, svhd, n, cb, fp32, False)
    assert torch.equal(C_on, C_off)


@pytest.mark.parametrize("K, cb, m, k, n, fp32", CASES[:1])
@torch.inference_mode()
def test_m58_disabled_is_mmode1(device, K, cb, m, k, n, fp32):
    """EXL3_GEMV_M58=0 takes the MMODE 1 path: the result must match the enabled path bit for bit (same arithmetic)"""
    if not _supported(device):
        pytest.skip("MMODE 6 is Pascal (sm_60) only")
    p, suh, svh, g = _weights(K, k, n, 11)
    A = (torch.randn(m, k, generator = g) * 0.5).half()
    tr = torch.from_numpy(p.copy()).to(device)
    Ad, suhd, svhd = A.to(device), suh.to(device), svh.to(device)
    assert torch.equal(_run(Ad, tr, suhd, svhd, n, cb, fp32, True), _run(Ad, tr, suhd, svhd, n, cb, fp32, False))


def _launched_gemv(A, tr, suh, svh, n, cb):
    """Kernel names of the GEMV launches of one call (the dispatch test: which MMODE was actually selected)"""
    from torch.profiler import profile, ProfilerActivity
    m = A.shape[0]
    C = torch.empty(m, n, dtype = torch.half, device = A.device)
    A_had = torch.empty_like(A)
    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        ext.exl3_gemv(A, tr, C, suh, A_had, svh, cb == "mcg", cb == "mul1")
        torch.cuda.synchronize(A.device)
    return [e.key for e in prof.key_averages() if "exl3_gemv_kernel" in e.key]


@torch.inference_mode()
def test_m58_dispatch_gate(device):
    """Wide shapes take MMODE 6 with the knob on, 5 / 6 bpw narrow shapes too, 4 bpw narrow shapes keep MMODE 1"""
    if not _supported(device):
        pytest.skip("MMODE 6 is Pascal (sm_60) only")
    g = generator(5)
    for (K, k, n), want6 in (((4, 512, 8320), True), ((4, 6144, 5120), False), ((5, 5120, 4096), True)):
        p, suh, svh, g = _weights(K, k, n, k + n + K)
        A = (torch.randn(8, k, generator = g) * 0.5).half().to(device)
        tr = torch.from_numpy(p.copy()).to(device)
        suhd, svhd = suh.to(device), svh.to(device)
        old = _set_m58(True)
        try:
            names = _launched_gemv(A, tr, suhd, svhd, n, "mul1")
        finally:
            _restore_m58(old)
        assert names, "no exl3_gemv_kernel launch seen"
        mmode = {int(nm.split("<")[1].split(",")[3]) for nm in names}
        assert mmode == ({6} if want6 else {1}), f"K{K} k{k} n{n}: launched MMODE {mmode} ({names})"
