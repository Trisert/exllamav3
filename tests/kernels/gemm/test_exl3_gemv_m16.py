"""
Pascal (sm_60) and Turing (sm_75) m = 9..16 GEMV, MMODE 5 of exl3_gemv_kernel (quant/exl3_gemv_kernel.cuh): the
MMODE 1 MMA path with the fragment's second row group live (A rows lane/4 + 8 in a01[1] / a23[1], C rows lane/4 + 8
in the high half of FragC_h), so rows 8..15 cost one more A load, one more fold accumulator and 16 instead of 8 smem
rows.

Contract: bit-identical to MMODE 1, i.e. to the same rows run as a separate call of 8 rows (each row's MMA, fold
cadence and cross-warp reduction are unchanged; the MMA has no cross-row terms), and within the GEMV tolerance of the
float64 reference (see test_exl3_gemv.py). Instances exist for K = 4 / 5 / 6 with mul1 only and the path is taken on
Pascal and Turing only; EXL3_GEMV_M16=0 restores the tile-GEMM fallback, so the forced entry point raises for m > 8.
"""

import os

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.env import compute_capability, is_rocm
from testlib.exl3 import generator, rand_scale

CASES = [
    # K, cb, m, k, n, fp32 out
    (4, "mul1", 13, 2048, 2048, False),      # narrow config
    (4, "mul1", 13, 1024, 8320, False),      # wide config, lm_head class
    (5, "mul1", 13, 2048, 2048, False),      # narrow config, Flash-Next attention / GDN projection class
    (5, "mul1", 16, 512, 8448, True),        # wide config
    (5, "mul1", 9, 256, 1024, False),
    (6, "mul1", 12, 512, 1024, False),       # MTP fc class
    (6, "mul1", 16, 256, 8320, True),        # wide config
]


def _supported(device, K):
    """The m = 9..16 path is instantiated for 4 / 5 / 6 bpw mul1 and taken on Pascal and Turing; Turing rejects
    K >= 5 before the row check (5 and 6 bpw staged tiles are measured on Pascal only)"""
    if is_rocm():
        return False
    cc = compute_capability(device)
    if cc == (6, 0):
        return True
    if cc == (7, 5):
        return K <= 4
    return False


def _run(A, tr, suh, svh, n, cb, fp32):
    m = A.shape[0]
    rows = 2 * n
    big = torch.full((m * n + 2 * rows,), 7.0, dtype = torch.float if fp32 else torch.half, device = A.device)
    C = big[rows : rows + m * n].view(m, n)
    A_had = torch.empty_like(A)
    ext.exl3_gemv(A, tr, C, suh, A_had, svh, cb == "mcg", cb == "mul1")
    torch.cuda.synchronize(A.device)
    assert (big[:rows] == 7.0).all() and (big[-rows:] == 7.0).all(), "wrote outside C"
    return C


def _run_split(A, tr, suh, svh, n, cb, fp32):
    """The same rows as two <= 8-row calls: rows 0..7 (MMODE 1) and rows 8..m-1 zero-padded to 8 rows (MMODE 1,
    so the arithmetic is the one MMODE 5 has to reproduce bit for bit)"""
    m = A.shape[0]
    lo = _run(A[:8].contiguous(), tr, suh, svh, n, cb, fp32)
    A_hi = torch.zeros(8, A.shape[1], dtype = A.dtype, device = A.device)
    A_hi[:m - 8] = A[8:]
    hi = _run(A_hi, tr, suh, svh, n, cb, fp32)
    return torch.cat((lo, hi[:m - 8]), dim = 0)


@pytest.mark.parametrize("K, cb, m, k, n, fp32", CASES)
@torch.inference_mode()
def test_m16_bit_identical_to_split_mma_calls(device, K, cb, m, k, n, fp32):
    if not _supported(device, K):
        pytest.skip("m = 9..16 GEMV is Pascal/Turing only, and K = 5 / 6 only on Pascal")
    seed = m * 1000 + k + n + K
    g = generator(seed)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)).reshape(k // 16, n // 16, -1)
    suh, svh = rand_scale(k, g), rand_scale(n, g)
    A = (torch.randn(m, k, generator = g) * 0.5).half()
    tr = torch.from_numpy(p.copy()).to(device)
    Ad, suhd, svhd = A.to(device), suh.to(device), svh.to(device)

    C16 = _run(Ad, tr, suhd, svhd, n, cb, fp32)
    C_split = _run_split(Ad, tr, suhd, svhd, n, cb, fp32)
    assert torch.equal(C16, C_split), \
        f"MMODE 5 differs from the 8-row MMODE 1 calls: max |diff| {(C16.float() - C_split.float()).abs().max().item():.3e}"
    assert torch.equal(C16, _run(Ad, tr, suhd, svhd, n, cb, fp32)), "repeat launch differs"

    ref = tref.linear(A, p, suh, svh, K, cb)
    e = C16.cpu().double() - ref
    rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    rel_max = (e.abs().max() / ref.abs().max()).item()
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"relative RMS {rel_rms:.2e}, max {rel_max:.2e}"


@pytest.mark.parametrize("m", [9, 13, 16])
@torch.inference_mode()
def test_m16_rows_beyond_m_untouched(device, m):
    """Zero-padding A to 16 rows leaves the first m output rows unchanged: the lanes of the rows past m load no A
    and their smem slots are dropped by rows_out"""
    K, cb, k, n = 5, "mul1", 1024, 1024
    if not _supported(device, K):
        pytest.skip("m = 9..16 GEMV is Pascal only for K = 5")
    g = generator(7 + m)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(7 + m)).reshape(k // 16, n // 16, -1)
    suh, svh = rand_scale(k, g).to(device), rand_scale(n, g).to(device)
    tr = torch.from_numpy(p.copy()).to(device)
    A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
    A16 = torch.zeros(16, k, dtype = torch.half, device = device)
    A16[:m] = A
    C = _run(A, tr, suh, svh, n, cb, False)
    C16 = _run(A16, tr, suh, svh, n, cb, False)
    assert torch.equal(C, C16[:m])


@pytest.mark.parametrize("K, cb, m, k, n, fp32", CASES[:1])
@torch.inference_mode()
def test_m16_disabled_falls_back(device, K, cb, m, k, n, fp32):
    """EXL3_GEMV_M16=0 removes the path, so the forced entry point reports the call as not eligible (the exl3_gemm
    dispatch then takes the tile GEMM, which is the pre-existing m > 8 behaviour)"""
    if not _supported(device, K):
        pytest.skip("m = 9..16 GEMV is Pascal/Turing only")
    g = generator(99)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(99)).reshape(k // 16, n // 16, -1)
    suh, svh = rand_scale(k, g).to(device), rand_scale(n, g).to(device)
    tr = torch.from_numpy(p.copy()).to(device)
    A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
    C = torch.empty(m, n, dtype = torch.half, device = device)
    A_had = torch.empty_like(A)
    old = os.environ.get("EXL3_GEMV_M16")
    os.environ["EXL3_GEMV_M16"] = "0"
    try:
        with pytest.raises(RuntimeError, match = "not eligible"):
            ext.exl3_gemv(A, tr, C, suh, A_had, svh, cb == "mcg", cb == "mul1")
    finally:
        if old is None: del os.environ["EXL3_GEMV_M16"]
        else: os.environ["EXL3_GEMV_M16"] = old
