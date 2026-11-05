"""
Pascal (sm_60) m = 3..4 GEMV, MMODE 4 of exl3_gemv_kernel (quant/exl3_gemv_kernel.cuh): one output (column lane / 4,
row lane % 4) per lane and n8 fragment, with the same HFMA2 chain order, pair sum, fp16 run sum and fp32 fold cadence
as the emulated MMA path it replaces (MMODE 1, exl3_mma_m16n8k16_f16_pregathered). Contract: bit-identical to MMODE 1
(EXL3_GEMV_M34=0, read per call) for every eligible integer bitrate, codebook, both kernel configurations and both
output dtypes, and within the GEMV tolerance of the float64 reference (see test_exl3_gemv.py). Other m and other
architectures do not take MMODE 4 (the switch changes nothing there).
"""

import os

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.env import is_rocm
from testlib.exl3 import generator, rand_scale

CASES = [
    # K, codebook, m, k, n, fp32 out
    (5, "mul1", 4, 2048, 2048, False),      # narrow config, Flash-Next attention / GDN projection class
    (5, "mul1", 3, 2048, 2048, False),
    (5, "mul1", 4, 512, 8448, True),        # wide config
    (5, "mcg", 3, 384, 640, True),
    (6, "mul1", 4, 512, 1024, False),
    (6, "mcg", 3, 256, 8320, True),         # wide config
    (4, "mul1", 4, 1024, 8320, False),      # wide config, lm_head class (4 bpw, shuffle extraction)
    (4, "3inst", 3, 1024, 1024, True),
    (4, "mcg", 4, 512, 384, False),
    (3, "mul1", 4, 512, 640, False),
    (3, "mcg", 3, 512, 8448, True),
    (2, "mul1", 4, 1024, 640, True),
    (2, "mcg", 3, 512, 8320, False),
]


def _run(A, tr, suh, svh, n, cb, fp32, m34):
    m = A.shape[0]
    rows = 2 * n
    big = torch.full((m * n + 2 * rows,), 7.0, dtype = torch.float if fp32 else torch.half, device = A.device)
    C = big[rows : rows + m * n].view(m, n)
    A_had = torch.empty_like(A)
    old = os.environ.get("EXL3_GEMV_M34")
    os.environ["EXL3_GEMV_M34"] = "1" if m34 else "0"
    try:
        ext.exl3_gemv(A, tr, C, suh, A_had, svh, cb == "mcg", cb == "mul1")
    finally:
        if old is None: del os.environ["EXL3_GEMV_M34"]
        else: os.environ["EXL3_GEMV_M34"] = old
    torch.cuda.synchronize(A.device)
    assert (big[:rows] == 7.0).all() and (big[-rows:] == 7.0).all(), "wrote outside C"
    return C


@pytest.mark.parametrize("K, cb, m, k, n, fp32", CASES)
@torch.inference_mode()
def test_m34_bit_identical_to_mma_path(device, K, cb, m, k, n, fp32):
    if is_rocm():
        pytest.skip("CUDA GEMV only")
    seed = m * 1000 + k + n + K
    g = generator(seed)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)).reshape(k // 16, n // 16, -1)
    suh, svh = rand_scale(k, g), rand_scale(n, g)
    A = (torch.randn(m, k, generator = g) * 0.5).half()
    tr = torch.from_numpy(p.copy()).to(device)
    Ad, suhd, svhd = A.to(device), suh.to(device), svh.to(device)

    C_new = _run(Ad, tr, suhd, svhd, n, cb, fp32, True)
    C_old = _run(Ad, tr, suhd, svhd, n, cb, fp32, False)
    assert torch.equal(C_new, C_old), \
        f"MMODE 4 differs from MMODE 1: max |diff| {(C_new.float() - C_old.float()).abs().max().item():.3e}"
    assert torch.equal(C_new, _run(Ad, tr, suhd, svhd, n, cb, fp32, True)), "repeat launch differs"

    ref = tref.linear(A, p, suh, svh, K, cb)
    e = C_new.cpu().double() - ref
    rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    rel_max = (e.abs().max() / ref.abs().max()).item()
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"relative RMS {rel_rms:.2e}, max {rel_max:.2e}"


@pytest.mark.parametrize("m", [3, 4])
@torch.inference_mode()
def test_m34_rows_beyond_m_untouched(device, m):
    """At m = 3 the lanes of row 3 read no A and store nothing; zero-padding A to four rows leaves the first m
    output rows unchanged"""
    if is_rocm():
        pytest.skip("CUDA GEMV only")
    K, cb, k, n = 5, "mul1", 1024, 1024
    g = generator(7 + m)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(7 + m)).reshape(k // 16, n // 16, -1)
    suh, svh = rand_scale(k, g).to(device), rand_scale(n, g).to(device)
    tr = torch.from_numpy(p.copy()).to(device)
    A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
    A4 = torch.zeros(4, k, dtype = torch.half, device = device)
    A4[:m] = A
    C = _run(A, tr, suh, svh, n, cb, False, True)
    C4 = _run(A4, tr, suh, svh, n, cb, False, True)
    assert torch.equal(C, C4[:m])
