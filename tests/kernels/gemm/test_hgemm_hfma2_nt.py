"""
ext.hgemm_hfma2_nt (Pascal sm_60, HFMA2 GEMM with the weight in its natural [N][K] layout, used by the hyper-connection
mix GEMMs): matches a float32 reference to relative RMS 1e-3, handles Nb < N (a weight with fewer rows than the padded
output, e.g. 324 -> 384) without reading outside the weight, is bit-reproducible, and the output rows / padding columns
are written only inside the tile grid.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def _sm60(device):
    return torch.cuda.get_device_capability(device) == (6, 0)


pytestmark = [
    pytest.mark.skipif(not hasattr(ext, "hgemm_hfma2_nt"), reason = "extension built without hgemm_hfma2_nt"),
]


@pytest.mark.parametrize("bm", [128, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M,Nb,N,K", [
    (1, 128, 128, 32), (129, 256, 256, 128), (645, 1024, 1024, 2048), (2048, 324, 384, 10240),
    (2048, 10240, 10240, 320), (333, 129, 256, 64), (200, 1, 128, 32),
])
@torch.inference_mode()
def test_hfma2_nt_matches_reference(device, dtype, M, Nb, N, K, bm):
    if not _sm60(device):
        pytest.skip("sm_60 kernel")
    torch.manual_seed(4321)
    a = torch.empty((M, K), device = device, dtype = torch.half).normal_(std = 0.25)
    # Weight in a larger allocation so a read past row Nb would pick up the sentinel, not zeros
    wstore = torch.full((Nb + 64, K), 7.0, device = device, dtype = torch.half)
    wstore[:Nb].normal_(std = 0.25)
    w = wstore[:Nb]
    c = torch.full((M, N), 123, device = device, dtype = dtype)
    ext.hgemm_hfma2_nt(a, w, c, bm)
    ref = a.float() @ w.float().t()
    got = c[:, :Nb].float()
    rel = ((got - ref).square().mean() / ref.square().mean()).sqrt()
    assert rel < 1e-3, f"rel rms {rel}"
    c2 = torch.full((M, N), 55, device = device, dtype = dtype)
    ext.hgemm_hfma2_nt(a, w, c2, bm)
    assert torch.equal(c[:, :Nb], c2[:, :Nb]), "not reproducible"


@pytest.mark.parametrize("bm", [128, 64])
@pytest.mark.parametrize("M,Nb,N,K", [(257, 324, 384, 96), (130, 200, 256, 160), (64, 128, 128, 32), (300, 640, 640, 224)])
@torch.inference_mode()
def test_hfma2_nt_guards_padding_and_matches_non_nt(device, M, Nb, N, K, bm):
    """Row-strided C (ldc > N) with guard rows and columns: nothing outside the M x N window is written; the padding
    columns Nb..N-1 replicate the last real column (the clamped weight row); when Nb == N the result is bit-identical
    to the [K][N] kernel on the transposed weight. K = 96 / 160 / 224 give 3 / 5 / 7 K tiles (odd, > 1)"""
    if not _sm60(device):
        pytest.skip("sm_60 kernel")
    torch.manual_seed(99)
    a = torch.empty((M, K), device = device, dtype = torch.half).normal_(std = 0.25)
    w = torch.empty((Nb, K), device = device, dtype = torch.half).normal_(std = 0.25)
    ldc = N + 16
    storage = torch.full((M + 8, ldc), 123, device = device, dtype = torch.half)
    c = storage[:M, :N]
    ext.hgemm_hfma2_nt(a, w, c, bm)
    torch.cuda.synchronize()
    assert bool((storage[M:] == 123).all()), "guard rows written"
    assert bool((storage[:M, N:] == 123).all()), "row padding written"
    if Nb < N:
        assert torch.equal(c[:, Nb:], c[:, Nb - 1:Nb].expand(-1, N - Nb)), "padding columns are not the clamped row"
    ref = a.float() @ w.float().t()
    assert ((c[:, :Nb].float() - ref).square().mean() / ref.square().mean()).sqrt() < 1e-3
    if Nb == N:
        c2 = torch.empty((M, N), device = device, dtype = torch.half)
        ext.hgemm_hfma2(a, w.t().contiguous(), c2, bm)
        assert torch.equal(c, c2), "NT kernel differs from the [K][N] kernel on the transposed weight"


@torch.inference_mode()
def test_hfma2_nt_try_rejects_uncovered(device):
    if not _sm60(device):
        pytest.skip("sm_60 kernel")
    a = torch.randn((64, 320), device = device).half()
    w = torch.randn((256, 320), device = device).half()
    c = torch.zeros((64, 256), device = device, dtype = torch.half)
    # K not a multiple of 32
    assert not ext.hgemm_hfma2_nt_try(a[:, :312].contiguous(), w[:, :312].contiguous(), c, 1)
    # Nb larger than the output width
    assert not ext.hgemm_hfma2_nt_try(a, torch.randn((384, 320), device = device).half(), c, 1)
    # output width not a multiple of 128
    assert not ext.hgemm_hfma2_nt_try(a, w[:200].contiguous(), torch.zeros((64, 200), device = device, dtype = torch.half), 1)
    # misaligned A (offset of one half element)
    base = torch.randn((64 * 320 + 8,), device = device).half()
    assert not ext.hgemm_hfma2_nt_try(base[1:1 + 64 * 320].view(64, 320), w, c, 1)
    # non-contiguous weight (every other row)
    assert not ext.hgemm_hfma2_nt_try(a, torch.randn((512, 320), device = device).half()[::2], c, 1)
    torch.cuda.synchronize()


@torch.inference_mode()
def test_hfma2_nt_try_gates(device):
    if not _sm60(device):
        pytest.skip("sm_60 kernel")
    a = torch.randn((64, 320), device = device).half()
    w = torch.randn((256, 320), device = device).half()
    c = torch.zeros((64, 256), device = device, dtype = torch.half)
    assert not ext.hgemm_hfma2_nt_try(a, w, c, 1024), "too few rows must fall through to cuBLAS"
    assert ext.hgemm_hfma2_nt_try(a, w, c, 1)
    torch.cuda.synchronize()
    ref = a.float() @ w.float().t()
    assert ((c.float() - ref).square().mean() / ref.square().mean()).sqrt() < 1e-3
