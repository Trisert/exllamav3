"""
ext.hgemm_hfma2 (Pascal sm_60 dense fp16 GEMM, HFMA2 with 32-term fp16 partials added into fp32): matches ext.hgemm
(cuBLAS, fp32 accumulation) to relative RMS 1e-3 on plain, row-strided-output and batched shapes without touching
the output padding, selects rows exactly, is bit-reproducible, and is what hgemm_recon / hgemm_batched dispatch to
on sm_60 for covered shapes.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def _sm60(device):
    return torch.cuda.get_device_capability(device) == (6, 0)


pytestmark = [
    pytest.mark.skipif(not hasattr(ext, "hgemm_hfma2"), reason = "extension built without hgemm_hfma2"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("batch,M,N,K", [
    (1, 1, 128, 32), (1, 129, 256, 128), (1, 645, 1024, 2048), (2, 130, 512, 2048), (3, 200, 2048, 512),
    (1, 2048, 2048, 4096),
])
@torch.inference_mode()
def test_hfma2_matches_cublas(device, dtype, batch, M, N, K):
    if not _sm60(device):
        pytest.skip("sm_60 kernel")
    torch.manual_seed(12345)
    a = torch.empty((batch, M, K), device = device, dtype = torch.half).normal_(std = 0.25)
    b = torch.empty((batch, K, N), device = device, dtype = torch.half).normal_(std = 0.25)
    ldc = N + 16
    storage = torch.full((batch, M, ldc), 123, device = device, dtype = dtype)
    c = storage[:, :, :N]
    if batch == 1:
        ext.hgemm_hfma2(a[0], b[0], c[0])
    else:
        c = torch.empty((batch, M, N), device = device, dtype = dtype)
        ext.hgemm_hfma2(a, b, c)
        storage = None
    if storage is not None:
        assert bool((storage[:, :, N:] == 123).all()), "output padding written"
    ref = torch.empty((batch, M, N), device = device, dtype = torch.float32)
    for i in range(batch):
        ext.hgemm(a[i], b[i], ref[i])
    rel = ((c.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert float(rel) < 1e-3

    before = c.clone()
    if batch == 1:
        ext.hgemm_hfma2(a[0], b[0], c[0])
    else:
        ext.hgemm_hfma2(a, b, c)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(c.view(bits), before.view(bits)), "not bit-reproducible"


@torch.inference_mode()
def test_hfma2_exact_row_selection(device):
    if not _sm60(device):
        pytest.skip("sm_60 kernel")
    M, N, K = 129, 256, 128
    a = torch.zeros(M, K, device = device, dtype = torch.half)
    idx = torch.arange(M, device = device) % K
    a[torch.arange(M, device = device), idx] = 1
    b = torch.randn(K, N, device = device, dtype = torch.half)
    c = torch.empty(M, N, device = device, dtype = torch.float)
    ext.hgemm_hfma2(a, b, c)
    assert torch.equal(c, b[idx].float())


@torch.inference_mode()
def test_hfma2_dispatch(device):
    """hgemm_recon and hgemm_batched take the kernel on sm_60 for grids of at least about one wave (bit-identical
    to the forced call) and keep cuBLAS below that (bit-identical to ext.hgemm)"""
    if not _sm60(device):
        pytest.skip("sm_60 kernel")
    torch.manual_seed(1)
    a = torch.randn(1024, 512, device = device, dtype = torch.half)
    b = torch.randn(512, 2048, device = device, dtype = torch.half) * 0.05
    c1 = torch.empty(1024, 2048, device = device, dtype = torch.float)
    c2 = torch.empty_like(c1)
    ext.hgemm_hfma2(a, b, c1)
    ext.hgemm_recon(a, b, c2)
    assert torch.equal(c1, c2)
    ab = torch.randn(8, 256, 512, device = device, dtype = torch.half)
    bb = torch.randn(8, 512, 1024, device = device, dtype = torch.half) * 0.05
    d1 = torch.empty(8, 256, 1024, device = device, dtype = torch.half)
    d2 = torch.empty_like(d1)
    ext.hgemm_hfma2(ab, bb, d1)
    ext.hgemm_batched(ab, bb, d2)
    assert torch.equal(d1.view(torch.int16), d2.view(torch.int16))
    # Small grid: cuBLAS
    a = torch.randn(160, 2048, device = device, dtype = torch.half)
    b = torch.randn(2048, 512, device = device, dtype = torch.half) * 0.05
    e1 = torch.empty(160, 512, device = device, dtype = torch.float)
    e2 = torch.empty_like(e1)
    ext.hgemm(a, b, e1)
    ext.hgemm_recon(a, b, e2)
    assert torch.equal(e1, e2)


@torch.inference_mode()
def test_hfma2_rejects_uncovered(device):
    if not _sm60(device):
        pytest.skip("sm_60 kernel")
    a = torch.randn(64, 48, device = device, dtype = torch.half)     # K % 32 != 0
    b = torch.randn(48, 128, device = device, dtype = torch.half)
    c = torch.empty(64, 128, device = device, dtype = torch.float)
    with pytest.raises(RuntimeError):
        ext.hgemm_hfma2(a, b, c)
