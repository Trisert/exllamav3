"""
ext.hgemv_f16 (dense fp16 GEMV for 1..4 rows, fp32 accumulation, deterministic split-K; the decode path of
unquantized linears on sm_60 / sm_75): matches an fp32 reference to fp32 rounding (fp32 output) or to the fp16
output rounding (fp16 output) for every K split, writes only its output columns of a row-strided output, is
bit-reproducible, and is what ext.hgemm and LinearFP16 dispatch to for covered shapes where it is enabled.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.fp16 import LinearFP16


pytestmark = [
    pytest.mark.skipif(not hasattr(ext, "hgemv_f16"), reason = "extension built without hgemv_f16"),
]


def _default_on(device):
    return torch.cuda.get_device_capability(device) in ((6, 0), (7, 5))


SHAPES = [(2048, 8192), (2048, 4096), (4096, 2048), (2048, 512), (512, 2048), (2048, 64), (2048, 32), (2048, 8),
          (100, 264), (4160, 1000), (2048, 248320)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M", [1, 2, 3, 4])
@pytest.mark.parametrize("K,N", SHAPES)
@torch.inference_mode()
def test_hgemv_matches_reference(device, dtype, M, K, N):
    if N > 100000 and M > 1:
        pytest.skip("one large shape is enough")
    torch.manual_seed(1234)
    x = torch.empty((M, K), device = device, dtype = torch.half).normal_()
    w = torch.empty((K, N), device = device, dtype = torch.half).normal_(std = 0.02)
    ref = x.float() @ w.float()
    ldc = N + 24
    storage = torch.full((M, ldc), 123, device = device, dtype = dtype)
    y = storage[:, :N]
    ext.hgemv_f16(x, w, y, 0)
    assert bool((storage[:, N:] == 123).all()), "output padding written"
    err = (y.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    tol = 2e-6 if dtype == torch.float32 else 1e-3
    assert err <= tol * scale, f"max err {err} vs scale {scale}"
    # Same result for every K split (fp32 summation order only)
    for ks in (1, 2, 3, 7, 16):
        if ks * M * N > (1 << 20) or ks * 32 > K * 4:
            continue
        y2 = torch.empty((M, N), device = device, dtype = dtype)
        ext.hgemv_f16(x, w, y2, ks)
        err2 = (y2.float() - ref).abs().max().item()
        assert err2 <= tol * scale, f"ksplit {ks}: max err {err2}"


@pytest.mark.parametrize("M", [1, 2, 4])
@torch.inference_mode()
def test_hgemv_deterministic(device, M):
    torch.manual_seed(5)
    K, N = 4096, 2048
    x = torch.randn((M, K), device = device, dtype = torch.half)
    w = torch.randn((K, N), device = device, dtype = torch.half) * 0.02
    y1 = torch.empty((M, N), device = device, dtype = torch.float32)
    ext.hgemv_f16(x, w, y1, 0)
    for _ in range(5):
        y2 = torch.empty_like(y1)
        ext.hgemv_f16(x, w, y2, 0)
        assert torch.equal(y1, y2)


@torch.inference_mode()
def test_hgemv_strided_input_and_3d(device):
    torch.manual_seed(7)
    K, N = 2048, 1024
    xs = torch.randn((3, K + 64), device = device, dtype = torch.half)
    x = xs[:, :K]
    w = torch.randn((K, N), device = device, dtype = torch.half) * 0.02
    y = torch.empty((3, N), device = device, dtype = torch.float32)
    ext.hgemv_f16(x, w, y, 0)
    ref = x.float() @ w.float()
    assert (y - ref).abs().max().item() <= 2e-6 * ref.abs().max().item()
    # [bsz, seqlen, K] input, [bsz, seqlen, N] output
    x3 = torch.randn((1, 2, K), device = device, dtype = torch.half)
    y3 = torch.empty((1, 2, N), device = device, dtype = torch.half)
    ext.hgemv_f16(x3, w, y3, 0)
    ref3 = x3.float() @ w.float()
    assert (y3.float() - ref3).abs().max().item() <= 1e-3 * ref3.abs().max().item()


@pytest.mark.parametrize("M", [1, 2, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_hgemm_and_linear_dispatch(device, M, dtype):
    torch.manual_seed(11)
    K, N = 2048, 4096
    x = torch.randn((M, K), device = device, dtype = torch.half)
    w = torch.randn((K, N), device = device, dtype = torch.half) * 0.02
    y = torch.empty((M, N), device = device, dtype = dtype)
    ext.hgemm(x, w, y)
    if M <= 4 and _default_on(device):
        g = torch.empty_like(y)
        ext.hgemv_f16(x, w, g, 0)
        assert torch.equal(y, g), "hgemm did not take the GEMV"
    lin = LinearFP16(K, N, w, None, K, N, 0, 0, out_dtype = dtype)
    yl = lin.forward(x.unsqueeze(0), {}).view(M, N)
    ref = x.float() @ w.float()
    tol = 2e-6 if dtype == torch.float32 and M <= 4 and _default_on(device) else 1e-3
    assert (yl.float() - ref).abs().max().item() <= tol * ref.abs().max().item()
    if M <= 4 and _default_on(device):
        assert torch.equal(yl, y)


@torch.inference_mode()
def test_hgemv_rejects_uncovered(device):
    w = torch.randn((256, 12), device = device, dtype = torch.half)   # N % 8 != 0
    x = torch.randn((1, 256), device = device, dtype = torch.half)
    y = torch.empty((1, 12), device = device, dtype = torch.half)
    with pytest.raises(Exception):
        ext.hgemv_f16(x, w, y, 0)
    ext.hgemm(x, w, y)   # falls back to cuBLAS
    assert (y.float() - x.float() @ w.float()).abs().max().item() < 0.05
