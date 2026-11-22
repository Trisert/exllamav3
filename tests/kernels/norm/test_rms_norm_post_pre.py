"""
rms_norm_post_pre: a post-norm residual (r += norm(x) * w1) and the next pre-norm (out = norm(r) * w2, fp16) in one
launch, bit-identical to rms_norm with add_residual followed by rms_norm over r (post-norm models such as LFM3)
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize("dim", [256, 2048, 4096])
@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("x_dtype", [torch.float, torch.half])
@pytest.mark.parametrize("r_dtype", [torch.float, torch.half])
@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("bias", [0.0, 1.0])
@torch.inference_mode()
def test_post_pre_matches_two_launches(dim, rows, x_dtype, r_dtype, w_dtype, bias):
    gen = torch.Generator(device = "cuda").manual_seed(dim + rows)
    dev = "cuda"
    x = (torch.randn((rows, dim), generator = gen, device = dev) * 3).to(x_dtype)
    r0 = (torch.randn((rows, dim), generator = gen, device = dev) * 5).to(r_dtype)
    w1 = (torch.randn(dim, generator = gen, device = dev) * 0.3 + 1).to(w_dtype)
    w2 = (torch.randn(dim, generator = gen, device = dev) * 0.3 + 1).to(w_dtype)
    eps1, eps2, cs1, cs2 = 1e-5, 1e-6, 1.0, 1.0

    r_ref = r0.clone()
    ext.rms_norm(x, w1, r_ref, eps1, bias, cs1, False, True, 1)
    out_ref = torch.empty((rows, dim), dtype = torch.half, device = dev)
    ext.rms_norm(r_ref, w2, out_ref, eps2, bias, cs2, False, False, 1)

    r = r0.clone()
    out = torch.empty((rows, dim), dtype = torch.half, device = dev)
    ext.rms_norm_post_pre(x, w1, r, w2, out, eps1, bias, cs1, eps2, bias, cs2)
    torch.cuda.synchronize()
    assert torch.equal(r, r_ref)
    assert torch.equal(out, out_ref)
