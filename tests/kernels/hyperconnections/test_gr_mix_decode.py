"""
GatedResidual decode mix (ext.gr_mix at R <= 4 on the Qwen3.8 shape, D 2560 / rank 320): the gr_*_kernel3 pair
(hc_gr_decode3.cuh, variant 1) must be bit-identical to the gr_*_kernel2 pair (variant 0) for every row count, with and
without the weighted stream copy, both module forms (post / final mixer) and both output dtypes, and match an fp64
torch reference. Also: a misaligned dots workspace falls back to the old pair, other shapes ignore the variant.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

H = 4


def _inputs(R, D, LR, use_post, device, seed = 0):
    g = torch.Generator().manual_seed(seed)
    M = LR + (H if use_post else 0)
    streams = (torch.randn((R, H, D), generator = g) * 3.0).to(device)
    fn = (torch.randn((M, H * D), generator = g) * (H * D) ** -0.5).half().to(device)
    upt = (torch.randn((H, D // 4, LR, 4), generator = g) * LR ** -0.5).half().to(device)
    w = (torch.rand((H * D,), generator = g) * 0.2 + 0.9).half().to(device)
    return streams, fn, upt, w, M


def _run(streams, wstreams, fn, upt, w, M, use_post, out_dtype, variant, dots = None):
    R, _, D = streams.shape
    dev = streams.device
    if dots is None:
        dots = torch.full((R, M + 1, H), float("nan"), device = dev)
    post = torch.full((R, H), float("nan"), device = dev) if use_post else None
    mixed = torch.full((R, D), float("nan"), dtype = out_dtype, device = dev)
    ext.gr_mix(streams, wstreams, fn, upt, w, 1e-6, dots, post, mixed, variant)
    torch.cuda.synchronize(dev)
    return dots.clone(), (post.clone() if use_post else None), mixed.clone()


def _reference(streams, fn, upt, w, LR, use_post, eps = 1e-6):
    R, _, D = streams.shape
    s = streams.double()
    rmr = torch.rsqrt(s.pow(2).mean(-1, keepdim = True) + eps)
    normed = s * rmr * w.double().view(H, D)                              # (R, H, D)
    dm = normed.reshape(R, H * D) @ fn.double().t()
    t = torch.nn.functional.silu(dm[:, : LR] / H)
    up = upt.double().permute(0, 1, 3, 2).reshape(H * D, LR)              # up[h * D + 4 c + k, i]
    gate = torch.sigmoid(t @ up.t()).view(R, H, D)
    mixed = (gate * normed).mean(1)
    post = 2.0 * torch.sigmoid(dm[:, LR :] / H) if use_post else None
    return post, mixed


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


@pytest.mark.parametrize("R", [1, 2, 3, 4])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("use_post", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_gr_mix_decode3_bit_identical(device, R, weighted, use_post, out_dtype):
    D, LR = 2560, 320
    streams, fn, upt, w, M = _inputs(R, D, LR, use_post, device, seed = R)
    wstreams = (streams * w.float().view(H, D)).contiguous() if weighted else None
    d0, p0, m0 = _run(streams, wstreams, fn, upt, w, M, use_post, out_dtype, 0)
    d1, p1, m1 = _run(streams, wstreams, fn, upt, w, M, use_post, out_dtype, 1)
    assert torch.equal(d0, d1)
    assert torch.equal(m0, m1)
    if use_post:
        assert torch.equal(p0, p1)
    ref_post, ref_mixed = _reference(streams, fn, upt, w, LR, use_post)
    assert _rel(m1, ref_mixed) < 3e-3
    if use_post:
        assert _rel(p1, ref_post) < 1e-3
    # run to run
    d2, p2, m2 = _run(streams, wstreams, fn, upt, w, M, use_post, out_dtype, 1)
    assert torch.equal(m1, m2) and torch.equal(d1, d2)
    # the default (-1) selects one of the two pairs: same bits either way
    _, _, m3 = _run(streams, wstreams, fn, upt, w, M, use_post, out_dtype, -1)
    assert torch.equal(m1, m3)


@torch.inference_mode()
def test_gr_mix_decode3_misaligned_dots_falls_back(device):
    D, LR, R = 2560, 320, 2
    streams, fn, upt, w, M = _inputs(R, D, LR, True, device)
    buf = torch.full((R * (M + 1) * H + 1,), float("nan"), device = device)
    dots = buf[1:].view(R, M + 1, H)
    assert dots.data_ptr() % 16 != 0
    _, p1, m1 = _run(streams, None, fn, upt, w, M, True, torch.half, 1, dots = dots)
    _, p0, m0 = _run(streams, None, fn, upt, w, M, True, torch.half, 0)
    assert torch.equal(m0, m1) and torch.equal(p0, p1)


@pytest.mark.parametrize("D, LR, R", [(4096, 512, 2), (1024, 320, 2), (2560, 320, 8)])
@torch.inference_mode()
def test_gr_mix_decode3_other_shapes_unchanged(device, D, LR, R):
    streams, fn, upt, w, M = _inputs(R, D, LR, True, device)
    _, p0, m0 = _run(streams, None, fn, upt, w, M, True, torch.half, 0)
    _, p1, m1 = _run(streams, None, fn, upt, w, M, True, torch.half, 1)
    assert torch.equal(m0, m1) and torch.equal(p0, p1)
