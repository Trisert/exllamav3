"""
GatedResidual decode mix on 8-bit tables (ext.gr_mix_q8, EXL3_GR_Q8; hc_gr_decode3.cuh with Q8): int8 fn with an fp32
scale per (row, stream) and int8 repacked up with an fp32 scale per up row. Contract: matches the fp64 reference of
the mix computed with the dequantized tables (int8 * scale) to the same tolerance as the fp16 kernel pair, for R = 1..4,
with and without the weighted stream copy, both module forms and both output dtypes; run-to-run identical; returns
False for shapes the pair does not cover (R > 4). Against the fp16 tables the quantization error is bounded too.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

from test_gr_mix_decode import H, _inputs, _reference, _rel


def _quant(fn, upt, M, D, LR):
    f = fn.float().view(M, H, D)
    fn_s = (f.abs().amax(-1) / 127.0).clamp_min(1e-30)
    fn8 = torch.round(f / fn_s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    up = upt.float().permute(0, 1, 3, 2).reshape(H * D, LR)
    up_s = (up.abs().amax(-1) / 127.0).clamp_min(1e-30)
    up8 = torch.round(up / up_s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    upt8 = up8.view(H, D // 4, 4, LR).permute(0, 1, 3, 2).contiguous()
    fn_dq = (fn8.float() * fn_s.unsqueeze(-1)).view(M, H * D)
    upt_dq = (up8.float() * up_s.unsqueeze(-1)).view(H, D // 4, 4, LR).permute(0, 1, 3, 2)
    return fn8.view(M, H * D).contiguous(), fn_s.contiguous(), upt8, up_s.contiguous(), fn_dq, upt_dq


def _run_q8(streams, wstreams, q, M, use_post, out_dtype):
    R, _, D = streams.shape
    dev = streams.device
    dots = torch.full((R, M + 1, H), float("nan"), device = dev)
    post = torch.full((R, H), float("nan"), device = dev) if use_post else None
    mixed = torch.full((R, D), float("nan"), dtype = out_dtype, device = dev)
    w = q[-1]
    ok = ext.gr_mix_q8(streams, wstreams, q[0], q[1], q[2], q[3], w, 1e-6, dots, post, mixed)
    torch.cuda.synchronize(dev)
    return ok, (post.clone() if use_post else None), mixed.clone()


@pytest.mark.parametrize("R", [1, 2, 3, 4])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("use_post", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_gr_mix_q8_matches_dequantized_reference(device, R, weighted, use_post, out_dtype):
    D, LR = 2560, 320
    streams, fn, upt, w, M = _inputs(R, D, LR, use_post, device, seed = 10 + R)
    fn8, fn_s, upt8, up_s, fn_dq, upt_dq = _quant(fn, upt, M, D, LR)
    q = (fn8, fn_s, upt8, up_s, w)
    wstreams = (streams * w.float().view(H, D)).contiguous() if weighted else None
    ok, p1, m1 = _run_q8(streams, wstreams, q, M, use_post, out_dtype)
    assert ok
    ref_post, ref_mixed = _reference(streams, fn_dq, upt_dq, w, LR, use_post)
    assert _rel(m1, ref_mixed) < 3e-3
    if use_post:
        assert _rel(p1, ref_post) < 1e-3
    _, p2, m2 = _run_q8(streams, wstreams, q, M, use_post, out_dtype)
    assert torch.equal(m1, m2)
    # quantization error against the fp16 tables
    _, fp_mixed = _reference(streams, fn, upt, w, LR, use_post)
    assert _rel(m1, fp_mixed) < 2e-2


@torch.inference_mode()
def test_gr_mix_q8_declines_large_r(device):
    D, LR = 2560, 320
    streams, fn, upt, w, M = _inputs(5, D, LR, True, device)
    fn8, fn_s, upt8, up_s, _, _ = _quant(fn, upt, M, D, LR)
    ok, _, _ = _run_q8(streams, None, (fn8, fn_s, upt8, up_s, w), M, True, torch.half)
    assert not ok
