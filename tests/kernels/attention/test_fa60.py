"""
fa60 (exllamav3_ext/pascal/fa60.cu): HFMA2 flash-attention prefill kernel for head_dim 256 on Pascal (sm_60),
against an fp32 reference and against the SDPA window path it replaces, for prefill chunks appended to a cache
(bottom-right causal), with and without GQA. The kernel uses only HFMA2 / fp32 SIMT instructions, so it runs on
every CUDA device from sm_60 up.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.cuda_only


def _reference(q, k, v, scale, causal):
    Tq, Hq, D = q.shape
    Tkv, Hkv, _ = k.shape
    grp = Hq // Hkv
    qf = q.float().transpose(0, 1)                                   # [Hq, Tq, D]
    kf = k.float().repeat_interleave(grp, dim = 1).transpose(0, 1)   # [Hq, Tkv, D]
    vf = v.float().repeat_interleave(grp, dim = 1).transpose(0, 1)
    s = (qf @ kf.transpose(1, 2)) * scale
    if causal:
        rows = torch.arange(Tq, device = q.device)[:, None] + (Tkv - Tq)
        cols = torch.arange(Tkv, device = q.device)[None, :]
        s = s.masked_fill(cols > rows, float("-inf"))
    return (torch.softmax(s, dim = -1) @ vf).transpose(0, 1)         # [Tq, Hq, D]


def _rel_err(o, ref):
    return ((o.float() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("Tq, Tkv", [(256, 256), (300, 1000), (37, 37), (64, 5000), (1000, 1000), (1, 700), (65, 81)])
@pytest.mark.parametrize("Hq, Hkv", [(24, 4), (8, 8)])
@pytest.mark.parametrize("causal", [True, False])
def test_fa60_matches_reference(Tq, Tkv, Hq, Hkv, causal, device):
    gen = torch.Generator(device = device).manual_seed(Tq * 7 + Tkv)
    q = torch.randn(Tq, Hq, 256, device = device, dtype = torch.half, generator = gen)
    k = torch.randn(Tkv, Hkv, 256, device = device, dtype = torch.half, generator = gen)
    v = torch.randn(Tkv, Hkv, 256, device = device, dtype = torch.half, generator = gen)
    scale = 256 ** -0.5
    o = torch.empty_like(q)
    ext.fa60_fwd(q, k, v, o, scale, causal)
    ref = _reference(q, k, v, scale, causal)
    assert torch.isfinite(o).all()
    assert _rel_err(o, ref) < 4e-3


def test_fa60_strided_q(device):
    # q as a head-slice view of a wider projection output (row stride > Hq * 256)
    gen = torch.Generator(device = device).manual_seed(5)
    qkv = torch.randn(333, 32, 256, device = device, dtype = torch.half, generator = gen)
    q = qkv[:, :24]
    k = torch.randn(1333, 4, 256, device = device, dtype = torch.half, generator = gen)
    v = torch.randn(1333, 4, 256, device = device, dtype = torch.half, generator = gen)
    o = torch.empty(333, 24, 256, device = device, dtype = torch.half)
    ext.fa60_fwd(q, k, v, o, 0.0625, True)
    ref = _reference(q, k, v, 0.0625, True)
    assert _rel_err(o, ref) < 4e-3


@pytest.mark.parametrize("peak", [4.0, 12.0])
def test_fa60_peaked_scores(peak, device):
    # Large-magnitude logits (QK-normed heads with sharp attention): q and k share a direction per key group,
    # so the scores reach |s * scale| ~ peak * 16 and the fp16 partials of Q K^T run near their range
    gen = torch.Generator(device = device).manual_seed(int(peak))
    Tq, Tkv = 200, 900
    base = torch.randn(Tkv, 4, 256, device = device, generator = gen)
    k = (base + 0.3 * torch.randn(Tkv, 4, 256, device = device, generator = gen)).half()
    q = (peak * base[-Tq:].repeat_interleave(6, dim = 1) / 16 +
         0.3 * torch.randn(Tq, 24, 256, device = device, generator = gen)).half()
    v = torch.randn(Tkv, 4, 256, device = device, dtype = torch.half, generator = gen)
    o = torch.empty_like(q)
    ext.fa60_fwd(q, k, v, o, 256 ** -0.5, True)
    ref = _reference(q, k, v, 256 ** -0.5, True)
    assert torch.isfinite(o).all()
    assert _rel_err(o, ref) < 1e-2


@pytest.mark.slow
@pytest.mark.parametrize("Tq, Tkv", [(2048, 16384), (2048, 30000)])
def test_fa60_long(Tq, Tkv, device):
    # Long cached prefix, against the SDPA window path (fp32 reference over 30000 keys does not fit everywhere)
    from exllamav3.modules.attention_fn.torch import _sdpa_chunks
    gen = torch.Generator(device = device).manual_seed(Tkv)
    q = torch.randn(Tq, 24, 256, device = device, dtype = torch.half, generator = gen)
    k = torch.randn(Tkv, 4, 256, device = device, dtype = torch.half, generator = gen)
    v = torch.randn(Tkv, 4, 256, device = device, dtype = torch.half, generator = gen)
    o = torch.empty_like(q)
    ext.fa60_fwd(q, k, v, o, 256 ** -0.5, True)
    ref = _sdpa_chunks(q, k, v, Tkv, Tq, 24, 4, True, 256 ** -0.5, 512, True)
    assert torch.isfinite(o).all()
    assert _rel_err(o, ref.float()) < 4e-3
