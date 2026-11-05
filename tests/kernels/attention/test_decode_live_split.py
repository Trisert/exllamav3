"""
EXL3_ATTN_LIVE_SPLIT: the paged decode split kernel divides the live kv length (cache_seqlens, read on the device)
over its splits instead of the block-table bound. Only the split boundaries move, so the result must match the
bound-split launch up to the reduction order, for fp16 and 4-bit caches, several lengths (including lengths
that leave splits empty and lengths past one split per tile), q_len 1 and 2, and both tile widths.
"""

import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode

PS = 256


def _setup(qc, q_len, seed, hq = 16, hk = 2, d = 256, pages = 32, npps = 16):
    gen = torch.Generator().manual_seed(seed)
    dev = "cuda"
    q = torch.randn((1, q_len, hq, d), generator = gen).half().to(dev)
    bt = torch.randperm(pages, generator = gen)[:npps].int().view(1, npps).to(dev)
    if qc:
        gpt = hk * d // 32
        kc = torch.randint(-2 ** 31, 2 ** 31 - 1, (pages, PS, gpt * 4), generator = gen, dtype = torch.int32).to(dev)
        vc = torch.randint(-2 ** 31, 2 ** 31 - 1, (pages, PS, gpt * 4), generator = gen, dtype = torch.int32).to(dev)
        ks = (torch.rand((pages * PS, gpt), generator = gen) * 0.5 + 0.1).half().to(dev)
        vs = (torch.rand((pages * PS, gpt), generator = gen) * 0.5 + 0.1).half().to(dev)
        return q, kc, vc, bt, dict(qc = (ks, vs, 4, 4), n_kv_heads_override = hk)
    kc = torch.randn((pages, PS, hk, d), generator = gen).half().to(dev)
    vc = torch.randn((pages, PS, hk, d), generator = gen).half().to(dev)
    return q, kc, vc, bt, {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize("qc", (False, True))
@pytest.mark.parametrize("q_len", (1, 2))
@pytest.mark.parametrize("length", (1, 17, 300, 500, 1000, 2047, 3900))
@pytest.mark.parametrize("block_n", (16, 32))
@torch.inference_mode()
def test_live_split_matches_bound_split(qc, q_len, length, block_n):
    q, kc, vc, bt, kw = _setup(qc, q_len, length)
    sl = torch.tensor([length], dtype = torch.int32, device = "cuda")
    args = (q, None, None, kc, vc, bt, sl)
    ref = paged_attn_triton_decode(*args, causal = True, pre_appended_len = q_len, block_n = block_n,
                                   live_split = False, **kw).float()
    out = paged_attn_triton_decode(*args, causal = True, pre_appended_len = q_len, block_n = block_n,
                                   live_split = True, **kw).float()
    assert torch.isfinite(out).all()
    err = (out - ref).abs().max().item()
    assert err < 2e-3 * max(1.0, ref.abs().max().item()), err
