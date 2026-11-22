"""
CUDA-native decode split kernel (exllamav3_ext/attn_decode_native.cu, EXL3_ATTN_NATIVE), the graphed decode's
stand-in for the Triton split kernel: launched here directly through TritonKernel.launch, its partials reduced in
torch the way the Triton combine kernel does (then the H32 rotation back), against the eager Triton decode on the
same 8-bit paged cache. Lengths cover partial tiles, empty splits and page boundaries; both split modes; GQA groups
1 / 2 / 4 / 8.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode, _get_h32

PS = 256
HD = 128


def _setup(seed, hq, hk, pages = 40, npps = 32):
    gen = torch.Generator().manual_seed(seed)
    dev = "cuda"
    q = torch.randn((1, 1, hq, HD), generator = gen).half().to(dev)
    bt = torch.randperm(pages, generator = gen)[:npps].int().view(1, npps).to(dev)
    gpt = hk * HD // 32
    kc = torch.randint(-2 ** 31, 2 ** 31 - 1, (pages, PS, gpt * 8), generator = gen, dtype = torch.int32).to(dev)
    vc = torch.randint(-2 ** 31, 2 ** 31 - 1, (pages, PS, gpt * 8), generator = gen, dtype = torch.int32).to(dev)
    ks = (torch.rand((pages * PS, gpt), generator = gen) * 0.5 + 0.1).half().to(dev)
    vs = (torch.rand((pages * PS, gpt), generator = gen) * 0.5 + 0.1).half().to(dev)
    return q, kc, vc, bt, ks, vs


def _native(q, kc, vc, bt, sl, ks, vs, hk, num_splits, live, split_len = 0):
    hq = q.shape[2]
    group = hq // hk
    block_rows = 16 if group <= 16 else 32
    scale = HD ** -0.5
    k = ext.make_decode_split_native(group = group, head_dim = HD, bits = 8, scale = scale, n_kv_heads = hk,
                                     page_size = PS, live_split = live, block_rows = block_rows)
    programs = hk
    po = torch.full((programs * num_splits * block_rows * HD,), float("nan"), device = "cuda")
    pml = torch.full((programs * num_splits * block_rows * 2,), float("nan"), device = "cuda")
    h32 = _get_h32(q.device)
    out = torch.empty_like(q)
    k.launch(programs, num_splits, 1, [q, kc, vc, bt, sl, out, po, pml, ks, vs, h32, split_len, bt.shape[1],
                                       num_splits, q])
    torch.cuda.synchronize()
    po = po.view(programs, num_splits, block_rows, HD)[:, :, :group]
    pml = pml.view(programs, num_splits, block_rows, 2)
    assert (pml[:, :, group:, 0] == -float("inf")).all() and (pml[:, :, group:, 1] == 0).all()
    m, l = pml[:, :, :group, 0], pml[:, :, :group, 1]
    mm = m.max(dim = 1, keepdim = True).values
    w = torch.where(m == -float("inf"), torch.zeros_like(m), torch.exp(m - mm))
    o = (po * w[..., None]).sum(1) / (l * w).sum(1)[..., None]        # (hk, group, HD)
    o = o.reshape(hq, HD // 32, 32) @ h32.float()                     # rotate back
    return o.reshape(1, 1, hq, HD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize("heads", [(16, 4), (16, 2), (8, 8), (4, 2)])
@pytest.mark.parametrize("length", [0, 1, 31, 33, 255, 256, 1000, 4095, 8000])
@pytest.mark.parametrize("live", [True, False])
@torch.inference_mode()
def test_native_split_matches_triton(heads, length, live):
    hq, hk = heads
    q, kc, vc, bt, ks, vs = _setup(length + hq, hq, hk)
    sl = torch.tensor([length], dtype = torch.int32, device = "cuda")
    total = length + 1
    ref = paged_attn_triton_decode(q, None, None, kc, vc, bt, sl, causal = True, pre_appended_len = 1,
                                   qc = (ks, vs, 8, 8), n_kv_heads_override = hk).float()
    num_splits = 28
    split_len = ((total + num_splits - 1) // num_splits + 15) // 16 * 16
    out = _native(q, kc, vc, bt, sl, ks, vs, hk, num_splits, live, split_len)
    assert torch.isfinite(out).all()
    err = (out - ref).abs().max().item()
    assert err < 3e-3 * max(1.0, ref.abs().max().item()), err
