"""
Dual persistent kernel (exl3_moe_coop_d_kernel: shared + routed experts in one queue) with token rows that have no
active routed slot, as under -mcs when every pick of a token lies in the CPU range (here: all routing weights of the
row zero, which slot_info treats the same way). Such a row gets no down-stage arrival, so its output is written by
write_empty_row_chunk: the shared expert's term alone. That write must see the shared expert's output of the same
launch. Contract: bit-identical to the shared launch followed by the routed launch, for integer and half-integer
routed rates (EXL3_MOE_COOP_DUAL_H), bsz 1 (empty-row write in the gate/up stage) and bsz > 1 (rotation pre-kernel).
Shapes are Qwen3.8-Flash-Next's (H 2560, intermediate 640, top-10).
"""
import os
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_scale, rand_trellis
from testlib.moe import ptr_table

H, I, E, TOPK = 2560, 640, 48, 10


def _proj(k, n, K, E_, gen, device, keep):
    t = [rand_trellis(k, n, K, gen, device) for _ in range(E_)]
    su = [rand_scale(k, gen, device) for _ in range(E_)]
    sv = [rand_scale(n, gen, device) for _ in range(E_)]
    keep.append((t, su, sv))
    return [ptr_table(t, device), ptr_table(su, device), ptr_table(sv, device)]


def _scratch(smax, rows, fused, device):
    base = smax * (I // 128) + rows * (H // 128) + 2 * smax + 3
    return [
        torch.empty((smax, H), dtype = torch.half, device = device),
        torch.empty((smax, H), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, H), dtype = torch.float, device = device),
        torch.zeros((base + (smax + 2 if fused else 0),), dtype = torch.int, device = device),
    ]


@pytest.mark.parametrize("bsz,empty", [(1, (0,)), (2, (1,)), (3, (0, 2)), (4, (0, 1, 2, 3)), (4, (3,))])
@pytest.mark.parametrize("cb,K,K_sh", [(2, 1.5, 5), (2, 2.5, 4), (2, 3, 5)])
def test_dual_empty_rows(device, cb, K, K_sh, bsz, empty):
    gen = torch.Generator().manual_seed(int(2000 + 10 * K + K_sh + 7 * bsz + sum(empty)))
    keep = []
    rt = sum((_proj(k, n, K, E, gen, device, keep) for k, n in ((H, I), (H, I), (I, H))), [])
    st = sum((_proj(k, n, K_sh, 1, gen, device, keep) for k, n in ((H, I), (H, I), (I, H))), [])
    x = (torch.randn((bsz, H), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(bsz)]).to(device)
    rw = (torch.rand((bsz, TOPK), generator = gen) * 0.3 + 0.01).half().to(device)
    for r in empty: rw[r] = 0.0       # no active routed slot: the row is the shared term alone
    sh_w = (torch.randn((H,), generator = gen) * 0.05).half().to(device)

    sc_r = _scratch(8 * TOPK, bsz, True, device)
    sc_s = _scratch(8, 8, True, device)
    b_r = _scratch(8 * TOPK, bsz, False, device)
    b_s = _scratch(8, bsz, False, device)

    def run_pair():
        out = torch.full((8, H), float("nan"), dtype = torch.float, device = device)
        sh_buf = torch.full((8, H), float("nan"), dtype = torch.float, device = device)    # stale = NaN
        ext.exl3_moe_coop_pair(x, sel, rw, H, rt, K, sc_r, st, K_sh, sc_s, cb == 1, cb == 2, out[:bsz], sh_buf, sh_w)
        torch.cuda.synchronize()
        return out[:bsz].clone()

    def run_sep():
        out = torch.full((8, H), float("nan"), dtype = torch.float, device = device)
        sh_buf = torch.zeros((8, H), dtype = torch.float, device = device)
        sh_sel = torch.zeros((bsz, 1), dtype = torch.long, device = device)
        sh_rw = torch.ones((bsz, 1), dtype = torch.half, device = device)
        ext.exl3_moe_coop(x, sh_sel, sh_rw, -1, -1, H, *st, None, None, None, K_sh, K_sh, K_sh, cb == 1, cb == 2,
                          0, 0.0, True, *b_s[:6], b_s[6], sh_buf[:bsz], None, None)
        ext.exl3_moe_coop(x, sel, rw, -1, -1, H, *rt, None, None, None, K, K, K, cb == 1, cb == 2,
                          0, 0.0, True, *b_r[:6], b_r[6], out[:bsz], sh_buf[:bsz], sh_w)
        torch.cuda.synchronize()
        return out[:bsz].clone(), sh_buf[:bsz].clone()

    ref, sh_ref = run_sep()
    assert torch.isfinite(ref).all()
    gate = torch.sigmoid((x.float() * sh_w.float()).sum(-1, keepdim = True))
    for r in empty:
        torch.testing.assert_close(ref[r], gate[r] * sh_ref[r], rtol = 1e-5, atol = 1e-6)
    for it in range(3):
        o = run_pair()
        bad = [r for r in range(bsz) if not torch.equal(o[r], ref[r])]
        assert not bad, f"dual vs separate, call {it}: rows {bad} differ (empty rows {list(empty)})"
    for sc in (sc_r, sc_s):
        assert int(sc[6][-(sc[0].shape[0] + 2):].abs().sum()) == 0, "queue / ready counters not left clean"
