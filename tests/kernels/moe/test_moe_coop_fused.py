"""
Persistent single-launch form of the fused decode MoE kernels (exl3_moe_coop_f_kernel, EXL3_MOE_COOP_FUSED): it runs
the two-launch path's blocks as work items of one queue, so its output must be bit-identical to the two launches.
The launcher takes it when the counter scratch has the queue / ready tail (exl3_moe_coop_ctr_len_fused) and, by
default on sm_60 / sm_75, when the gate/up stage fills the device; a scratch of the base length keeps the two
launches. Shapes are the 35B-A3B MoE (Hi 2048, I 512, top-8 of 64 experts: 64 gate/up blocks per token) so the
default selection takes the persistent kernel on the P100 and the RTX 2060. Also checks that both paths leave the
shared counters clean for each other (they share one scratch per device in the model).
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_scale, rand_trellis
from testlib.moe import ptr_table


def _proj(k, n, K, E, gen, device):
    t = [rand_trellis(k, n, K, gen, device) for _ in range(E)]
    su = [rand_scale(k, gen, device) for _ in range(E)]
    sv = [rand_scale(n, gen, device) for _ in range(E)]
    return (ptr_table(t, device), ptr_table(su, device), ptr_table(sv, device)), (t, su, sv)


def _run_pair(device, seed, K, cb, bsz, E = 64, topk = 8, Hi = 2048, I = 512, Ho = 2048, shared = False,
              min_e = -1, max_e = -1):
    gen = torch.Generator().manual_seed(seed)
    g, kg = _proj(Hi, I, K, E, gen, device)
    u, ku = _proj(Hi, I, K, E, gen, device)
    d, kd = _proj(I, Ho, K, E, gen, device)
    n_global = E if min_e < 0 else E + 8
    x = (torch.randn((bsz, Hi), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(n_global, generator = gen)[:topk] for _ in range(bsz)]).to(device)
    rw = (torch.rand((bsz, topk), generator = gen) * 0.3).half().to(device)
    rw[0, 1] = 0.0
    sh_out = torch.randn((bsz, Hi), generator = gen).float().to(device) if shared else None
    sh_w = (torch.randn((Hi,), generator = gen) * 0.05).half().to(device) if shared else None
    smax = 8 * topk
    had_g = torch.empty((smax, Hi), dtype = torch.half, device = device)
    had_u = torch.empty_like(had_g)
    gu_g = torch.empty((smax, 1, I), dtype = torch.half, device = device)
    gu_u = torch.empty_like(gu_g)
    act = torch.empty((smax, 1, I), dtype = torch.half, device = device)
    d_out = torch.empty((smax, 1, Ho), dtype = torch.float, device = device)
    # out rows passed = bsz: the down counters cover bsz token rows (exl3_moe_coop_ctr_len)
    base = smax * (I // 128) + bsz * (Ho // 128) + 2 * smax + 3
    ctr_fused = torch.zeros((base + smax + 2,), dtype = torch.int, device = device)
    ctr_base = ctr_fused[:base]     # same memory, base length: the two-launch path

    def run(ctr):
        out = torch.full((8, Ho), float("nan"), dtype = torch.float, device = device)
        ext.exl3_moe_coop(x, sel, rw, min_e, max_e, Hi, *g, *u, *d, None, None, None, K, K, K, cb == 1, cb == 2,
                          0, 0.0, True, had_g, had_u, gu_g, gu_u, act, d_out, ctr, out[:bsz], sh_out, sh_w)
        torch.cuda.synchronize()
        return out[:bsz].clone()

    # Alternate the paths on one scratch: each must leave the counters clean for the other
    f1 = run(ctr_fused)
    b1 = run(ctr_base)
    f2 = run(ctr_fused)
    f3 = run(ctr_fused)
    b2 = run(ctr_base)
    assert torch.isfinite(f1).all()
    assert torch.equal(f1, b1), f"fused vs two-launch: max diff {(f1 - b1).abs().max().item():.3e}"
    for o in (f2, f3, b2):
        assert torch.equal(o, f1), "not reproducible across calls / path switches"
    # completion counters and the queue / ready tail (the run table in between is per-call data)
    n_ctr = smax * (I // 128) + bsz * (Ho // 128)
    assert int(ctr_fused[:n_ctr].abs().sum()) == 0 and int(ctr_fused[base:].abs().sum()) == 0, "counters not left clean"


@pytest.mark.parametrize("bsz", [1, 2, 4, 8])
@pytest.mark.parametrize("cb,K", [(2, 2), (2, 3), (2, 4), (1, 3), (1, 4), (0, 3), (2, 1.5), (2, 2.5), (2, 3.5), (1, 5)])
def test_fused_bit_identical(device, cb, K, bsz):
    _run_pair(device, int(K * 10) + cb * 100 + bsz, K, cb, bsz, shared = bsz % 2 == 1)


def _scratch(smax, rows, Hi, I, Ho, device, fused):
    base = smax * (I // 128) + rows * (Ho // 128) + 2 * smax + 3
    return [
        torch.empty((smax, Hi), dtype = torch.half, device = device),
        torch.empty((smax, Hi), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, Ho), dtype = torch.float, device = device),
        torch.zeros((base + (smax + 2 if fused else 0),), dtype = torch.int, device = device),
    ]


@pytest.mark.parametrize("bsz", [1, 2, 8])
@pytest.mark.parametrize("cb,K,K_sh", [(2, 3, 5), (2, 3, 3), (1, 3, 4), (2, 2, 4), (1, 4, 6)])
def test_dual_shared_routed(device, cb, K, K_sh, bsz):
    """Shared + routed experts in one persistent launch (exl3_moe_coop_pair, the BC_BlockSparseMLP decode path)
    against the shared launch followed by the routed launch, bit for bit"""
    Hi, I, Ho, E, topk = 2048, 512, 2048, 64, 8
    gen = torch.Generator().manual_seed(1000 + 10 * K + K_sh + bsz)
    keep = []
    def tabs(E_, K_):
        out = []
        for k, n in ((Hi, I), (Hi, I), (I, Ho)):
            t, kp = _proj(k, n, K_, E_, gen, device)
            out += list(t); keep.append(kp)
        return out
    rt = tabs(E, K)
    st = tabs(1, K_sh)
    x = (torch.randn((bsz, Hi), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:topk] for _ in range(bsz)]).to(device)
    rw = (torch.rand((bsz, topk), generator = gen) * 0.3).half().to(device)
    sh_w = (torch.randn((Hi,), generator = gen) * 0.05).half().to(device)

    def run_pair():
        out = torch.full((8, Ho), float("nan"), dtype = torch.float, device = device)
        sh_buf = torch.zeros((8, Hi), dtype = torch.float, device = device)
        ext.exl3_moe_coop_pair(x, sel, rw, Hi, rt, K, sc_r, st, K_sh, sc_s, cb == 1, cb == 2, out[:bsz], sh_buf, sh_w)
        torch.cuda.synchronize()
        return out[:bsz].clone()

    def run_sep():
        out = torch.full((8, Ho), float("nan"), dtype = torch.float, device = device)
        sh_buf = torch.zeros((8, Hi), dtype = torch.float, device = device)
        sh_sel = torch.zeros((bsz, 1), dtype = torch.long, device = device)
        sh_rw = torch.ones((bsz, 1), dtype = torch.half, device = device)
        ext.exl3_moe_coop(x, sh_sel, sh_rw, -1, -1, Hi, *st, None, None, None, K_sh, K_sh, K_sh, cb == 1, cb == 2,
                          0, 0.0, True, *b_s[:6], b_s[6], sh_buf[:bsz], None, None)
        ext.exl3_moe_coop(x, sel, rw, -1, -1, Hi, *rt, None, None, None, K, K, K, cb == 1, cb == 2,
                          0, 0.0, True, *b_r[:6], b_r[6], out[:bsz], sh_buf[:bsz], sh_w)
        torch.cuda.synchronize()
        return out[:bsz].clone()

    sc_r = _scratch(8 * topk, bsz, Hi, I, Ho, device, True)
    sc_s = _scratch(8, 8, Hi, I, Ho, device, True)        # the shared output holds 8 rows
    b_r = _scratch(8 * topk, bsz, Hi, I, Ho, device, False)
    b_s = _scratch(8, bsz, Hi, I, Ho, device, False)
    ref = run_sep()
    o1 = run_pair()
    o2 = run_pair()
    assert torch.isfinite(o1).all()
    assert torch.equal(o1, ref), f"dual vs separate: max diff {(o1 - ref).abs().max().item():.3e}"
    assert torch.equal(o2, o1)
    for sc in (sc_r, sc_s):
        assert int(sc[6][-(sc[0].shape[0] + 2):].abs().sum()) == 0, "queue / ready counters not left clean"


def test_fused_duplicates_and_masking(device):
    # few experts at bsz 8: runs of several slots per expert (multi-row tiles, ready counted per slot)
    _run_pair(device, 7, 3, 1, 8, E = 6, topk = 4, I = 1024)
    # expert-range masking: out-of-range picks are inactive slots whose items exit at once
    _run_pair(device, 8, 4, 2, 1, min_e = 8, max_e = 72)
    _run_pair(device, 9, 3, 2, 4, min_e = 8, max_e = 72, shared = True)
