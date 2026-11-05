"""
Pascal (sm_60) fused decode MoE, runs of 2..4 rows (the MTP verify at m = 3..4): gemv_tile of exl3_moe_coop_kernel.cuh
takes the one-output-per-lane form of the emulated MMA (row lane % 4, column lane / 4 per n8 fragment, as MMODE 4 of
exl3_gemv_kernel) with the same HFMA2 chain, pair sum, fp16 run sum and fold cadence. Contract: bit-identical to the
emulated MMA path (EXL3_MOE_COOP_M34=0, read per call) for the routed experts (persistent and two-launch kernels,
integer and half-integer rates, runs of 1..4 rows mixed) and for the shared expert (one expert, every row in one run).
Shapes are Qwen3.8-Flash-Next's (H 2560, intermediate 640, top-10). On other architectures the switch changes nothing.
"""
import os
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_scale, rand_trellis
from testlib.moe import ptr_table

H, I = 2560, 640


def _proj(k, n, K, E, gen, device):
    t = [rand_trellis(k, n, K, gen, device) for _ in range(E)]
    su = [rand_scale(k, gen, device) for _ in range(E)]
    sv = [rand_scale(n, gen, device) for _ in range(E)]
    return (ptr_table(t, device), ptr_table(su, device), ptr_table(sv, device)), (t, su, sv)


def _sel(bsz, topk, E, overlap, gen):
    # Rows share part of their picks, so expert runs of 1..bsz rows occur (MTP drafts route alike)
    base = torch.randperm(E, generator = gen)
    rows = []
    for r in range(bsz):
        own = torch.randperm(E, generator = gen)
        pick = list(base[:overlap].tolist())
        for e in own.tolist():
            if len(pick) == topk: break
            if e not in pick: pick.append(e)
        rows.append(torch.tensor(pick)[torch.randperm(topk, generator = gen)])
    return torch.stack(rows)


def _with_m34(v, fn):
    old = os.environ.get("EXL3_MOE_COOP_M34")
    os.environ["EXL3_MOE_COOP_M34"] = v
    try:
        return fn()
    finally:
        if old is None: del os.environ["EXL3_MOE_COOP_M34"]
        else: os.environ["EXL3_MOE_COOP_M34"] = old


def _case(device, seed, K, cb, bsz, E, topk, overlap, fused, shared):
    gen = torch.Generator().manual_seed(seed)
    keep = []
    g, kg = _proj(H, I, K, E, gen, device)
    u, ku = _proj(H, I, K, E, gen, device)
    d, kd = _proj(I, H, K, E, gen, device)
    x = (torch.randn((bsz, H), generator = gen) * 0.05).half().to(device)
    sel = _sel(bsz, topk, E, overlap, gen).to(device)
    rw = (torch.rand((bsz, topk), generator = gen) * 0.3).half().to(device)
    sh_out = torch.randn((bsz, H), generator = gen).float().to(device) if shared else None
    sh_w = (torch.randn((H,), generator = gen) * 0.05).half().to(device) if shared else None
    smax = 8 * topk
    had_g = torch.empty((smax, H), dtype = torch.half, device = device)
    had_u = torch.empty_like(had_g)
    gu_g = torch.empty((smax, 1, I), dtype = torch.half, device = device)
    gu_u = torch.empty_like(gu_g)
    act = torch.empty((smax, 1, I), dtype = torch.half, device = device)
    d_out = torch.empty((smax, 1, H), dtype = torch.float, device = device)
    base = smax * (I // 128) + bsz * (H // 128) + 2 * smax + 3
    ctr = torch.zeros((base + (smax + 2 if fused else 0),), dtype = torch.int, device = device)

    def run():
        out = torch.full((8, H), float("nan"), dtype = torch.float, device = device)
        ext.exl3_moe_coop(x, sel, rw, -1, -1, H, *g, *u, *d, None, None, None, K, K, K, cb == 1, cb == 2,
                          0, 0.0, True, had_g, had_u, gu_g, gu_u, act, d_out, ctr, out[:bsz], sh_out, sh_w)
        torch.cuda.synchronize()
        return out[:bsz].clone()

    ref = _with_m34("0", run)
    new = _with_m34("1", run)
    new2 = run()     # default (unset) = on
    assert torch.isfinite(ref).all()
    assert torch.equal(new, ref), f"m34 vs emulated MMA: max diff {(new - ref).abs().max().item():.3e}"
    assert torch.equal(new2, new)
    return ref


@pytest.mark.parametrize("bsz", [2, 3, 4])
@pytest.mark.parametrize("cb,K", [(2, 1.5), (2, 2), (2, 3), (2, 4), (1, 4), (2, 5), (0, 3), (2, 2.5)])
def test_routed_runs(device, cb, K, bsz):
    """Routed experts, top-10 of 48 with 5 shared picks per row: runs of 1..bsz rows in one call"""
    _case(device, 300 + int(K * 10) + 7 * cb + bsz, K, cb, bsz, E = 48, topk = 10, overlap = 5,
          fused = True, shared = bsz == 3)


@pytest.mark.parametrize("cb,K", [(2, 1.5), (2, 3)])
def test_routed_two_launch(device, cb, K):
    """Two-launch kernels (counter scratch without the queue tail), runs of up to 4 rows"""
    _case(device, 400 + int(K * 10), K, cb, 4, E = 24, topk = 10, overlap = 7, fused = False, shared = True)


@pytest.mark.parametrize("bsz", [2, 3, 4])
@pytest.mark.parametrize("cb,K", [(2, 5), (2, 4), (1, 5)])
def test_shared_expert(device, cb, K, bsz):
    """Shared expert as the model launches it: one expert, weight 1, every row in one run of bsz rows"""
    _case(device, 500 + 10 * K + cb + bsz, K, cb, bsz, E = 1, topk = 1, overlap = 1, fused = True, shared = False)
