"""
EXL3_ROUTER_FUSED: the router projection (hgemv) and the softmax top-k of routing_std in one launch, the top-k
running in the hgemv block that holds the final logits. It must be bit-identical to the two launches (hgemv +
routing_std_topk_kernel): logits, selected experts and weights. Covers 2..4 rows (the hgemm path; one row takes
routing_gemv_kernel and is never fused), split and unsplit K, expert counts up to 256, with and without the
per-expert scale. Runs where the hgemv path is on (sm_60, sm_75).
"""

import os
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

CONFIGS = [(m, k, e, topk) for m in (2, 3, 4) for k in (64, 512, 2048, 4096) for e, topk in ((256, 8), (128, 8), (64, 4), (32, 2))]


def _hgemv_on():
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    return cc in ((6, 0), (7, 5)) and os.environ.get("EXL3_HGEMV", "") != "0"


def _run(x, gate, gate_t, rows, e, topk, pes, fused):
    old = os.environ.get("EXL3_ROUTER_FUSED")
    os.environ["EXL3_ROUTER_FUSED"] = "1" if fused else "0"
    try:
        scores = torch.empty((rows, e), dtype = torch.half, device = x.device)
        idx = torch.empty((rows, topk), dtype = torch.long, device = x.device)
        w = torch.empty((rows, topk), dtype = torch.half, device = x.device)
        ext.routing_std(x, gate, scores, idx, w, pes, gate_t, None, None, None)
        torch.cuda.synchronize()
        return scores, idx, w
    finally:
        if old is None: del os.environ["EXL3_ROUTER_FUSED"]
        else: os.environ["EXL3_ROUTER_FUSED"] = old


@pytest.mark.skipif(not _hgemv_on(), reason = "hgemv path is off on this device")
@pytest.mark.parametrize("rows, k, e, topk", CONFIGS)
@pytest.mark.parametrize("scaled", (False, True))
@torch.inference_mode()
def test_router_fused_bit_identical(rows, k, e, topk, scaled):
    gen = torch.Generator().manual_seed(rows * 1000 + k + e)
    x = torch.randn((rows, k), generator = gen).half().cuda()
    gate = (torch.randn((k, e), generator = gen) / k ** 0.5).half().cuda()
    gate_t = gate.T.contiguous()
    pes = (torch.rand((e,), generator = gen) + 0.5).bfloat16().cuda() if scaled else None
    s0, i0, w0 = _run(x, gate, gate_t, rows, e, topk, pes, False)
    for _ in range(3):   # the split counters must be left at zero
        s1, i1, w1 = _run(x, gate, gate_t, rows, e, topk, pes, True)
        assert torch.equal(s0, s1)
        assert torch.equal(i0, i1)
        assert torch.equal(w0, w1)
