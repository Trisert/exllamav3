"""
ext.short_conv_decode (short_conv.cu): the fused decode / short-window core of ShortConv (LFM2 family).

Reference: the PyTorch glue of ShortConv.forward / _causal_conv1d, replicated verbatim below (fp16 b * x, cat with the
state, state write, cast to the conv weight dtype, depthwise F.conv1d, cast back to fp16, c * conv_out). The kernel
accumulates the taps in fp32 starting from the bias, in tap order, like ATen's depthwise kernel; the state must match
exactly (it only holds fp16 inputs) and so must the output. Measured on the RTX 2060 over
several hundred thousand elements per configuration: bf16 weights (the LFM2 checkpoints) are bit-identical to the torch
path with and without bias, fp16 weights are bit-identical without bias; fp16 weights with bias differ (the torch
path adds the bias in another order / precision), so ShortConv.forward does not take the fused path for that
combination and the test skips it.
"""

import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def reference(bc, state, slots, weight, bias, bsz, seqlen, history, ksz):
    dim = weight.shape[0]
    b, c, x = bc.view(bsz, seqlen, 3 * dim).chunk(3, dim = -1)
    conv_input = (b * x).transpose(1, 2).contiguous()
    ys = []
    for i, slot in enumerate(slots.tolist()):
        st = state[slot].unsqueeze(0)
        y = torch.cat([st[:, :, :ksz], conv_input[i].unsqueeze(0)], dim = -1)
        if history:
            write_size = min(state.shape[-1], y.shape[-1])
            state[slot, :, -write_size:].copy_(y[0, :, -write_size:])
        else:
            state[slot, :, :ksz].copy_(y[0, :, -ksz:])
        y = F.conv1d(y.to(weight.dtype), weight.unsqueeze(1), bias, padding = 0, groups = dim)
        ys.append(y[:, :, -seqlen:].to(conv_input.dtype))
    conv_out = torch.cat(ys, dim = 0).transpose(1, 2).contiguous()
    return c * conv_out


@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("bsz,seqlen,hist", [(1, 1, 0), (1, 1, 4), (3, 1, 0), (1, 3, 0), (2, 4, 4), (1, 16, 8), (2, 5, 0)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_short_conv_decode(wdtype, bsz, seqlen, hist, has_bias):
    if wdtype == torch.half and has_bias:
        pytest.skip("fp16 weight + bias: torch accumulation differs; ShortConv.forward keeps the torch path")
    torch.manual_seed(bsz * 100 + seqlen)
    dev = "cuda"
    dim, ksz, nslots = 2048, 7, 5
    S = ksz + hist
    weight = (torch.randn(dim, ksz, device = dev) * 0.4).to(wdtype)
    bias = (torch.randn(dim, device = dev) * 0.2).to(wdtype) if has_bias else None
    bc = (torch.randn(bsz * seqlen, 3 * dim, device = dev) * 1.5).half()
    state0 = (torch.randn(nslots, dim, S, device = dev) * 1.5).half()
    slots = torch.tensor([3, 0, 4][:bsz], dtype = torch.int32, device = dev)
    for history in ([False, True] if hist else [False]):
        st_ref = state0.clone()
        st_new = state0.clone()
        ref = reference(bc, st_ref, slots, weight, bias, bsz, seqlen, history, ksz)
        out = torch.empty(bsz * seqlen, dim, dtype = torch.half, device = dev)
        ext.short_conv_decode(bc, st_new, slots, weight, bias, out, bsz, seqlen, history)
        assert torch.equal(st_ref, st_new), "state mismatch"
        out = out.view(bsz, seqlen, dim)
        assert torch.equal(out, ref), f"{(out != ref).sum().item()} of {out.numel()} outputs differ"
