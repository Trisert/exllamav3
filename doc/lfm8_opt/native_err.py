"""Max |error| of the native decode split (HFMA2 and fp32 variants) against the eager Triton decode, 8-bit cache,
relative to max |ref| (tests/kernels/attention/test_decode_native.py setup)"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "tests", "kernels", "attention"))
import torch
from test_decode_native import _setup, _native
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode

for hq, hk in ((16, 4), (16, 2)):
    for length in (255, 4095, 8000):
        q, kc, vc, bt, ks, vs = _setup(length + hq, hq, hk)
        sl = torch.tensor([length], dtype = torch.int32, device = "cuda")
        ref = paged_attn_triton_decode(q, None, None, kc, vc, bt, sl, causal = True, pre_appended_len = 1,
                                       qc = (ks, vs, 8, 8), n_kv_heads_override = hk).float()
        # fp64 reference from the dequantized cache would be better; Triton is the kernel being replaced
        row = []
        for fp32 in ("1", "0"):
            os.environ["EXL3_ATTN_NATIVE_FP32"] = fp32
            out = _native(q, kc, vc, bt, sl, ks, vs, hk, 28, True)
            row.append((out - ref).abs().max().item() / ref.abs().max().item())
        print(f"heads {hq}/{hk} length {length:5d}: rel max err fp32 {row[0]:.2e}  hfma2 {row[1]:.2e}")
