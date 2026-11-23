"""fa60 head_dim 256 output hash for fixed inputs (bit-exactness across builds; run with EXL_SRC per build)"""
import sys, os, hashlib
sys.path.insert(0, os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3.ext import exllamav3_ext as ext

gen = torch.Generator(device = "cuda").manual_seed(5)
for (Tq, Tkv, Hq, Hkv, causal) in [(300, 1000, 24, 4, True), (1000, 1000, 8, 8, False), (65, 81, 24, 4, True)]:
    q = torch.randn(Tq, Hq, 256, device = "cuda", dtype = torch.half, generator = gen)
    k = torch.randn(Tkv, Hkv, 256, device = "cuda", dtype = torch.half, generator = gen)
    v = torch.randn(Tkv, Hkv, 256, device = "cuda", dtype = torch.half, generator = gen)
    o = torch.empty_like(q)
    ext.fa60_fwd(q, k, v, o, 256 ** -0.5, causal)
    torch.cuda.synchronize()
    print(Tq, Tkv, hashlib.sha1(o.cpu().numpy().tobytes()).hexdigest()[:16])
