"""Probe the LFM3 attention gate projection: inner type, weight layout, and which kernel an m = 1 call launches."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3 import Config, Model
from exllamav3.modules.attn import Attention

with torch.inference_mode():
    config = Config.from_directory(sys.argv[1])
    model = Model.from_config(config)
    model.load()
    def walk(m):
        for s in getattr(m, "modules", []):
            if isinstance(s, Attention): return s
            r = walk(s)
            if r is not None: return r
    attn = walk(model)
    g = attn.g_proj
    inner = g.inner
    print("g_proj", type(inner).__name__, "in", g.in_features, "out", g.out_features, "headwise", attn.headwise_gate)
    w = inner.get_weight_tensor() if hasattr(inner, "get_weight_tensor") else None
    if w is not None:
        print("weight", tuple(w.shape), w.dtype, "stride", w.stride(), "contig", w.is_contiguous(), "ptr%16", w.data_ptr() % 16)
    x = torch.randn(1, 1, g.in_features, dtype = torch.half, device = w.device)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        for _ in range(3): g.forward(x, {})
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit = 6))
