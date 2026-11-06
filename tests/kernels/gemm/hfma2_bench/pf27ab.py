# In-process ABAB of 27B prefill: the model is loaded once and ext.hgemm_recon is routed, on the P100 for the
# shapes hgemm_hfma2_try takes, to the scratch module's old kernel (v0, A) or new kernel (v4, B; same code as the
# extension's new kernel). Alternates A / B per 2048-token prompt so both see the same thermal drift.
import sys, os, time, subprocess, collections, argparse
src = os.environ["EXL_SRC"]
sys.path.insert(0, src); sys.path.insert(0, os.path.join(src, "examples"))
import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.ext import exllamav3_ext as ext
import bench_prefill
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
vm = load(name = "hfma2_vars", sources = [os.path.join(here, "vars.cu")],
          extra_cuda_cflags = ["-O3", "-gencode=arch=compute_60,code=sm_60", "-Xptxas=-v", "-ccbin=/usr/bin/g++-14",
                               "--use_fast_math", "-Xptxas=-dlcm=cg", "-std=c++20"], extra_include_paths = [here])
ap = argparse.ArgumentParser()
model_init.add_args(ap, cache = True)
ap.add_argument("-n", type = int, default = 2048)
ap.add_argument("-pairs", type = int, default = 2)
ap.add_argument("-start", type = int, default = 0)
args = ap.parse_args(["-m", "/home/nicola/exllamav3/models/Qwen3.8-27B-exl3-4.00bpw", "-gs", "4.0,15.0", "-cs", "32768", "-cq", "4"] + sys.argv[1:])
bench_prefill.src = src
torch.set_grad_enabled(False); _im = torch.inference_mode(); _im.__enter__()

orig = ext.hgemm_recon
mode = {"v": 0}
calls = collections.Counter()
def routed(a, b, c):
    if (a.device.type == "cuda" and torch.cuda.get_device_capability(a.device) == (6, 0) and a.dim() == 2 and
        a.is_contiguous() and b.is_contiguous() and c.is_contiguous() and a.dtype == torch.half and b.dtype == torch.half and
        a.shape[1] % 32 == 0 and b.shape[1] % 128 == 0 and ((a.shape[0] + 63) // 64) * (b.shape[1] // 128) >= 96):
        calls[mode["v"]] += 1
        vm.run(mode["v"], a, b, c, 64)
        return
    calls["orig"] += 1
    orig(a, b, c)
ext.hgemm_recon = routed

t0 = time.time()
model, config, cache, tokenizer = model_init.init(args)
print(f"loaded in {time.time() - t0:.0f} s", flush = True)
ids_all = tokenizer.encode(bench_prefill.corpus(), add_bos = False)[0]
g = Generator(model = model, cache = cache, tokenizer = tokenizer)

def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu,clocks.sm", "--format=csv,noheader"],
                          capture_output = True, text = True).stdout.strip().replace("\n", " | ")
off = [1000]
def take(n):
    o = off[0]; off[0] += n + 997
    if off[0] + 1000 > ids_all.shape[0]: o = 1000; off[0] = o + n + 997
    return ids_all[o:o + n].unsqueeze(0)
def run(ids):
    job = Job(input_ids = ids, max_new_tokens = 1, sampler = ArgmaxSampler(), stop_conditions = [])
    g.enqueue(job); last = {}
    while g.num_remaining_jobs():
        for r in g.iterate():
            if r.get("eos"): last = r
    torch.cuda.synchronize()
    return last

mode["v"] = 4; run(take(2048))   # warm-up (Triton autotune, both kernels)
print("routing", dict(calls), flush = True)
for p in range(args.start, args.start + args.pairs):
    for v, tag in ((0, "A_old") if p % 2 == 0 else (4, "B_new"), (4, "B_new") if p % 2 == 0 else (0, "A_old")):
        mode["v"] = v
        s0 = smi(); last = run(take(args.n))
        nn = args.n - last.get("cached_tokens", 0); t = last.get("time_prefill", 1e-6)
        print(f"PREFILL {tag} pair {p} {nn} tok: {nn / t:.1f} t/s ({t:.2f} s) | {s0} -> {smi()}", flush = True)
print("routing", dict(calls), flush = True)
