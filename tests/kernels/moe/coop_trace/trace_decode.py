import sys, os, time, collections
os.environ["EXL3_MOE_TRACE"] = "1"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
OUT = os.environ.get("COOP_TRACE_DIR", os.path.join(ROOT, "tmp", "coop_trace"))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
import torch, argparse
import numpy as np
from exllamav3 import model_init, Generator, Job
import exllamav3_ext as ext
assert ext.__file__.startswith(ROOT), ext.__file__

KIND = ["sharedA", "routedA", "sharedB", "routedB"]


def analyze(rec, name):
    if rec.shape[0] == 0:
        print(name, "no records"); return
    t0, t1, meta = rec[:, 0], rec[:, 1], rec[:, 2]
    order = np.argsort(t0, kind="stable")
    t0, t1, meta = t0[order], t1[order], meta[order]
    sm = meta & 0xff; kind = (meta >> 8) & 0xff; blk = (meta >> 16) & 0xffff; item = (meta >> 32)
    starts = np.nonzero(item == 0)[0]
    calls = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(t0) else len(t0)
        calls.append(slice(s, e))
    print(f"{name}: {len(calls)} kernel calls, {len(t0)} records")
    agg = collections.defaultdict(list)
    for c in calls:
        a0, a1, k, b = t0[c], t1[c], kind[c], blk[c]
        if len(a0) < 8: continue
        wall = a1.max() - a0.min()
        nblk = len(np.unique(b))
        busy = (a1 - a0).sum()
        agg["wall_us"].append(wall / 1000)
        agg["nitems"].append(len(a0))
        agg["nblk"].append(nblk)
        agg["util"].append(busy / (wall * nblk))
        for kk in range(4):
            m = k == kk
            if not m.any(): continue
            d = (a1[m] - a0[m]) / 1000
            agg[f"{KIND[kk]}_n"].append(m.sum())
            agg[f"{KIND[kk]}_mean_us"].append(d.mean())
            agg[f"{KIND[kk]}_max_us"].append(d.max())
            agg[f"{KIND[kk]}_short"].append((d < 3).sum())          # items that exit at once (no run)
            agg[f"{KIND[kk]}_start_us"].append((a0[m].min() - a0.min()) / 1000)
            agg[f"{KIND[kk]}_end_us"].append((a1[m].max() - a0.min()) / 1000)
    for k, v in agg.items():
        print(f"  {k:20s} mean {np.mean(v):9.2f}  p10 {np.percentile(v,10):9.2f}  p90 {np.percentile(v,90):9.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache=True, add_draft_model_args=True, add_sampling_args=True)
    args = parser.parse_args(["-m", os.path.join(ROOT, "models", "Ornith-1.5-35B-A3B-EXL3-4bpw"), "-gs", "4.0,15.0",
                              "-cs", "32768", "-cq", "4", "-mtp", "-ndt", "1", "-sdh", "16384",
                              "-temp", "0.7", "-topp", "0.8", "-topk", "20", "-minp", "0", "-presp", "1.5"])
    torch.set_grad_enabled(False); _im = torch.inference_mode(); _im.__enter__()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=args.num_draft_tokens, dynamic_draft_tokens=args.dynamic_draft,
                    draft_confidence=args.draft_confidence, sparse_draft_head=args.sparse_draft_head)
    prompt = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\nImplement a quicksort in go<|im_end|>\n<|im_start|>assistant\n"

    def run(n):
        job = Job(input_ids=tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True), max_new_tokens=n,
                  sampler=model_init.get_arg_sampler(args), stop_conditions=[], seed=1)
        gen.enqueue(job)
        its = 0
        t = time.time()
        while gen.num_remaining_jobs():
            gen.iterate(); its += 1
        return its, time.time() - t

    run(300)
    for d in (0, 1):
        ext.exl3_moe_coop_trace_dump(d)          # drop warm-up records
    its, dt = run(40)
    print(f"traced run: {its} iterations, {1000*dt/its:.1f} ms/iter", flush=True)
    for d, nm in ((1, "P100"), (0, "RTX2060")):
        rec = ext.exl3_moe_coop_trace_dump(d).numpy()
        np.save(os.path.join(OUT, f"coop_trace_{nm}.npy"), rec)
        analyze(rec, nm)
