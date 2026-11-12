"""In-process batch-1 decode benchmark (wall tokens/s, unprofiled), correctness oracle, and profiled window.

  CUDA_VISIBLE_DEVICES=1 .venv/bin/python util/bench_decode.py -m models/LFM3-1.7B-exl3-4.0bpw -cs 16384 -cq 8 \
      -depths 512 8192 -samplers greedy user -n 256 -reps 3 -json out.json

  -oracle out.json     greedy 128 tokens from a fixed prompt + final-step top-20 logits; then exit
  -profile trace.json  profiled decode window (-pn tokens, default 64) at the first depth, chrome trace written
                       to trace.json (analyze with util/analyze_decode_trace.py); wall time is NOT meaningful there
The "user" sampler is temp 0.3 top_p 0.9 top_k 50 repetition_penalty 1.05 (min_p 0), as in the TabbyAPI reference.
Decode time is taken between the first and last streamed token of a job, so prefill of the depth prefix is excluded.
"""
import sys, os, time, json, argparse, glob, statistics, subprocess

src = os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src)

import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import ArgmaxSampler

ORACLE_PROMPT = ("The history of the printing press began in the fifteenth century, when Johannes Gutenberg "
                 "introduced movable type to Europe. Over the following decades")


def corpus_text():
    files = []
    for pat in ("exllamav3/**/*.py", "exllamav3/**/*.cu*", "doc/*.md", "*.md", "tests/**/*.py", "examples/*.py"):
        files += sorted(glob.glob(os.path.join(src, pat), recursive = True))
    text = ""
    for f in files:
        try:
            text += open(f, errors = "ignore").read() + "\n"
        except Exception:
            pass
    return text


def smi_temp(idx):
    try:
        return subprocess.run(["nvidia-smi", "-i", str(idx), "--query-gpu=temperature.gpu,clocks.sm",
                               "--format=csv,noheader,nounits"], capture_output = True, text = True).stdout.strip()
    except Exception:
        return "n/a"


@torch.inference_mode()
def main(args):
    model, config, cache, tokenizer = model_init.init(args)[:4]
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
    smi_idx = 1 - int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    out = dict(model = args.model_dir, cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES"),
               graphs = os.environ.get("EXL3_GRAPHS", "1"), device = torch.cuda.get_device_name(0), runs = [])

    # ---------------------------------------------------------------- oracle
    if args.oracle:
        ids = tokenizer.encode(ORACLE_PROMPT, add_bos = True)
        job = Job(input_ids = ids, max_new_tokens = 128, stop_conditions = [], sampler = ArgmaxSampler(),
                  return_logits = True)
        generator.enqueue(job)
        toks, last_logits = [], None
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if "token_ids" in r and r["token_ids"] is not None:
                    toks += r["token_ids"].flatten().tolist()
                if "logits" in r and r["logits"] is not None:
                    last_logits = r["logits"].reshape(-1, r["logits"].shape[-1])[-1].float()
        v, i = torch.topk(last_logits, 20)
        res = dict(model = args.model_dir, prompt = ORACLE_PROMPT, graphs = out["graphs"], device = out["device"],
                   n_ids = len(toks), ids = toks, top20_ids = i.tolist(), top20_vals = v.tolist(),
                   text = tokenizer.decode(torch.tensor(toks)))
        json.dump(res, open(args.oracle, "w"), indent = 1)
        print(f"ORACLE {len(toks)} ids, top1 {i[0].item()} {v[0].item():.4f}; wrote {args.oracle}")
        print(res["text"][:300])
        return

    # ---------------------------------------------------------------- corpus prefix
    ids_all = tokenizer.encode(corpus_text(), add_bos = False)[0]
    need = max(args.depths) + 64
    assert ids_all.shape[0] > need * 4, f"corpus too short: {ids_all.shape[0]}"
    tail = tokenizer.encode("\n\nSummarize the code above and continue it:", add_bos = False)[0]

    def make_ids(depth, off):
        base = ids_all[off:off + depth]
        return torch.cat([base, tail]).unsqueeze(0)

    def get_sampler(name):
        return ArgmaxSampler() if name == "greedy" else model_init.get_arg_sampler(args)

    def run_job(ids, n, sampler):
        job = Job(input_ids = ids, max_new_tokens = n, stop_conditions = [], sampler = sampler)
        generator.enqueue(job)
        stamps = []
        while generator.num_remaining_jobs():
            rs = generator.iterate()
            now = time.perf_counter()
            if any(r.get("stage") == "streaming" for r in rs):
                stamps.append(now)
        return stamps  # one stamp per iterate() that streamed a token (no drafting); prefill chunks excluded

    # ---------------------------------------------------------------- profiled window
    if args.profile:
        depth = args.depths[0]
        sampler = get_sampler(args.samplers[0])
        ids = make_ids(depth, 0)
        job = Job(input_ids = ids, max_new_tokens = 4 * (args.pwarm + args.pn + 20), stop_conditions = [], sampler = sampler)
        generator.enqueue(job)
        # run through prefill + warm-up decode
        step = 0
        while generator.num_remaining_jobs() and step < args.pwarm:
            generator.iterate(); step += 1
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.pn):
                generator.iterate()
            torch.cuda.synchronize()
        prof.export_chrome_trace(args.profile)
        print(f"PROFILE wrote {args.profile} for {args.pn} decode steps at depth {depth}, sampler {args.samplers[0]}")
        return

    # ---------------------------------------------------------------- unprofiled decode benchmark
    run_job(make_ids(256, 0), 32, ArgmaxSampler())  # warmup (graph capture etc.)
    run_job(make_ids(256, 0), 32, model_init.get_arg_sampler(args))
    for depth in args.depths:
        for sname in args.samplers:
            reps = []
            for rep in range(args.reps):
                # distinct offset per rep: no prefix-cache hit, a fresh prefill each time
                off = (1 + rep + 3 * args.depths.index(depth)) * 1000
                ids = make_ids(depth, off)
                pre = smi_temp(smi_idx)
                st = run_job(ids, args.n + 1, get_sampler(sname))
                post = smi_temp(smi_idx)
                # st[0] is the iterate that finishes prefill and emits token 1 (excluded); decode = later stamps
                dts = [b - a for a, b in zip(st[1:-1], st[2:])]
                tot = st[-1] - st[1]
                ntok = len(st) - 2
                ms = 1000 * tot / ntok
                reps.append(dict(depth = depth, sampler = sname, tokens = ntok, ms_per_tok = ms, tps = 1000 / ms,
                                 step_p50_ms = 1000 * statistics.median(dts), step_p90_ms = 1000 * sorted(dts)[int(.9 * len(dts))],
                                 smi = f"{pre} -> {post}"))
                print(f"  depth {depth:5d} {sname:6s} rep {rep}: {ntok} tok {ms:.3f} ms/tok = {1000 / ms:.1f} t/s "
                      f"(p50 {reps[-1]['step_p50_ms']:.2f} p90 {reps[-1]['step_p90_ms']:.2f} ms) smi {pre} -> {post}", flush = True)
            t = [r["tps"] for r in reps]
            m = [r["ms_per_tok"] for r in reps]
            print(f"RESULT {out['device']} graphs={out['graphs']} {os.path.basename(args.model_dir)} depth {depth} {sname}: "
                  f"median {statistics.median(t):.1f} t/s ({statistics.median(m):.3f} ms/tok), "
                  f"min {min(t):.1f} max {max(t):.1f} (n={len(t)})", flush = True)
            out["runs"].append(dict(depth = depth, sampler = sname, median_tps = statistics.median(t),
                                    median_ms = statistics.median(m), min_tps = min(t), max_tps = max(t), reps = reps))
    if args.json:
        json.dump(out, open(args.json, "w"), indent = 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, add_sampling_args = True,
                        default_sampling_args = dict(temperature = 0.3, top_p = 0.9, top_k = 50,
                                                     repetition_penalty = 1.05, min_p = 0.0))
    parser.add_argument("-depths", type = int, nargs = "+", default = [512])
    parser.add_argument("-samplers", type = str, nargs = "+", default = ["greedy"], choices = ["greedy", "user"])
    parser.add_argument("-n", type = int, default = 256, help = "decode tokens per rep (>= 256)")
    parser.add_argument("-reps", type = int, default = 3)
    parser.add_argument("-json", type = str, default = None)
    parser.add_argument("-oracle", type = str, default = None)
    parser.add_argument("-profile", type = str, default = None)
    parser.add_argument("-pn", type = int, default = 64)
    parser.add_argument("-pwarm", type = int, default = 40, help = "iterate() calls before the profiled window (prefill chunks + decode warm-up)")
    args = parser.parse_args()
    main(args)
