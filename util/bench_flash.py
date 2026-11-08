"""Flash-Next end-to-end measurement harness: one load, then decode and/or prefill runs,
optionally with a torch.profiler window and with timing hooks on the n-gram row table.

  .venv/bin/python tmp/flash.py -m models/Qwen3.8-Flash-Next-exl3-1.66bpw -gs 4.0,15.0 -cs 24576 -cq 4 \
      -mcs 256 -mtp -ndt 3 -sdh 16384 -mode both -max_new 300 \
      [-prompts "Implement a quicksort in go"] [-lengths 2048 8192] [-repeat 2] [-tag x] \
      [-profile_decode] [-profile_prefill] [-hooks] [-json out.json]

Prints one line per measurement. cold prefill runs use a distinct corpus offset each.
"""
import sys, os, time, json, argparse, subprocess, glob

src = os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src)
sys.path.insert(0, os.path.join(src, "examples"))

import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import ArgmaxSampler
from chat_templates import prompt_formats


def smi(index = 0):
    try:
        o = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=temperature.gpu,clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output = True, text = True).stdout.strip()
        return o
    except Exception:
        return "n/a"


_corpus = None
def corpus():
    global _corpus
    if _corpus is None:
        files = []
        for pat in ("exllamav3/**/*.py", "exllamav3/**/*.cu*", "doc/*.md", "*.md", "tests/**/*.py", "examples/*.py"):
            files += sorted(glob.glob(os.path.join(src, pat), recursive = True))
        text = ""
        for f in files:
            try:
                text += open(f, errors = "ignore").read() + "\n"
            except Exception:
                pass
        _corpus = text
    return _corpus


# ---- timing hooks on the row tables (n-gram disk streaming) -----------------------------------
HOOKS = {}

def install_hooks():
    from exllamav3.modules.row_table import RowTable
    from exllamav3.modules.ngram_embedding import NGramEmbedding
    import functools
    def rec_of(key):
        return HOOKS.setdefault(key, {"n": 0, "t": 0.0, "sizes": [], "worst": []})
    def wrap(cls, name, key, tag_self = True):
        fn = getattr(cls, name)
        @functools.wraps(fn)
        def inner(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(self, *a, **kw)
            finally:
                r = rec_of((self.key.split(".")[-2] if "." in self.key else self.key, name))
                dt = time.perf_counter() - t0
                r["n"] += 1
                r["t"] += dt
                r["sizes"].append(a[0].shape[-1] if hasattr(a[0], "shape") else 0)
                r["worst"].append(round(1000 * dt, 1))
        setattr(cls, name, inner)
    wrap(RowTable, "_stage", "stage")
    wrap(RowTable, "_gather_rows", "gather")
    wrap(RowTable, "lookup", "lookup")
    wrap(RowTable, "prefetch", "prefetch")


def print_hooks(title = ""):
    if not HOOKS:
        return
    print(f"HOOKS {title}:")
    for k, v in sorted(HOOKS.items()):
        sizes = v["sizes"]
        worst = sorted(v["worst"], reverse = True)[:5]
        print(f"   {k}: n={v['n']} t={v['t']:.3f}s mean={1000 * v['t'] / max(v['n'], 1):.1f} ms "
              f"size/mean={sum(sizes) / max(len(sizes), 1):.0f} worst_ms={worst}")
    for v in HOOKS.values():
        v["n"] = 0
        v["t"] = 0.0
        v["sizes"] = []
        v["worst"] = []


class Rec:
    def __init__(self):
        self.rows = []
    def add(self, d):
        self.rows.append(d)


@torch.inference_mode()
def main(args):
    if args.hooks:
        install_hooks()

    t0 = time.time()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)[:7]
    load_s = time.time() - t0
    print(f"LOAD {load_s:.1f} s (incl. warmup)", flush = True)

    generator = Generator(
        model = model, cache = cache, tokenizer = tokenizer,
        draft_model = draft_model, draft_cache = draft_cache,
        num_draft_tokens = args.num_draft_tokens, dynamic_draft_tokens = args.dynamic_draft,
        draft_confidence = args.draft_confidence, sparse_draft_head = args.sparse_draft_head,
        hybrid_draft_tokens = args.hybrid_draft_tokens,
    )
    fmt = prompt_formats[args.mode]("User", "Assistant")
    system_prompt = fmt.default_system_prompt(False)
    stop_conditions = [sc for sc in fmt.stop_conditions(tokenizer) if sc]
    if config.eos_token_id_list and all(config.eos_token_id_list):
        stop_conditions += config.eos_token_id_list

    rec = Rec()
    sampler = model_init.get_arg_sampler(args) if getattr(args, "sample", False) else ArgmaxSampler()

    def decode_turn(user, context, max_new):
        context.append((user, None))
        text = fmt.format(system_prompt, context, False)
        ids = tokenizer.encode(text, add_bos = fmt.add_bos(), encode_special_tokens = True)
        pre = smi(args.smi_index)
        job = Job(input_ids = ids, max_new_tokens = max_new, stop_conditions = stop_conditions, sampler = sampler)
        generator.enqueue(job)
        out = ""
        last = {}
        steps = 0
        t0 = time.perf_counter()
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                out += r.get("text", "") or ""
                if r.get("eos"):
                    last = r
                steps += 1
        wall = time.perf_counter() - t0
        post = smi(args.smi_index)
        context[-1] = (user, out)
        new_tokens = last.get("new_tokens", 0)
        tg = last.get("time_generate", 1e-6) or 1e-6
        tp = last.get("time_prefill", 1e-6) or 1e-6
        new_ctx = ids.shape[-1] - last.get("cached_tokens", 0)
        d = dict(kind = "decode", prompt = user[:40], new_tokens = new_tokens, wall = wall, steps = steps,
                 t_s = new_tokens / tg, ms_step = 1000 * tg / max(steps, 1),
                 prefill_t_s = new_ctx / tp, new_ctx = new_ctx, smi = f"{pre} -> {post}")
        if "accepted_draft_tokens" in last:
            a = last["accepted_draft_tokens"]
            dd = a + last["rejected_draft_tokens"]
            d["accept"] = a / max(dd, 1)
            d["tok_step"] = (new_tokens - 1) / max(new_tokens - 1 - a, 1)
        sh = generator.sparse_head
        if sh is not None:
            d["sdh_miss"] = sh.miss_rate
            d["sdh" ] = (sh.mode_sparse, sh.mode_switches)
        rec.add(d)
        print(f"DECODE {new_tokens} tok at {d['t_s']:.2f} t/s | {d['ms_step']:.1f} ms/step"
              + (f" | accept {100 * d.get('accept', 0):.1f}% {d.get('tok_step', 0):.2f} tok/step" if "accept" in d else "")
              + f" | prefill {new_ctx} tok at {d['prefill_t_s']:.0f} t/s | P100 {pre} -> {post}", flush = True)
        return d

    def prefill_run(n, offset):
        ids = ids_all[offset:offset + n].unsqueeze(0)
        pre = smi(args.smi_index)
        job = Job(input_ids = ids, max_new_tokens = 1, stop_conditions = [], sampler = ArgmaxSampler())
        generator.enqueue(job)
        last = {}
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("eos"):
                    last = r
        post = smi(args.smi_index)
        new_ctx = n - last.get("cached_tokens", 0)
        t = last.get("time_prefill", 1e-6) or 1e-6
        d = dict(kind = "prefill", n = n, new_ctx = new_ctx, t_s = new_ctx / t, seconds = t, smi = f"{pre} -> {post}")
        rec.add(d)
        print(f"PREFILL {n} tok: {new_ctx} uncached at {d['t_s']:.1f} t/s ({t:.2f} s) | P100 {pre} -> {post}", flush = True)

    def profile(kind, fn):
        with torch.profiler.profile(
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes = False, profile_memory = False,
        ) as prof:
            fn()
        print(f"PROFILE {kind} top CUDA:")
        rows = prof.key_averages().table(sort_by = "self_cuda_time_total", row_limit = 45)
        print(rows)
        if args.profile_file:
            with open(args.profile_file, "a") as f:
                f.write(f"=== {kind} ===\n{rows}\n")
        print(f"PROFILE {kind} top CPU:")
        print(prof.key_averages().table(sort_by = "self_cpu_time_total", row_limit = 25))

    ids_all = tokenizer.encode(corpus(), add_bos = False)[0]
    print(f"CORPUS {ids_all.shape[0]} tokens", flush = True)

    if args.run == "prefill" or args.run == "both":
        need = max(args.lengths) + 4096 * (len(args.lengths) * args.repeat + 2)
        assert ids_all.shape[0] >= need // 4, f"corpus too short: {ids_all.shape[0]} tokens"
        stride = max((ids_all.shape[0] - max(args.lengths) - 1) // (len(args.lengths) * args.repeat + 2), 1)

    # warm-up: kernel selection, autotune, Triton compile (torch compile and gemm autotune), 256 tok
    prefill_run(256, 0)
    if args.hooks:
        print_hooks("warmup")

    if args.run == "decode" or args.run == "both":
        if args.profile_decode:
            # a fixed decode window recorded by the profiler schedule: profile_from warm-up steps, 60 active
            context = [(args.prompts[0], "")]
            text = fmt.format(system_prompt, context, False)
            ids = tokenizer.encode(text, add_bos = fmt.add_bos(), encode_special_tokens = True)
            job = Job(input_ids = ids, max_new_tokens = args.profile_from + 70, stop_conditions = [], sampler = sampler)
            generator.enqueue(job)
            with torch.profiler.profile(
                activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule = torch.profiler.schedule(wait = 0, warmup = args.profile_from, active = 60, repeat = 1),
            ) as prof:
                while generator.num_remaining_jobs():
                    for r in generator.iterate():
                        pass
                    prof.step()
            print("PROFILE decode top CUDA:")
            print(prof.key_averages().table(sort_by = "self_cuda_time_total", row_limit = 45))
            print("PROFILE decode top CPU:")
            print(prof.key_averages().table(sort_by = "self_cpu_time_total", row_limit = 25))

        context = []
        for i, user in enumerate(args.prompts):
            if args.repeat > 1:
                for r in range(args.repeat):
                    context2 = list(context)
                    decode_turn(user, context2, args.max_new)
                    context = context2
            else:
                decode_turn(user, context, args.max_new)
        if args.hooks:
            print_hooks("decode")

    if args.run == "prefill" or args.run == "both":
        k = 1
        rates = {}
        for n in args.lengths:
            for r in range(args.repeat):
                if args.profile_prefill and k == 1:
                    with torch.profiler.profile(
                        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    ) as prof:
                        prefill_run(n, k * stride)
                    print("PROFILE prefill top CUDA:")
                    print(prof.key_averages().table(sort_by = "self_cuda_time_total", row_limit = 45))
                    print("PROFILE prefill top CPU:")
                    print(prof.key_averages().table(sort_by = "self_cpu_time_total", row_limit = 30))
                else:
                    prefill_run(n, k * stride)
                k += 1
        if args.hooks:
            print_hooks("prefill")

    dec = [d["t_s"] for d in rec.rows if d["kind"] == "decode"]
    pp = {}
    for d in rec.rows:
        if d["kind"] == "prefill":
            pp.setdefault(d["n"], []).append(d["t_s"])
    print(f"SUMMARY {args.tag}: decode {sum(dec) / len(dec):.2f} t/s over {len(dec)} turns" if dec else f"SUMMARY {args.tag}: no decode")
    if pp:
        print("SUMMARY prefill: " + ", ".join(f"{n}: {sum(v) / len(v):.0f} t/s" for n, v in pp.items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(tag = args.tag, load_s = load_s, rows = rec.rows, hooks = HOOKS), f, indent = 1)
        print(f"JSON {args.json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, add_draft_model_args = True, add_sampling_args = True)
    parser.add_argument("-mode", "--mode", type = str, default = "qwen35")
    parser.add_argument("-run", "--run", type = str, default = "both", choices = ["decode", "prefill", "both"])
    parser.add_argument("-prompts", "--prompts", nargs = "+", default = ["Implement a quicksort in go"])
    parser.add_argument("-max_new", "--max_new", type = int, default = 300)
    parser.add_argument("-repeat", "--repeat", type = int, default = 1)
    parser.add_argument("-lengths", "--lengths", type = int, nargs = "+", default = [2048, 8192])
    parser.add_argument("-smi_index", "--smi_index", type = int, default = 0)
    parser.add_argument("-tag", "--tag", type = str, default = "run")
    parser.add_argument("-tag2", "--tag2", type = str, default = None)
    parser.add_argument("-json", "--json", type = str, default = None)
    parser.add_argument("-profile_file", "--profile_file", type = str, default = None)
    parser.add_argument("-profile_decode", "--profile_decode", action = "store_true")
    parser.add_argument("-profile_prefill", "--profile_prefill", action = "store_true")
    parser.add_argument("-profile_from", "--profile_from", type = int, default = 60)
    parser.add_argument("-hooks", "--hooks", action = "store_true")
    args = parser.parse_args()
    main(args)
