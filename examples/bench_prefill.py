"""
Prompt-processing benchmark: prefill throughput at several prompt lengths, same model/cache arguments as chat.py.
Each prompt is real text (source files of this repository) cut at a different offset per run, so no run can reuse
the cache of an earlier one. One token is generated per prompt, the time is the generator's own time_prefill.

  python examples/bench_prefill.py -m <model> -gs 4.0,15.0 -cs 32768 -cq 4 [-lengths 512 2048 8192 16384 30000]
      [-repeat 2] [-smi_index 0] [-tag name]

Run it from the checkout you want to measure, or with EXL_SRC=<dir>. -mtp is not needed: prefill does not use the draft.
"""
import sys, os
src = os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src)

import argparse, glob, subprocess
import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import ArgmaxSampler


def smi(index):
    try:
        o = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
            capture_output = True, text = True
        ).stdout.strip()
        t, c = [x.strip() for x in o.split(",")]
        return f"{t} C {c} MHz"
    except Exception:
        return "n/a"


def corpus():
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


@torch.inference_mode()
def main(args):
    # init() returns the draft model, config and cache only when the draft arguments were added
    model, config, cache, tokenizer = model_init.init(args)[:4]
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
    ids_all = tokenizer.encode(corpus(), add_bos = False)[0]
    need = max(args.lengths) + 4096 * (len(args.lengths) * args.repeat + 1)
    assert ids_all.shape[0] >= need // 4, f"corpus too short: {ids_all.shape[0]} tokens"
    stride = max((ids_all.shape[0] - max(args.lengths) - 1) // (len(args.lengths) * args.repeat + 1), 1)

    def run(n, offset):
        ids = ids_all[offset:offset + n].unsqueeze(0)
        pre = smi(args.smi_index)
        job = Job(input_ids = ids, max_new_tokens = 1, sampler = ArgmaxSampler(), stop_conditions = [])
        generator.enqueue(job)
        last = {}
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("eos"):
                    last = r
        post = smi(args.smi_index)
        new_ctx = n - last.get("cached_tokens", 0)
        t = last.get("time_prefill", 1e-6) or 1e-6
        return new_ctx, t, pre, post

    run(256, 0)   # warm-up: kernel selection, autotune, Triton compile
    k = 1
    rates = {}
    for n in args.lengths:
        for r in range(args.repeat):
            new_ctx, t, pre, post = run(n, k * stride)
            k += 1
            rates.setdefault(n, []).append(new_ctx / t)
            print(f"PREFILL {args.tag} {n} tok: {new_ctx / t:.1f} t/s ({t:.2f} s, {new_ctx} uncached) | P100 {pre} -> {post}",
                  flush = True)
    print(f"SUMMARY {args.tag}: " + ", ".join(f"{n}: {sum(v) / len(v):.0f} t/s" for n, v in rates.items()), flush = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, add_draft_model_args = False)
    parser.add_argument("-lengths", "--lengths", type = int, nargs = "+", default = [512, 2048, 8192, 16384, 30000])
    parser.add_argument("-repeat", "--repeat", type = int, default = 2)
    parser.add_argument("-smi_index", "--smi_index", type = int, default = 0, help = "nvidia-smi index of the P100")
    parser.add_argument("-tag", "--tag", type = str, default = "run")
    main(parser.parse_args())
