"""Realistic agentic-session benchmark for Qwen3.8-Flash-Next (the user's workload: code, multi-turn,
long context). Measures prefill and decode per turn of a growing conversation and the session totals,
on the same model/cache/draft arguments as chat.py.

  .venv/bin/python util/bench_session.py -m models/Qwen3.8-Flash-Next-exl3-1.66bpw -gs 4.0,15.0 \
      -cs 24576 -cq 4 -mcs 256 -mtp -ndt 3 -sdh 16384 -workload code-edit -turns 4 -max_new 400

Workloads:
  code-edit  a large source excerpt in the prompt, "reproduce it with one function renamed" (copy-heavy
             decode: the case that benefits from -hdt)
  new-code   "implement X" with a growing code context (free-form code decode)
  prose      explanation of a document (prose decode: low draft acceptance, the worst case)
Each turn appends the model's reply and a synthetic tool result, so the context grows like a real agent
session. Prints per-turn lines and a summary with the session-average decode/prefill rates and the
fraction of session wall time spent in prefill.
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
        return subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=temperature.gpu,clocks.sm",
                               "--format=csv,noheader,nounits"], capture_output = True, text = True).stdout.strip()
    except Exception:
        return "n/a"


def read_repo_text(max_chars):
    out = ""
    for pat in ("exllamav3/modules/*.py", "exllamav3/generator/*.py", "examples/*.py", "doc/*.md"):
        for f in sorted(glob.glob(os.path.join(src, pat))):
            try:
                out += f"\n\n# ===== {os.path.basename(f)} =====\n" + open(f, errors = "ignore").read()
            except Exception:
                pass
            if len(out) >= max_chars:
                return out[:max_chars]
    return out[:max_chars]


WORKLOADS = {
    "code-edit": dict(
        excerpt = 7000,
        task = ("The repository excerpt below is in the context. Reproduce the function "
                "exllamav3/generator/generator.py's iterate() skeleton from it, with every occurrence of "
                "the identifier 'iterate' renamed to 'advance', and nothing else changed. Output the code only."),
        tool = "ERROR: test_generator_step failed: expected 3 rows, got 2\n  at tests/generator/test_step.py:118",
    ),
    "new-code": dict(
        excerpt = 2000,
        task = ("Here is part of the exllamav3 codebase. Implement a new module "
                "exllamav3/modules/histogram.py with a streaming quantile estimator for fp16 tensors, "
                "AVX2-friendly, plus a short docstring. Output the code only."),
        tool = "ERROR: torch not found in the isolated build environment",
    ),
    "prose": dict(
        excerpt = 5000,
        task = ("Read the project documentation excerpt in the context and explain, in prose, how the "
                "Pascal fork's MoE cooperative kernels work and why the m = 4 case is harder. "
                "Do not write code."),
        tool = "The user says: go on, more detail on the shared expert and the counters.",
    ),
}


@torch.inference_mode()
def main(args):
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
    sampler = model_init.get_arg_sampler(args) if getattr(args, "sample", False) else ArgmaxSampler()

    w = WORKLOADS[args.workload]
    excerpt = read_repo_text(w["excerpt"])
    context = []
    turns = []
    for t in range(args.turns):
        if t == 0:
            user = excerpt + "\n\n" + w["task"]
        elif t == 1:
            user = w["tool"] + "\n\nFix it and continue."
        else:
            user = w["tool"] + f"\n\nStill not satisfied. Iterate again ({t})."
        context.append((user, None))
        text = fmt.format(system_prompt, context, False)
        ids = tokenizer.encode(text, add_bos = fmt.add_bos(), encode_special_tokens = True)
        pre = smi(args.smi_index)
        job = Job(input_ids = ids, max_new_tokens = args.max_new, stop_conditions = stop_conditions, sampler = sampler)
        generator.enqueue(job)
        out = ""
        last = {}
        steps = 0
        ts = time.perf_counter()
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                out += r.get("text", "") or ""
                if r.get("eos"):
                    last = r
                steps += 1
        wall = time.perf_counter() - ts
        post = smi(args.smi_index)
        context[-1] = (user, out)
        nt = last.get("new_tokens", 0)
        tg = last.get("time_generate", 1e-6) or 1e-6
        tp = last.get("time_prefill", 1e-6) or 1e-6
        new_ctx = ids.shape[-1] - last.get("cached_tokens", 0)
        d = dict(turn = t + 1, ctx = ids.shape[-1], new_ctx = new_ctx, reply = nt, wall = wall,
                 prefill_s = tp, decode_s = tg, prefill_t_s = new_ctx / tp, decode_t_s = nt / tg,
                 ms_step = 1000 * tg / max(steps, 1), smi = f"{pre} -> {post}")
        if "accepted_draft_tokens" in last:
            a = last["accepted_draft_tokens"]
            dd = a + last["rejected_draft_tokens"]
            d["accept"] = a / max(dd, 1)
            d["tok_step"] = (nt - 1) / max(nt - 1 - a, 1)
        turns.append(d)
        print(f"TURN {d['turn']}: ctx {d['ctx']} (+{new_ctx} new) | prefill {d['prefill_t_s']:.0f} t/s ({tp:.1f} s) | "
              f"decode {d['reply']} tok at {d['decode_t_s']:.2f} t/s ({d['ms_step']:.1f} ms/step"
              + (f", accept {100 * d['accept']:.0f}% {d['tok_step']:.2f} tok/step" if "accept" in d else "")
              + f") | P100 {pre} -> {post}", flush = True)

    wall = sum(d["wall"] for d in turns)
    pf = sum(d["prefill_s"] for d in turns)
    dec = [d["decode_t_s"] for d in turns]
    print(f"SUMMARY {args.tag} {args.workload}: {args.turns} turns, session wall {wall:.1f} s, "
          f"prefill {pf:.1f} s ({100 * pf / wall:.0f}%), mean decode {sum(dec) / len(dec):.2f} t/s, "
          f"session throughput {sum(d['new_ctx'] + d['reply'] for d in turns) / wall:.1f} tok/s", flush = True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(tag = args.tag, workload = args.workload, load_s = load_s, turns = turns), f, indent = 1)
        print(f"JSON {args.json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, add_draft_model_args = True, add_sampling_args = True)
    parser.add_argument("-mode", "--mode", type = str, default = "qwen35")
    parser.add_argument("-workload", "--workload", type = str, default = "code-edit", choices = list(WORKLOADS))
    parser.add_argument("-turns", "--turns", type = int, default = 3)
    parser.add_argument("-max_new", "--max_new", type = int, default = 400)
    parser.add_argument("-smi_index", "--smi_index", type = int, default = 0)
    parser.add_argument("-tag", "--tag", type = str, default = "run")
    parser.add_argument("-json", "--json", type = str, default = None)
    main(parser.parse_args())
