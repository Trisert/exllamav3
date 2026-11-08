"""Agent-style rewrite turns: a source file in the prompt, 'reproduce it with a small change'. Greedy, MTP.

Compares hybrid drafting widths and/or the m = 9..16 GEMV path (EXL3_GEMV_M16) in one load: hybrid_draft_tokens and
the env switch are read per round, buffers are sized for the largest width passed to -hdt (>= max of -hdts).

  .venv/bin/python util/bench_rewrite.py -m models/Qwen3.8-Flash-Next-exl3-1.66bpw -gs 4.0,15.0 -cs 24576 -cq 4 \
      -mcs 256 -mtp -ndt 3 -sdh 16384 -hdt 15 -hdts 7 12 15 -rounds 2 -m16ab -chars 7000 -max_new 500 [-json out.json]

ABAB: each round runs the modes in one order and the next round in the reverse; with -m16ab every mode runs with the
new path on and off, alternating which side goes first.
"""
import sys, os, time, json, argparse
src = os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src)
sys.path.insert(0, src + "/examples")
import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import ArgmaxSampler
from chat_templates import prompt_formats
import bench_prefill as bp


def smi(index = 0):
    try:
        import subprocess
        o = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=temperature.gpu,clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output = True, text = True).stdout.strip()
        return o
    except Exception:
        return "n/a"


def main():
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, add_draft_model_args = True, add_sampling_args = True)
    parser.add_argument("-chars", type = int, default = 7000)
    parser.add_argument("-max_new", type = int, default = 500)
    parser.add_argument("-rounds", type = int, default = 4)
    parser.add_argument("-hdts", type = int, nargs = "+", default = [0, 3, 5, 7],
                        help = "hybrid draft widths to compare (hdt 0 = regular MTP drafting only); -hdt sizes the buffers")
    parser.add_argument("-m16ab", action = "store_true",
                        help = "also A/B the m = 9..16 GEMV path (EXL3_GEMV_M16 1 vs 0) on every mode")
    parser.add_argument("-chat", action = "store_true", help = "free-form chat prompts instead of the rewrite prompt")
    parser.add_argument("-json", type = str, default = None)
    parser.add_argument("-smi_index", type = int, default = 0)
    args = parser.parse_args()
    assert args.hybrid_draft_tokens >= max(args.hdts), "-hdt must be at least the largest -hdts width (buffer sizing)"

    with torch.inference_mode():
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
        gen = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft_model,
                        draft_cache = draft_cache, num_draft_tokens = args.num_draft_tokens,
                        dynamic_draft_tokens = args.dynamic_draft,
                        draft_confidence = args.draft_confidence, sparse_draft_head = args.sparse_draft_head,
                        ngram_match_min = args.ngram_match_min, hybrid_draft_tokens = args.hybrid_draft_tokens)
        fmt = prompt_formats["qwen35"]("User", "Assistant")
        sysp = fmt.default_system_prompt(False)
        stops = [sc for sc in fmt.stop_conditions(tokenizer) if sc]
        if config.eos_token_id_list and all(config.eos_token_id_list):
            stops += config.eos_token_id_list
        corpus = bp.corpus()
        lines = corpus.split("\n")

        def snippet(k):
            start = (k * 1237) % max(len(lines) - 400, 1)
            out, n = [], 0
            for ln in lines[start:]:
                out.append(ln); n += len(ln) + 1
                if n >= args.chars: break
            return "\n".join(out)

        def turn(hdt, k):
            gen.hybrid_draft_tokens = hdt
            code = snippet(k)
            prompts = ["Implement a quicksort in go with tests.",
                       "Write a rust function that parses a CSV line with quoted fields and explain it.",
                       "Explain how a B-tree insert works and give python code.",
                       "Write a bash script that backs up a directory and rotates old archives."]
            user = prompts[k % 4] if args.chat else (
                "Here is a source file excerpt. Reproduce it exactly, in full, as a single code block, but rename the "
                "first function you find to `renamed_function`. Output only the code block.\n\n```\n" + code + "\n```")
            text = fmt.format(sysp, [(user, None)], False)
            ids = tokenizer.encode(text, add_bos = fmt.add_bos(), encode_special_tokens = True)
            gen.pagetable.reset_page_table()
            job = Job(input_ids = ids, max_new_tokens = args.max_new, stop_conditions = stops, sampler = ArgmaxSampler())
            gen.enqueue(job)
            last = {}
            steps = 0
            pre = smi(args.smi_index)
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    steps += 1
                    if r.get("eos"): last = r
            post = smi(args.smi_index)
            n, t = last.get("new_tokens", 0), last.get("time_generate", 1e-9)
            acc = last.get("accepted_draft_tokens", 0); rej = last.get("rejected_draft_tokens", 0)
            m16 = os.environ.get("EXL3_GEMV_M16", "1")
            d = dict(kind = "turn", hdt = hdt, m16 = m16, k = k, ctx = int(ids.shape[-1]), new = n,
                     t_s = n / t, ms_step = 1000 * t / max(steps, 1), steps = steps,
                     accepted = acc, drafted = acc + rej, text = last.get("text", "") or "", smi = f"{pre} -> {post}")
            print(f"RW hdt={hdt:2d} m16={m16} k={k} ctx {ids.shape[-1]} | {n} tok {d['t_s']:.2f} t/s "
                  f"| {d['ms_step']:.1f} ms/step | draft {acc}/{acc + rej} | P100 {pre} -> {post}", flush = True)
            return d

        def set_m16(v):
            if v is None:
                os.environ.pop("EXL3_GEMV_M16", None)
            else:
                os.environ["EXL3_GEMV_M16"] = v

        set_m16(None)
        turn(0, 99)  # warm-up (kernel selection, autotune, Triton compile)
        modes = list(args.hdts)
        sides = [None, "0"] if args.m16ab else [None]
        rows = []
        k = 0
        for rd in range(args.rounds):
            order = modes if rd % 2 == 0 else modes[::-1]
            for h in order:
                side_order = sides if rd % 2 == 0 else sides[::-1]
                for s in side_order:
                    set_m16(s)
                    rows.append(turn(h, 100 + rd))
                    k += 1
        set_m16(None)

        print("RW SUMMARY")
        groups = {}
        for d in rows:
            groups.setdefault((d["hdt"], d["m16"]), []).append(d)
        for key in sorted(groups, key = lambda x: (x[0], x[1])):
            v = groups[key]
            ts = [d["t_s"] for d in v]
            mss = [d["ms_step"] for d in v]
            print(f"RW SUMMARY hdt={key[0]:2d} m16={key[1]}: {sum(ts) / len(ts):.2f} t/s "
                  f"(runs {[round(x, 1) for x in ts]}), {sum(mss) / len(mss):.1f} ms/step, "
                  f"draft {sum(d['accepted'] for d in v)}/{sum(d['drafted'] for d in v)}", flush = True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(dict(args = vars(args), rows = rows), f, indent = 1)
            print(f"JSON {args.json}")


if __name__ == "__main__":
    main()
