"""Agent-style rewrite turns: a source file in the prompt, 'reproduce it with a small change'. Greedy, MTP.
Compares hybrid drafting off / on in one load (hybrid_draft_tokens is read per round; buffers are sized for the max)."""
import sys, os, time, argparse
src = "/home/nicola/exllamav3/.claude/worktrees/draft-penalty"
sys.path.insert(0, src)
sys.path.insert(0, src + "/examples")
import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import ArgmaxSampler
from chat_templates import prompt_formats
import bench_prefill as bp


def main():
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache=True, add_draft_model_args=True, add_sampling_args=True)
    parser.add_argument("-chars", type=int, default=7000)
    parser.add_argument("-max_new", type=int, default=500)
    parser.add_argument("-rounds", type=int, default=4)
    parser.add_argument("-chat", action="store_true")
    args = parser.parse_args()
    with torch.inference_mode():
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model, draft_cache=draft_cache,
                        num_draft_tokens=args.num_draft_tokens, dynamic_draft_tokens=args.dynamic_draft,
                        draft_confidence=args.draft_confidence, sparse_draft_head=args.sparse_draft_head,
                        ngram_match_min=args.ngram_match_min, hybrid_draft_tokens=args.hybrid_draft_tokens)
        hdt_max = args.hybrid_draft_tokens
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
            prompts = ["Implement a quicksort in go with tests.", "Write a rust function that parses a CSV line with quoted fields and explain it.",
                       "Explain how a B-tree insert works and give python code.", "Write a bash script that backs up a directory and rotates old archives."]
            user = prompts[k % 4] if args.chat else ("Here is a source file excerpt. Reproduce it exactly, in full, as a single code block, but rename the "
                    "first function you find to `renamed_function`. Output only the code block.\n\n```\n" + code + "\n```")
            text = fmt.format(sysp, [(user, None)], False)
            ids = tokenizer.encode(text, add_bos=fmt.add_bos(), encode_special_tokens=True)
            gen.pagetable.reset_page_table()
            job = Job(input_ids=ids, max_new_tokens=args.max_new, stop_conditions=stops, sampler=ArgmaxSampler())
            gen.enqueue(job)
            last = {}
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("eos"): last = r
            n, t = last.get("new_tokens", 0), last.get("time_generate", 1e-9)
            acc = last.get("accepted_draft_tokens", 0); rej = last.get("rejected_draft_tokens", 0)
            print(f"RW hdt={hdt:2d} k={k} ctx {ids.shape[-1]} | decode {n} tok {n / t:.1f} t/s | draft {acc}/{acc + rej}", flush=True)
            return n / t

        turn(0, 99)  # warm-up
        res = {}
        modes = [0] + [h for h in (3, 5, 7) if h <= hdt_max]
        k = 0
        for rd in range(args.rounds):
            order = modes if rd % 2 == 0 else modes[::-1]
            for m in order:
                res.setdefault(m, []).append(turn(m, 100 + rd))
        for m, v in res.items():
            print(f"RW SUMMARY hdt={m}: mean {sum(v) / len(v):.1f} t/s over {len(v)} runs {[round(x, 1) for x in v]}", flush=True)


if __name__ == "__main__":
    main()
