"""
Non-interactive benchmark that mirrors examples/chat.py: same model/cache/MTP arguments, same prompt format, and the
same per-turn statistics (generate t/s, prompt t/s, draft acceptance), plus the P100 temperature and SM clock.
Greedy sampling, so two builds with bit-identical kernels generate the same tokens and the draft acceptance matches.

  python examples/bench_chat.py -m <model> -gs 4.0,15.0 -cs 32768 -cq 4 -mtp -ndt 1 -dds \
      [-mode qwen35] [-prompts "Implement a quicksort in go" "Implement quicksort in rust"] [-max_new 1000] [-smi_index 0]

Run it from the checkout you want to measure (the exllamav3 package and extension are imported from the parent of
this file), or with EXL_SRC=<dir>. Compare two builds by alternating them, see examples/bench_chat_ab.sh.
"""
import sys, os
src = os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src)
sys.path.insert(0, os.path.join(src, "examples"))

import argparse, subprocess, time
import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import ArgmaxSampler
from chat_templates import prompt_formats


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


@torch.inference_mode()
def main(args):
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(
        model = model, cache = cache, tokenizer = tokenizer,
        draft_model = draft_model, draft_cache = draft_cache,
        num_draft_tokens = args.num_draft_tokens, dynamic_draft_tokens = args.dynamic_draft,
        draft_confidence = args.draft_confidence,
    )
    fmt = prompt_formats[args.mode]("User", "Assistant")
    system_prompt = fmt.default_system_prompt(False)
    stop_conditions = [sc for sc in fmt.stop_conditions(tokenizer) if sc]
    if config.eos_token_id_list and all(config.eos_token_id_list):
        stop_conditions += config.eos_token_id_list

    context = []
    gen_rates, accept = [], []
    for i, user in enumerate(args.prompts):
        context.append((user, None))
        text = fmt.format(system_prompt, context, False)
        ids = tokenizer.encode(text, add_bos = fmt.add_bos(), encode_special_tokens = True)
        pre = smi(args.smi_index)
        job = Job(input_ids = ids, max_new_tokens = args.max_new, stop_conditions = stop_conditions,
                  sampler = ArgmaxSampler())
        generator.enqueue(job)
        out, last = "", {}
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                out += r.get("text", "") or ""
                if r.get("eos"):
                    last = r
        context[-1] = (user, out)
        post = smi(args.smi_index)
        new_tokens = last.get("new_tokens", 0)
        tps = new_tokens / (last.get("time_generate", 1e-6) or 1e-6)
        new_ctx = ids.shape[-1] - last.get("cached_tokens", 0)
        ptps = new_ctx / (last.get("time_prefill", 1e-6) or 1e-6)
        line = f"TURN {i+1}: prompt {new_ctx} tok at {ptps:.1f} t/s | generate {new_tokens} tok at {tps:.2f} t/s"
        if "accepted_draft_tokens" in last:
            a = last["accepted_draft_tokens"]
            d = a + last["rejected_draft_tokens"]
            line += f" | draft {a}/{d} ({100 * a / max(d, 1):.1f}%)"
            accept.append(a / max(d, 1))
        line += f" | P100 {pre} -> {post}"
        print(line, flush = True)
        gen_rates.append(tps)
    print(f"SUMMARY {args.tag}: mean generate {sum(gen_rates)/len(gen_rates):.2f} t/s over {len(gen_rates)} turns"
          + (f", mean draft acceptance {100*sum(accept)/len(accept):.1f}%" if accept else ""), flush = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, add_draft_model_args = True)
    parser.add_argument("-mode", "--mode", type = str, default = "qwen35")
    parser.add_argument("-prompts", "--prompts", nargs = "+",
                        default = ["Implement a quicksort in go", "Implement quicksort in rust"])
    parser.add_argument("-max_new", "--max_new", type = int, default = 1000)
    parser.add_argument("-smi_index", "--smi_index", type = int, default = 0, help = "nvidia-smi index of the P100")
    parser.add_argument("-tag", "--tag", type = str, default = "run")
    main(parser.parse_args())
