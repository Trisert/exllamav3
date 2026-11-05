"""
Non-interactive benchmark that mirrors examples/chat.py: same model/cache/MTP arguments, same prompt format, and the
same per-turn statistics (generate t/s, prompt t/s, draft acceptance), plus the P100 temperature and SM clock.
Greedy sampling by default (-sample for the chat.py sampler), so two builds with bit-identical kernels generate the same tokens and the draft acceptance matches.

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
    t_load = time.time()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    print(f"LOAD {time.time() - t_load:.1f} s (incl. warmup)", flush = True)
    generator = Generator(
        model = model, cache = cache, tokenizer = tokenizer,
        draft_model = draft_model, draft_cache = draft_cache,
        num_draft_tokens = args.num_draft_tokens, dynamic_draft_tokens = args.dynamic_draft,
        draft_confidence = args.draft_confidence,
        sparse_draft_head = args.sparse_draft_head,
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
        if args.cool:
            # wait until the P100 is at or below args.cool C before the turn (thermal throttling at 79 C otherwise
            # dominates long replies)
            while True:
                try:
                    t = int(subprocess.run(["nvidia-smi", "-i", str(args.smi_index), "--query-gpu=temperature.gpu",
                                            "--format=csv,noheader,nounits"], capture_output = True, text = True).stdout.strip())
                except Exception:
                    break
                if t <= args.cool: break
                time.sleep(3)
        pre = smi(args.smi_index)
        job = Job(input_ids = ids, max_new_tokens = args.max_new, stop_conditions = stop_conditions,
                  sampler = model_init.get_arg_sampler(args) if args.sample else ArgmaxSampler())
        generator.enqueue(job)
        out, last = "", {}
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                out += r.get("text", "") or ""
                if r.get("eos"):
                    last = r
        context[-1] = (user, out)
        if args.save_text:
            with open(args.save_text, "a" if i else "w") as f:
                f.write(out + "\n")
        post = smi(args.smi_index)
        new_tokens = last.get("new_tokens", 0)
        tps = new_tokens / (last.get("time_generate", 1e-6) or 1e-6)
        new_ctx = ids.shape[-1] - last.get("cached_tokens", 0)
        ptps = new_ctx / (last.get("time_prefill", 1e-6) or 1e-6)
        line = f"TURN {i+1}: prompt {new_ctx} tok at {ptps:.1f} t/s | generate {new_tokens} tok at {tps:.2f} t/s"
        if "accepted_draft_tokens" in last:
            a = last["accepted_draft_tokens"]
            d = a + last["rejected_draft_tokens"]
            # the prefill emits the first token; every step after it its accepted drafts plus one of its own
            line += f" | draft {a}/{d} ({100 * a / max(d, 1):.1f}%) | {(new_tokens - 1) / max(new_tokens - 1 - a, 1):.2f} tok/step"
            accept.append(a / max(d, 1))
        sh = generator.sparse_head
        if sh is not None:
            line += f" | sdh miss {100 * sh.miss_rate:.1f}% {'on' if sh.mode_sparse else 'off'} ({sh.mode_switches} sw)"
        line += f" | P100 {pre} -> {post}"
        print(line, flush = True)
        gen_rates.append(tps)
    print(f"SUMMARY {args.tag}: mean generate {sum(gen_rates)/len(gen_rates):.2f} t/s over {len(gen_rates)} turns"
          + (f", mean draft acceptance {100*sum(accept)/len(accept):.1f}%" if accept else ""), flush = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, add_draft_model_args = True, add_sampling_args = True)
    parser.add_argument("-mode", "--mode", type = str, default = "qwen35")
    parser.add_argument("-prompts", "--prompts", nargs = "+",
                        default = ["Implement a quicksort in go", "Implement quicksort in rust"])
    parser.add_argument("-max_new", "--max_new", type = int, default = 1000,
                        help = "Max new tokens per turn; chat.py runs to EOS, so compare its acceptance with a value that does too (a short cap measures only the code part of a reply, which drafts better)")
    parser.add_argument("-smi_index", "--smi_index", type = int, default = 0, help = "nvidia-smi index of the P100")
    parser.add_argument("-tag", "--tag", type = str, default = "run")
    parser.add_argument("-cool", "--cool", type = int, default = 0,
                        help = "Before each turn, wait until the P100 (-smi_index) is at or below this temperature (C); 0 = off")
    parser.add_argument("-save_text", "--save_text", type = str, default = None,
                        help = "Write the generated text of every turn to this file (compare builds/kernels byte for byte)")
    parser.add_argument("-sample", "--sample", action = "store_true",
                        help = "Use the sampler of chat.py (temperature, min-p, ...) instead of greedy; the draft acceptance then drops")
    main(parser.parse_args())
