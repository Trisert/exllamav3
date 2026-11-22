"""Needle-in-haystack gate: a passphrase inserted at 10 / 50 / 90 % of a haystack of ~50 % and ~85 % of the cache
(16384 tokens, 8-bit quantized KV as in the bench; the rest leaves room for 1536 tokens of reasoning + answer), chat
template, greedy. Pass = exact passphrase in the answer after the reasoning block.
usage: python needle.py <model_dir>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ArgmaxSampler

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
filler = ""
for f in ("README.md", "doc/env_vars.md", "doc/pascal_progress.md", "doc/pascal_p100.md"):
    filler += open(os.path.join(root, f), errors = "ignore").read() + "\n"
CACHE = 16384
NEEDLE = "The secret passphrase for the vault is HARBOR-7421-VIOLET."

with torch.inference_mode():
    config = Config.from_directory(sys.argv[1])
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = CACHE, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    model.load()
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok)
    fids = tok.encode(filler)[0]
    ok = 0; total = 0
    for frac_ctx in (0.5, 0.85):
        n = int(CACHE * frac_ctx) - 200
        hay_ids = fids[:n]
        for depth in (0.1, 0.5, 0.9):
            cut = int(n * depth)
            hay = tok.decode(hay_ids[:cut]) + "\n" + NEEDLE + "\n" + tok.decode(hay_ids[cut:])
            msgs = [{"role": "user", "content": "Read the following document.\n\n" + hay +
                     "\n\nWhat is the secret passphrase for the vault? Answer with the passphrase only."}]
            ids = tok.hf_chat_template(msgs, add_generation_prompt = True)
            job = Job(input_ids = ids, max_new_tokens = 1536, sampler = ArgmaxSampler(),
                      stop_conditions = [tok.eos_token_id])
            gen.enqueue(job)
            out = ""
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    out += r.get("text", "")
            # Reasoning model: judge the answer after the reasoning block when there is one
            answer = out.split("</think>")[-1]
            hit = "HARBOR-7421-VIOLET" in answer
            ok += hit; total += 1
            print(f"ctx {ids.shape[-1]:6d} tok, needle at {depth:.0%}: {'PASS' if hit else 'FAIL'}  "
                  f"{answer.strip()[-70:]!r} ({len(out)} chars)", flush = True)
    print(f"NEEDLE {ok}/{total} ({torch.cuda.get_device_name(0)})")
