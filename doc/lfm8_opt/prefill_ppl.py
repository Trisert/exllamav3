"""Prefill-mode perplexity: one forward over the first N tokens of README.md + doc/env_vars.md (logits for every
position, 16k 8-bit KV cache), so the sliding-window layers attend through the prefill path. Prints ppl and writes the
per-position top-1 ids for diffing. usage: python prefill_ppl.py <model_dir> <out.json> [N=4096]"""
import sys, os, json, math
src = os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, src)
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.cache import CacheLayer_quant

model_dir, out_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
text = open(os.path.join(src, "README.md")).read() + "\n" + open(os.path.join(src, "doc", "env_vars.md")).read()
with torch.inference_mode():
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 16384, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    model.load()
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok)
    ids = tok.encode(text, add_bos = True)[:, :N + 1]
    state = gen.new_recurrent_state().target
    logits = model.forward(ids[:, :N], {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 16384),
                                        "past_len": 0, "recurrent_states": [state]})[0].float()
    lp = torch.log_softmax(logits, dim = -1)
    nll = -lp.gather(-1, ids[0, 1:N + 1].to(lp.device)[:, None]).sum().item()
    ppl = math.exp(nll / N)
    print(f"PREFILL_PPL {ppl:.5f} over {N} tokens ({torch.cuda.get_device_name(0)})")
    json.dump({"ppl": ppl, "n": N, "top1": logits.argmax(-1).tolist()}, open(out_path, "w"))
