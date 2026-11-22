"""Teacher-forced decode-mode perplexity: one token per forward (m = 1, the batch-1 decode kernels), cache + recurrent
state, over the first N tokens of README.md + doc/env_vars.md. Prints ppl and writes per-step top-1 ids for diffing.
usage: python decode_ppl.py <model_dir> <out.json> [N=1536]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator

model_dir, out_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
text = open(os.path.join(root, "README.md")).read() + "\n" + open(os.path.join(root, "doc", "env_vars.md")).read()

with torch.inference_mode():
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
    state = generator.new_recurrent_state().target
    ids = tokenizer.encode(text, add_bos = True)[:, :N + 1]
    nll, top1 = 0.0, []
    for pos in range(N):
        params = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 4096), "past_len": pos,
                  "recurrent_states": [state]}
        logits = model.forward(ids[:, pos:pos + 1], params)[0, -1].float()
        lp = torch.log_softmax(logits, dim = -1)
        nll -= lp[ids[0, pos + 1]].item()
        top1.append(int(logits.argmax().item()))
    ppl = math.exp(nll / N)
    print(f"DECODE_PPL {ppl:.5f} over {N} tokens ({torch.cuda.get_device_name(0)})")
    json.dump({"ppl": ppl, "n": N, "top1": top1}, open(out_path, "w"))
