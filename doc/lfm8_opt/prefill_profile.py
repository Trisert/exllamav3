"""Kernel profile of one prefill forward (default 4096 tokens, 16k 8-bit KV cache): top kernels by CUDA time.
usage: python prefill_profile.py <model_dir> [tokens=4096] [past=0]"""
import sys, os
sys.path.insert(0, os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.cache import CacheLayer_quant

model_dir = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
past = int(sys.argv[3]) if len(sys.argv) > 3 else 0
with torch.inference_mode():
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 16384 + 256, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    model.load()
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok)
    ids = torch.randint(1000, 60000, (1, past + T), generator = torch.Generator().manual_seed(0))

    def run():
        state = gen.new_recurrent_state().target
        if past:
            model.forward(ids[:, :past], {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 16384 + 256),
                                          "past_len": 0, "recurrent_states": [state], "last_tokens_only": 1})
        params = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 16384 + 256),
                  "past_len": past, "recurrent_states": [state], "last_tokens_only": 1}
        if os.environ.get("PF_PREFILL"):   # model.prefill, as the perf script and the generator's prefill use
            del params["last_tokens_only"]
            return model.prefill(ids[:, past:], params)
        return model.forward(ids[:, past:], params)
    run(); run(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    e0.record(); run(); e1.record(); torch.cuda.synchronize()
    print(f"PREFILL wall {e0.elapsed_time(e1):.1f} ms for past {past} + {T} tokens ({torch.cuda.get_device_name(0)})")
    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        run(); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit = 22, max_name_column_width = 90))
