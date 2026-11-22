"""Decode on a non-current GPU (Devin review, PR #4): LFM3 with an 8-bit cache on cuda:<dev> while cuda:0 stays current,
greedy 32 tokens through the generator (graphed decode, native split kernel on a P100), ids printed for comparison
with the same run on that GPU alone. usage: secondary_gpu.py <model_dir> <device index>"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3 import CacheLayer_quant
from exllamav3.generator.sampler import ArgmaxSampler

model_dir, dev = sys.argv[1], int(sys.argv[2])
torch.cuda.set_device(0)
config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
model.load(device = f"cuda:{dev}")
tokenizer = Tokenizer.from_config(config)
generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
ids = tokenizer.encode("The history of the printing press", add_bos = True)
job = Job(input_ids = ids, max_new_tokens = 32, stop_conditions = [], sampler = ArgmaxSampler())
generator.enqueue(job)
toks = []
while generator.num_remaining_jobs():
    for r in generator.iterate():
        if r.get("stage") == "streaming" and "token_ids" in r:
            toks += r["token_ids"].flatten().tolist()
print("current device", torch.cuda.current_device(), "model on", dev, torch.cuda.get_device_name(dev))
print("IDS", toks)
