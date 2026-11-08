import sys, os, time, json, collections
sys.path.insert(0, "/home/nicola/exllamav3")
import torch, argparse
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler import ComboSampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache=True, add_draft_model_args=True, add_sampling_args=True)
    args = parser.parse_args(["-m", "/home/nicola/exllamav3/models/Ornith-1.5-35B-A3B-EXL3-4bpw", "-gs", "4.0,15.0",
                              "-cs", "32768", "-cq", "4", "-mtp", "-ndt", "1", "-sdh", "16384",
                              "-temp", "0.7", "-topp", "0.8", "-topk", "20", "-minp", "0", "-presp", "1.5"])
    torch.set_grad_enabled(False); _im = torch.inference_mode(); _im.__enter__()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=args.num_draft_tokens, dynamic_draft_tokens=args.dynamic_draft,
                    draft_confidence=args.draft_confidence, sparse_draft_head=args.sparse_draft_head)
    prompt = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\nImplement a quicksort in go<|im_end|>\n<|im_start|>assistant\n"
    def run(n):
        job = Job(input_ids=tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True), max_new_tokens=n,
                  sampler=model_init.get_arg_sampler(args), stop_conditions=[], seed=1)
        gen.enqueue(job)
        its = []
        while gen.num_remaining_jobs():
            t = time.time()
            res = gen.iterate()
            nt = sum(int(r["token_ids"].numel()) for r in res if r.get("token_ids") is not None)
            its.append((time.time() - t, nt))
        return its
    run(300)
    its = run(200)[8:]
    tt = sum(t for t, _ in its); tok = sum(n for _, n in its)
    print(f"MTP iterate: {1000*tt/len(its):.1f} ms/iter, {tok/len(its):.2f} tok/iter, {tok/tt:.1f} t/s  (iters {len(its)})", flush=True)

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        its = run(100)
    n = len(its)
    print(f"PROFILED {n} iters")
    out = os.environ["CLAUDE_JOB_DIR"] + "/tmp/ornith_trace.json"
    prof.export_chrome_trace(out)
    tr = json.load(open(out))
    for dev, nm in ((1, "P100"), (0, "RTX2060")):
        ev = [e for e in tr["traceEvents"] if e.get("cat") == "kernel" and e["args"].get("device") == dev]
        busy = sum(e["dur"] for e in ev)
        print(f"{nm} GPU busy {busy/n/1000:.2f} ms/iter, {len(ev)/n:.0f} kernels/iter")
        agg = collections.defaultdict(lambda: [0, 0])
        for e in ev:
            agg[e["name"][:80]][0] += e["dur"]; agg[e["name"][:80]][1] += 1
        for k, v in sorted(agg.items(), key=lambda x: -x[1][0])[:22]:
            print(f"    {v[0]/n/1000:6.2f} ms/iter {v[1]/n:6.1f}x  {k}")
    for e in sorted(prof.key_averages(), key=lambda x: -x.self_cpu_time_total)[:8]:
        print(f"  CPU self {e.self_cpu_time_total/n/1000:6.2f} ms/iter {e.count/n:6.1f}x  {e.key[:60]}")
