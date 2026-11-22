"""Token-exact check of chat.py's lfm3 prompt mode against the model's chat_template.jinja"""
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "examples"))
import jinja2
from exllamav3 import Config, Tokenizer
from chat_templates import prompt_formats

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw"
tokenizer = Tokenizer.from_config(Config.from_directory(model_dir))
src = open(os.path.join(model_dir, "chat_template.jinja")).read()
env = jinja2.Environment(extensions = ["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))

# jinja2 has no {% generation %} block (a transformers extension); it only marks the span, so drop the tags
src = src.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")
template = env.from_string(src)

fmt = prompt_formats["lfm3"]("User", "Assistant")
system = fmt.default_system_prompt(False)
cases = [
    [("Implement a quicksort in go", None)],
    [("Hi", "<think>The user greets.</think>Hello! How can I help?"), ("Implement a quicksort in go", None)],
    [("Hi", "Hello!"), ("And in Rust?", None)],
]
fails = 0
for think in (True, False):
    for msgs in cases:
        ours = fmt.format(system, msgs, think)
        if think:
            ours += fmt.thinktag()[0]
        hf_msgs = [{"role": "system", "content": system}]
        for u, a in msgs:
            hf_msgs.append({"role": "user", "content": u})
            if a is not None:
                hf_msgs.append({"role": "assistant", "content": a})
        ref = template.render(messages = hf_msgs, add_generation_prompt = True, enable_thinking = think,
                              bos_token = tokenizer.bos_token)
        ids_ours = tokenizer.encode(ours, add_bos = fmt.add_bos(), encode_special_tokens = True)[0].tolist()
        ids_ref = tokenizer.encode(ref, add_bos = False, encode_special_tokens = True)[0].tolist()
        ok = ids_ours == ids_ref
        fails += not ok
        print(f"think={think} turns={len(msgs)}: {'OK' if ok else 'MISMATCH'} ({len(ids_ref)} tokens)")
        if not ok:
            print("  ours:", repr(tokenizer.decode(ids_ours, decode_special_tokens = True)))
            print("  ref: ", repr(ref))
sys.exit(1 if fails else 0)
