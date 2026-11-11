"""Build the dense hot-row pack of the Flash-Next n-gram table (CPU only, no GPU, no model load).

Rows are ranked by document frequency over 4096-token training prompts (as util/ngram_cov.py), the N most frequent are copied
byte for byte from the table with ONE sequential scan of the file, and saved as {uids (sorted int64), rows (int16 [N, 31])}.

usage: python -I ngram_pack.py <model_dir> <out.pt> <rows> <train_max_tokens> <train_root> [<train_root> ...]
"""
import glob, json, os, struct, sys, time
import numpy as np
import torch
from tokenizers import Tokenizer

model_dir, out_path, N, train_max = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
roots = sys.argv[5:]
CH = 4096
t0 = time.time()

f = os.path.join(model_dir, "ngram_embedding.safetensors")
with open(f, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
    base = 8 + n

    def get(name):
        k = [k for k in hdr if k.endswith("." + name)][0]
        a, b = hdr[k]["data_offsets"]
        fh.seek(base + a)
        return np.frombuffer(fh.read(b - a), dtype=np.int64).copy()
    offsets, sizes, mult = get("head_offsets"), get("head_vocab_sizes"), get("layer_multipliers")
shards = sorted((k for k in hdr if k.endswith(".trellis")), key=lambda k: int(k.split("shard_")[1].split(".")[0]))
S = len(shards)
R = hdr[shards[0]]["shape"][0]           # rows per shard
W = hdr[shards[0]]["shape"][1]           # int16 words per row
total_rows = int(sizes.sum())
print(f"heads {len(sizes)} shards {S} rows/shard {R} words/row {W} total rows {total_rows}", flush=True)
assert S * R >= total_rows

cfg = json.load(open(os.path.join(model_dir, "config.json")))
EOS = (cfg.get("text_config") or cfg)["eos_token_id"]
if isinstance(EOS, list):
    EOS = EOS[0]
tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
H = len(sizes)
HPN = H // 2


def row_ids(t):
    n_ = len(t)
    pad = np.concatenate([[EOS, EOS], t])
    s0, s1, s2 = pad[2:], pad[1:-1], pad[:-2]
    m2 = s0 * mult[0] ^ s1 * mult[1]
    m3 = m2 ^ s2 * mult[2]
    out = np.empty((n_, H), dtype=np.int64)
    for h in range(HPN):
        out[:, h] = m2 % sizes[h] + offsets[h]
        out[:, HPN + h] = m3 % sizes[HPN + h] + offsets[HPN + h]
    return out


exts = ["py", "md", "rst", "txt", "js", "c", "h", "cpp", "cu", "json", "rs", "go", "ts", "sh", "yml", "yaml", "toml"]
paths = []
for root in roots:
    for e in exts:
        paths += glob.glob(os.path.join(root, "**", "*." + e), recursive=True)
paths = sorted(set(paths))
np.random.default_rng(0).shuffle(paths)
toks, total = [], 0
for p in paths:
    try:
        if os.path.getsize(p) > 800_000:
            continue
        txt = open(p, errors="ignore").read()
    except Exception:
        continue
    if not txt.strip():
        continue
    toks.append(np.array(tok.encode(txt).ids, dtype=np.int64))
    total += len(toks[-1])
    if total >= train_max:
        break
print(f"train: {len(toks)} files, {total} tokens ({time.time() - t0:.0f}s)", flush=True)
cat = np.concatenate(toks)
df = np.zeros(total_rows, dtype=np.int32)
for i in range(0, len(cat) - CH + 1, CH):
    df[np.unique(row_ids(cat[i:i + CH]).ravel())] += 1
nz = int((df > 0).sum())
print(f"document-frequency table built ({time.time() - t0:.0f}s); rows seen {nz}", flush=True)

N = min(N, nz)
thr = int(np.partition(df[df > 0], -N)[-N])
sel = np.nonzero(df >= thr)[0]
if len(sel) > N:                         # ties at the threshold: keep the lowest ids
    sel = sel[:N]
print(f"selected {len(sel)} rows (df >= {thr}), {len(sel) * W * 2 / 1e9:.2f} GB", flush=True)
del df

rows = np.empty((len(sel), W), dtype=np.int16)
fd = os.open(f, os.O_RDONLY)
BLK = 1 << 20
got = 0
for s in range(S):
    lo = int(np.searchsorted(sel, s * R))
    hi = int(np.searchsorted(sel, (s + 1) * R))
    if hi == lo:
        continue
    local = sel[lo:hi] - s * R
    shard_off = base + hdr[shards[s]]["data_offsets"][0]
    for a in range(0, R, BLK):
        b = min(a + BLK, R)
        i0 = int(np.searchsorted(local, a))
        i1 = int(np.searchsorted(local, b))
        if i1 == i0:
            continue
        buf = os.pread(fd, (b - a) * W * 2, shard_off + a * W * 2)
        blk = np.frombuffer(buf, dtype=np.int16).reshape(-1, W)
        rows[lo + i0: lo + i1] = blk[local[i0:i1] - a]
        got += i1 - i0
        os.posix_fadvise(fd, shard_off + a * W * 2, (b - a) * W * 2, os.POSIX_FADV_DONTNEED)
    if s % 8 == 0:
        print(f"  shard {s}/{S}: {got}/{len(sel)} rows ({time.time() - t0:.0f}s)", flush=True)
assert got == len(sel)
torch.save({"uids": torch.from_numpy(sel.astype(np.int64)), "rows": torch.from_numpy(rows),
            "rows_per_shard": R, "words": W, "total_rows": total_rows, "train_tokens": total,
            "train_roots": roots, "threshold_df": thr}, out_path)
print(f"saved {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)
