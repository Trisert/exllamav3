"""Offline coverage curve for a hot-row cache of the Flash-Next n-gram table.

CPU only. Reads the three tiny hash tensors from the table file header (no random reads of the table), tokenizes
a training corpus and a held-out test corpus, computes the row ids with the reference hash
(NGramEmbedding.compute_ngram_ids), counts for every row in how many 4096-token training prompts it appears
(document frequency: the quantity that decides cold reads per prompt), then for cache sizes N reports the share of a
test prompt's unique rows that the N most frequent rows cover.

usage: python -I ngram_cov.py <model_dir> <train_root> <train_max_tokens> <test_root1> [<test_root2> ...]
"""
import sys, os, json, struct, glob, time
import numpy as np
from tokenizers import Tokenizer

model_dir, train_root, train_max = sys.argv[1], sys.argv[2], int(sys.argv[3])
test_roots = sys.argv[4:]
CH = 4096
ROW_BYTES = 62

# --- hash parameters (tiny tensors in the file header region) -------------------------------------------------
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
print("heads", len(sizes), "sizes", sizes[:3], "...", "mult", mult, "total rows", int(sizes.sum()), flush=True)
EOS = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]["eos_token_id"] \
    if "text_config" in json.load(open(os.path.join(model_dir, "config.json"))) \
    else json.load(open(os.path.join(model_dir, "config.json")))["eos_token_id"]
tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
H = len(sizes)
HPN = H // 2


def row_ids(t):
    """t: int64 token array of one segment (a file / prompt). Returns (len, 16) global row ids."""
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


def read_files(root, exts, max_bytes=800_000, limit_tokens=None):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, "**", "*." + e), recursive=True)
    paths = sorted(set(paths))
    return paths


def tokenize_paths(paths, limit):
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
        ids = np.array(tok.encode(txt).ids, dtype=np.int64)
        toks.append(ids)
        total += len(ids)
        if limit and total >= limit:
            break
    return toks, total


def chunks(toks):
    """Concatenate per-file token arrays into one stream and cut 4096-token prompts (n-grams do not span the cut:
    the prompt starts a fresh sequence, as a real request does)."""
    cat = np.concatenate(toks)
    return [cat[i:i + CH] for i in range(0, len(cat) - CH + 1, CH)]


t0 = time.time()
train_paths = read_files(train_root, ["py", "md", "rst", "txt", "js", "c", "h", "cpp", "cu", "json"])
rng = np.random.default_rng(0)
rng.shuffle(train_paths)
train_toks, ntrain = tokenize_paths(train_paths, train_max)
print(f"train: {len(train_toks)} files, {ntrain} tokens ({time.time()-t0:.0f}s)", flush=True)
train_chunks = chunks(train_toks)
print("train prompts", len(train_chunks), flush=True)

total_rows = int(sizes.sum())
df = np.zeros(total_rows, dtype=np.int32)
for i, c in enumerate(train_chunks):
    u = np.unique(row_ids(c).ravel())
    df[u] += 1
print(f"document-frequency table built ({time.time()-t0:.0f}s); touched rows {(df>0).sum()}", flush=True)

nz = np.sort(df[df > 0])[::-1]
print("distinct rows seen in training:", len(nz), "of", total_rows, flush=True)

SIZES_N = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000]
thr = {}
for N in SIZES_N:
    thr[N] = int(nz[min(N, len(nz)) - 1]) if N <= len(nz) else 1
for root in test_roots:
    paths = read_files(root, ["py", "md", "cu", "cuh", "cpp", "h", "txt", "json"])
    toks, nt = tokenize_paths(paths, None)
    cks = chunks(toks)
    print(f"\nTEST {root}: {len(toks)} files, {nt} tokens, {len(cks)} prompts of {CH}", flush=True)
    uniq_tot = 0
    hit = {N: 0 for N in SIZES_N}
    hit_pairs = {N: 0 for N in SIZES_N}
    per_head_uniq = np.zeros(H)
    per_head_hit = {N: np.zeros(H) for N in SIZES_N}
    for c in cks:
        ids = row_ids(c)
        for h in range(H):
            u = np.unique(ids[:, h])
            per_head_uniq[h] += len(u)
            for N in SIZES_N:
                per_head_hit[N][h] += int((df[u] >= thr[N]).sum())
    uniq_tot = per_head_uniq.sum()
    print(f"unique rows per prompt: {uniq_tot/len(cks):.0f}  (bigram heads {per_head_uniq[:HPN].sum()/len(cks):.0f}, "
          f"trigram heads {per_head_uniq[HPN:].sum()/len(cks):.0f})")
    print(f"{'rows':>11} {'size':>9} {'train thr(df>=)':>16} {'hit all':>8} {'hit 2-gram':>11} {'hit 3-gram':>11}")
    for N in SIZES_N:
        ha = per_head_hit[N].sum() / uniq_tot
        h2 = per_head_hit[N][:HPN].sum() / per_head_uniq[:HPN].sum()
        h3 = per_head_hit[N][HPN:].sum() / per_head_uniq[HPN:].sum()
        print(f"{N:>11} {N*ROW_BYTES/1e9:>7.2f}GB {thr[N]:>16} {ha*100:>7.1f}% {h2*100:>10.1f}% {h3*100:>10.1f}%", flush=True)
print(f"\ndone in {time.time()-t0:.0f}s")
