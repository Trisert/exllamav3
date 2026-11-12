"""Offline codebook study for the exl3 trellis (L = 16 bit state, K bits/weight).

Any codebook is a table of 2^16 values (state -> weight), so the Viterbi MSE on an iid N(0,1) source can be measured
for any candidate without touching a GPU. Reports MSE in dB and the gap to the rate-distortion bound (6.02 K dB).

    python util/cb_study.py 2,3,4 [name-filter] [seq-len] [num-seqs]
"""
import torch, sys, math
torch.set_num_threads(4)
L = 16
S = torch.arange(1 << L, dtype=torch.int64)
M32 = 0xFFFFFFFF


def bytesum(x):
    x = x & M32
    return ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)).float()


def norm(v):
    v = v.float(); return (v - v.mean()) / v.std()


def popc(x):
    c = torch.zeros_like(x)
    for i in range(32): c += (x >> i) & 1
    return c.float()


g = torch.Generator().manual_seed(1)
cands = {}
cands["mul1 (current)"] = lambda: norm(bytesum(S * 0x83DCD12D))
cands["iid gaussian (ceiling)"] = lambda: norm(torch.randn(1 << L, generator=g))
cands["mul16x2: u16 mults A,B, 4 bytes"] = lambda: norm(bytesum((((S * 0xD12D) & 0xFFFF)) | (((S * 0x9E37) & 0xFFFF) << 16)))
cands["mul16 only (2 bytes)"] = lambda: norm((((S * 0xD12D) & 0xFFFF) & 255).float() + ((((S * 0xD12D) & 0xFFFF) >> 8)).float())
cands["xorshift bytesum"] = lambda: norm(bytesum(S ^ (S << 7) ^ (S << 13) ^ (S << 19)))
cands["rot-xor bytesum"] = lambda: norm(bytesum((S | (S << 16)) ^ ((S << 5) | (S << 21)) ^ (S << 11)))
cands["popcount(S*C)"] = lambda: norm(popc((S * 0x83DCD12D) & M32))


def two_lut():
    t0 = torch.randn(256, generator=g); t1 = torch.randn(256, generator=g)
    return norm(t0[S & 255] + t1[S >> 8])
cands["two-LUT 256+256 additive"] = two_lut


def two_lut_mix():
    t0 = torch.randn(256, generator=g); t1 = torch.randn(256, generator=g)
    h = ((S * 0xD12D) & 0xFFFF)
    return norm(t0[S & 255] + t1[h >> 8])
cands["LUT[lo byte]+LUT[hash16 hi byte]"] = two_lut_mix


def four_lut():
    t = [torch.randn(16, generator=g) for _ in range(4)]
    return norm(sum(t[i][(S >> (4 * i)) & 15] for i in range(4)))
cands["four-LUT 4x16 additive"] = four_lut


def lut3():
    t = [torch.randn(64, generator=g) for _ in range(3)]
    return norm(t[0][S & 63] + t[1][(S >> 5) & 63] + t[2][(S >> 10) & 63])
cands["three-LUT 3x64 overlapping"] = lut3


def viterbi(cb, K, n=1024, seqs=24, seed=7):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(seqs, n, generator=gen)
    R = 1 << K; H = 1 << (L - K)
    tot = 0.0
    for q in range(seqs):
        cost = torch.zeros(1 << L)
        for t in range(n):
            m = cost.view(R, H).min(0).values          # best over predecessors, indexed by (s' >> K)
            cost = m.repeat_interleave(R) + (w[q, t] - cb) ** 2
        tot += cost.min().item()
    return tot / (seqs * n)


if __name__ == "__main__":
    Ks = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 3, 4]
    only = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    seqs = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    print(f"{'codebook':36s} " + " ".join(f"K={k}: dB gap  " for k in Ks), flush=True)
    for name, f in cands.items():
        if only and only not in name: continue
        cb = f()
        row = []
        for K in Ks:
            mse = viterbi(cb, K, n, seqs)
            db = -10 * math.log10(mse); gap = 6.0206 * K - db
            row.append(f"{db:6.2f} {gap:5.2f}")
        print(f"{name:36s} " + "   ".join(row), flush=True)
