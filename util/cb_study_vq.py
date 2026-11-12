"""Pair-trellis (V = 2) codebook study with a trained LUT, for a Pascal friendly decode (one LDS.32 -> half2 per two weights).

State: L bits. Each step shifts in B = V*K bits and emits LUT[idx(state)] (V values). idx = top Q bits or low Q bits of the
state. The LUT is trained with Lloyd iterations (Viterbi assignment + centroid update) on iid N(0,1) data and evaluated on
fresh data. Compare against mul1 (scalar trellis, L = 16) from cb_study.py: ~ 6.02 K - 0.3 .. 0.65 dB.

    python util/cb_study_vq.py K L Q top|low [iters] [train_seqs] [n]
"""
import torch, sys, math
torch.set_num_threads(4)


def run(K, L, Q, where, iters=12, seqs=48, n=512, V=2, seed=3, verbose=True):
    B = V * K
    assert B <= L and Q <= L
    R = 1 << B; H = 1 << (L - B)
    S = torch.arange(1 << L)
    idx = (S >> (L - Q)) if where == "top" else (S & ((1 << Q) - 1))
    gen = torch.Generator().manual_seed(seed)
    lut = torch.randn(1 << Q, V, generator=gen)

    def viterbi(w, lut, train):
        nb = w.shape[0]; T = w.shape[1] // V
        wv = w.view(nb, T, V)
        cost = torch.zeros(nb, 1 << L)
        args = []
        for t in range(T):
            m, a = cost.view(nb, R, H).min(1)           # (nb, H)
            e = ((wv[:, t, None, :] - lut[None]) ** 2).sum(-1)   # (nb, 2^Q)
            cost = m.repeat_interleave(R, dim=1) + e[:, idx]
            if train: args.append(a.to(torch.int32))
        best, s = cost.min(1)
        if not train:
            return best.sum().item() / (nb * T * V), None
        sums = torch.zeros(1 << Q, V); cnt = torch.zeros(1 << Q)
        ar = torch.arange(nb)
        for t in range(T - 1, -1, -1):
            ii = idx[s]
            sums.index_add_(0, ii, wv[:, t, :]); cnt.index_add_(0, ii, torch.ones(nb))
            j = args[t][ar, s >> B].long()
            s = (s >> B) | (j << (L - B))
        return best.sum().item() / (nb * T * V), (sums, cnt)

    for it in range(iters):
        w = torch.randn(seqs, n, generator=gen)
        mse, (sums, cnt) = viterbi(w, lut, True)
        nz = cnt > 0
        lut = lut.clone(); lut[nz] = sums[nz] / cnt[nz, None]
        # unused entries are re-seeded so they are not lost
        if (~nz).any(): lut[~nz] = torch.randn(int((~nz).sum()), V, generator=gen)
        # renormalise to unit variance over used entries weighted by usage (keeps the scale comparable)
        if verbose: print(f"  it {it:2d} train mse {10*math.log10(1/mse):.2f} dB used {int(nz.sum())}/{1<<Q}", flush=True)
    te = torch.Generator().manual_seed(999)
    mse, _ = viterbi(torch.randn(seqs, n, generator=te), lut, False)
    db = -10 * math.log10(mse)
    return db, 6.0206 * K - db, lut


if __name__ == "__main__":
    K = int(sys.argv[1]); L = int(sys.argv[2]); Q = int(sys.argv[3]); where = sys.argv[4]
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    seqs = int(sys.argv[6]) if len(sys.argv) > 6 else 48
    n = int(sys.argv[7]) if len(sys.argv) > 7 else 512
    db, gap, lut = run(K, L, Q, where, iters, seqs, n)
    print(f"K={K} L={L} Q={Q} {where}: test {db:.2f} dB  gap {gap:.2f}")
