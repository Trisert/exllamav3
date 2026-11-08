import numpy as np, sys, collections, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
OUT = os.environ.get("COOP_TRACE_DIR", os.path.join(ROOT, "tmp", "coop_trace"))
def analyze(nm):
    rec = np.load(os.path.join(OUT, f"coop_trace_{nm}.npy")); rec = rec[np.argsort(rec[:,0], kind="stable")]
    t0,t1,meta,x = rec[:,0],rec[:,1],rec[:,2],rec[:,3]
    kind=(meta>>8)&255; blk=(meta>>16)&0xffff
    tw2 = (x & 0xffffffff)/1000.0; tw1 = (x >> 32)/1000.0
    run = np.maximum.accumulate(t1); cuts = np.nonzero(t0[1:] > run[:-1])[0] + 1
    calls = [c for c in np.split(np.arange(len(t0)), cuts) if len(c) > 300][1:-1]
    print(f"{nm}: {len(calls)} calls")
    S = collections.defaultdict(list)
    for c in calls:
        nblk = len(np.unique(blk[c])); wall = (t1[c].max()-t0[c].min())/1000
        S["wall"].append(wall)
        for kk,n in enumerate(["sharedA","routedA","sharedB","routedB"]):
            m = (kind[c]==kk)
            d = (t1[c][m]-t0[c][m])/1000
            real = tw2[c][m] > 0
            wait = np.where(real, tw1[c][m], 0)
            gemv = np.where(real, tw2[c][m]-tw1[c][m], 0)
            epi = np.where(real, d - tw2[c][m], 0)
            S[n+"_real"].append(real.sum())
            S[n+"_wait_us/blk"].append(wait.sum()/nblk); S[n+"_gemv_us/blk"].append(gemv.sum()/nblk); S[n+"_epi_us/blk"].append(epi.sum()/nblk)
            S[n+"_item_gemv_mean"].append(gemv[real].mean()); S[n+"_item_epi_mean"].append(epi[real].mean())
    for k,v in S.items(): print(f"  {k:24s} mean {np.mean(v):8.2f}  p10 {np.percentile(v,10):8.2f}  p90 {np.percentile(v,90):8.2f}")
for n in sys.argv[1:]: analyze(n)
