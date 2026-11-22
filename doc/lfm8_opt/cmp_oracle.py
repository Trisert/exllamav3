"""usage: python cmp_oracle.py ref.json cand.json -> first divergent token, final-step top-20 overlap and max |dlogit|"""
import json, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
x, y = a["ids"], b["ids"]
div = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
common = set(a["top20_ids"]) & set(b["top20_ids"])
va = dict(zip(a["top20_ids"], a["top20_vals"]))
vb = dict(zip(b["top20_ids"], b["top20_vals"]))
dmax = max(abs(va[i] - vb[i]) for i in common) if common else float("nan")
print(f"ids match {div is None} (first divergence {div} of {len(x)}), top20 overlap {len(common)}/20, "
      f"max |dlogit| on overlap {dmax:.4f}")
