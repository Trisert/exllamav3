# LFM3-1.7B-exl3 8.0bpw (head 6 bpw) — decode ceilings (batch 1)

Model: Trisert/LFM3-1.7B-exl3 @ 8.0bpw (e21a9080), 24 layers (13 short-conv, 6 full + 5 sliding attention, GQA 16/4,
hidden 2048, FFN 7680, vocab 128000).

Bytes read per decoded token (from the safetensors header):
- MLP 1133.9 MB, conv 218.8 MB, attention 116.4 MB, lm_head (6 bpw) 196.9 MB -> **1.666 GB**
- embedding 524 MB: one row per token, not counted
- KV at depth 512 negligible; at 8k only 11 attention layers (5 of them sliding 256) -> < 0.1 GB

Measured bandwidth (device copy, read+write / time):
- RTX 2060: 290 GB/s -> floor 5.74 ms/token -> **~174 t/s**
- Tesla P100: 538 GB/s -> floor 3.10 ms/token -> **~323 t/s**

Practical ceiling: kernel efficiency ~85% of copy BW plus ~0.5 ms of glue (norms, attention, conv, sampling)
- 2060: ~7.3 ms -> ~137 t/s practical
- P100: ~4.1 ms -> ~245 t/s practical (P100 fixed costs per launch are higher; ~200 t/s is a realistic target)

Baseline (Nov 22, flash-lead1 .so): 2060 117.7 t/s (8.49 ms, 68% of the floor); P100 51.7 t/s (19.36 ms, 16% of the floor).
