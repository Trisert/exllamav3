"""Cold random-read study of the Flash-Next n-gram table file (row gather I/O ceiling).

Measures, on fresh random rows each time (the table is 19.8 GB; the page cache holds a tiny
share), the per-row cost of the paths the row table can use to stream 62-byte rows from this
USB-BOT SATA SSD (queue depth 1):
  - buffered 62-byte preads (the current path), 1 and N threads
  - buffered with POSIX_FADV_RANDOM (no readahead)
  - buffered 4 KiB reads (page granularity)
  - O_DIRECT 512-byte-aligned reads (sector granularity, bypasses the page cache)
  - O_DIRECT 4 KiB-aligned reads
  - ext.ngram_gather_cpu (the C++ pool) for the same row set
Also reports the block-layer work (reads, sectors) per option from /sys/block/<dev>/stat, so
per-command overhead and read amplification (readahead) are visible, and checks that the 128
table shards are contiguous in the file (RowTable.open merges them into one handle, so a lookup
issues one ngram_gather_cpu call, not one per shard).

Measured Nov 8 (Kingston SA400 on /dev/sdc, ASM1051E USB bridge, Bulk-Only transport, queue
depth 1, btrfs + LUKS, another build running on the CPU):

  case                                    ms      us/row    reads   sect/rd   B/row
  buffered 62B, 1 thread                1402.6     342.4     4493     10.29    5779
  buffered 62B, 8 threads               1125.8     274.9     4105      8.28    4250
  buffered 62B FADV_RANDOM, 1 thread    1213.3     296.2     4086      8.12    4150
  buffered 4K page, 1 thread            1244.0     303.7     4074      8.07    4112
  O_DIRECT 512B-aligned, 1 thread       1248.4     304.8     4074      8.18    4166
  O_DIRECT 4K-aligned, 1 thread         1246.5     304.3     4098      8.15    4176
  ext.ngram_gather_cpu pool             1145.9     279.8     4054      8.17    4139

i.e. ~300 us per cold random row whatever the read size, no gain from threads or O_DIRECT: the
Bulk-Only bridge serializes commands (device queue_depth 1, nr_requests 2) and the table rows are
hash-scattered (4054 reads for 4096 rows). Sequential read of the same file: 197 MiB/s. The only
lever for the gather is overlap with compute (or not reading the rows).

  .venv/bin/python util/bench_ngram_io.py [-n 8192] [-passes 3] [-device sdc]
"""
import argparse, json, mmap, os, random, struct, sys, time
from concurrent.futures import ThreadPoolExecutor

TABLE = "/home/nicola/exllamav3/models/Qwen3.8-Flash-Next-exl3-1.66bpw/ngram_embedding.safetensors"
TENSOR = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.trellis"
ROW_BYTES = 62
SECTOR = 512


def table_layout(path = TABLE, tensor = TENSOR):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.get("__metadata__", {})
    nrows = int(meta["rows"])
    t = h[tensor]
    base = 8 + n + t["data_offsets"][0]
    row_bytes = (t["data_offsets"][1] - t["data_offsets"][0]) // t["shape"][0]
    # shards are the tensors named <prefix>.shard_N.trellis; RowTable.open merges consecutive ones
    shards = sorted((v["data_offsets"][0], v["data_offsets"][1]) for k, v in h.items()
                    if ".shard_" in k and k.endswith(".trellis"))
    contiguous = all(a == b for a, b in zip([s[1] for s in shards[:-1]], [s[0] for s in shards[1:]]))
    return path, base, nrows, row_bytes, contiguous


def stat_reads(device):
    with open(f"/sys/block/{device}/stat") as f:
        v = [int(x) for x in f.read().split()]
    return v[0], v[2]     # reads completed, sectors read


def run_threaded(fn, n, threads):
    if threads == 1:
        for i in range(n):
            fn(i)
    else:
        chunks = [(i * n // threads, (i + 1) * n // threads) for i in range(threads)]
        with ThreadPoolExecutor(max_workers = threads) as ex:
            list(ex.map(lambda ab: [fn(i) for i in range(*ab)], chunks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type = int, default = 8192)
    ap.add_argument("-passes", type = int, default = 3, help = "fresh row sets per option")
    ap.add_argument("-device", type = str, default = "sdc")
    ap.add_argument("-seed", type = int, default = 7)
    args = ap.parse_args()

    path, base, nrows, row_bytes, contiguous = table_layout()
    assert row_bytes == ROW_BYTES
    print(f"table {path}: rows {nrows} row_bytes {row_bytes} base {base}")
    print(f"shards contiguous in the file: {contiguous} -> RowTable.open merges them into one handle, "
          f"one ngram_gather_cpu call per lookup")
    rnd = random.Random(args.seed)
    n = args.n

    fd = os.open(path, os.O_RDONLY)
    fd_ra = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd_ra, 0, 0, os.POSIX_FADV_RANDOM)

    # binary streams for buffered reads at page granularity
    fpage = os.open(path, os.O_RDONLY)

    # O_DIRECT: a 1 MiB page-aligned scratch buffer (mmap), 512-byte-aligned offsets required
    odir = os.open(path, os.O_RDONLY | os.O_DIRECT)
    scratch = mmap.mmap(-1, 1 << 20)
    mv = memoryview(scratch)

    def rows(pass_i):
        return sorted(rnd.sample(range(nrows), n))

    def pread62(uids, fd = fd):
        for u in uids:
            os.pread(fd, row_bytes, base + u * row_bytes)

    def pread62_mt(uids, threads = 8):
        def one(i):
            os.pread(fd, row_bytes, base + uids[i] * row_bytes)
        run_threaded(one, len(uids), threads)

    def pread4k(uids, fd = fd):
        for u in uids:
            off = base + u * row_bytes
            os.pread(fd, 4096, off - off % 4096)

    def odirect(uids, step = SECTOR):
        for u in uids:
            off = base + u * row_bytes
            a = off - off % step
            ln = ((off - a) + row_bytes + step - 1) // step * step
            got = os.preadv(odir, [mv[:ln]], a)
            assert got == ln, (got, ln)

    def odirect4k(uids):
        odirect(uids, 4096)

    cases = [
        ("buffered 62B, 1 thread", lambda u: pread62(u)),
        ("buffered 62B, 8 threads", lambda u: pread62_mt(u)),
        ("buffered 62B FADV_RANDOM, 1 thread", lambda u: pread62(u, fd_ra)),
        ("buffered 4K page, 1 thread", lambda u: pread4k(u)),
        ("O_DIRECT 512B-aligned, 1 thread", lambda u: odirect(u)),
        ("O_DIRECT 4K-aligned, 1 thread", lambda u: odirect4k(u)),
    ]
    print(f"{'case':<38} {'ms':>8} {'us/row':>8} {'reads':>8} {'sect/rd':>8} {'B/row':>7}")
    for name, fn in cases:
        tot = 0.0
        dr_tot = ds_tot = 0
        for p in range(args.passes):
            uids = rows(p)
            r0, s0 = stat_reads(args.device)
            t0 = time.perf_counter()
            fn(uids)
            dt = time.perf_counter() - t0
            r1, s1 = stat_reads(args.device)
            tot += dt
            dr_tot += r1 - r0
            ds_tot += s1 - s0
        ntot = n * args.passes
        dt = tot / args.passes
        print(f"{name:<38} {1000 * dt:8.1f} {1e6 * tot / ntot:8.1f} {dr_tot // args.passes:8d} "
              f"{ds_tot / max(dr_tot, 1):8.2f} {512 * ds_tot / ntot:7.0f}")

    # the C++ pool, on fresh rows
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    tot = 0.0
    for p in range(args.passes):
        uids = rows(p + 100)
        ut = torch.tensor(uids, dtype = torch.int64)
        out = torch.empty((len(uids), row_bytes // 2), dtype = torch.int16, pin_memory = True)
        r0, s0 = stat_reads(args.device)
        t0 = time.perf_counter()
        ext.ngram_gather_cpu(fd, base, row_bytes, ut, 0, out)
        dt = time.perf_counter() - t0
        r1, s1 = stat_reads(args.device)
        tot += dt
        print(f"ext.ngram_gather_cpu pool             {1000 * dt:8.1f} {1e6 * dt / n:8.1f} "
              f"{r1 - r0:8d} {(s1 - s0) / max(r1 - r0, 1):8.2f} {512 * (s1 - s0) / n:7.0f}")


if __name__ == "__main__":
    main()
