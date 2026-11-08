"""Micro-test of the pinned-memory classification and DMA path used by the CPU-MoE handoff.

Checks, for a cudaHostRegister'ed POSIX-shm segment (same construction as MoeCpuHost):
  1. does torch classify the frombuffer view as pinned?
  2. CPU wall time of copy_(non_blocking=True) H2D (blocking = copies through a staging buffer)
  3. CPU wall time of ext.cuda_memcpy_async H2D, versus the DMA duration measured by events
  4. D2H in both forms, and data equality
Run from the worktree root: .venv/bin/python tmp/micro_copy.py
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from multiprocessing import shared_memory

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.model_tp_cuda import CUDA_HOST_REGISTER_PORTABLE, CUDA_HOST_REGISTER_MAPPED

SIZE = 32 << 20

def bench(dev, label):
    shm = shared_memory.SharedMemory(create = True, size = SIZE)
    base = np.frombuffer(shm.buf, dtype = np.uint8).ctypes.data
    ext.cuda_host_register(base, SIZE, CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
    host = torch.frombuffer(shm.buf, dtype = torch.int16, count = SIZE // 2, offset = 0)
    src = torch.arange(SIZE // 2, dtype = torch.int16)
    host.copy_(src[: host.numel()])
    d = torch.empty(SIZE // 2, dtype = torch.int16, device = dev)
    stream = torch.cuda.Stream(device = dev)
    nbytes = SIZE
    print(f"--- {label} ({torch.cuda.get_device_name(dev)}) is_pinned={host.is_pinned()} "
          f"nbytes={nbytes >> 20} MiB")

    # 1. torch copy_ H2D: CPU wall time of the call itself
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    d.copy_(host, non_blocking = True)
    t_cpu = time.perf_counter() - t0
    torch.cuda.synchronize(dev)
    ok1 = bool(torch.equal(d.cpu(), host))

    # 2. raw cudaMemcpyAsync H2D on a side stream, timed with events
    with torch.cuda.stream(stream):
        ev0 = torch.cuda.Event(enable_timing = True); ev1 = torch.cuda.Event(enable_timing = True)
        ev0.record(stream)
        t0 = time.perf_counter()
        ext.cuda_memcpy_async(d.data_ptr(), host.data_ptr(), nbytes, 1)
        t_cpu_async = time.perf_counter() - t0
        ev1.record(stream)
        ev1.synchronize()
        dma_ms = ev0.elapsed_time(ev1)
    ok2 = bool(torch.equal(d.cpu(), host))

    # 3. D2H: torch copy_ then raw async back into a second host buffer
    h2 = torch.frombuffer(shm.buf, dtype = torch.int16, count = SIZE // 2, offset = 0).clone()
    h2.zero_()
    t0 = time.perf_counter()
    h2.copy_(d, non_blocking = True)
    t_cpu_d2h = time.perf_counter() - t0
    torch.cuda.synchronize(dev)
    ok3 = bool(torch.equal(h2, host))
    h3 = torch.frombuffer(shm.buf, dtype = torch.int16, count = SIZE // 2, offset = 0)
    h3.zero_()
    t0 = time.perf_counter()
    ext.cuda_memcpy_async(h3.data_ptr(), d.data_ptr(), nbytes, 2)
    t_cpu_async_d2h = time.perf_counter() - t0
    torch.cuda.synchronize(dev)
    ok4 = bool(torch.equal(h3, host))

    print(f"    copy_       H2D: CPU {1000 * t_cpu:7.2f} ms")
    print(f"    memcpyAsync H2D: CPU {1000 * t_cpu_async:7.2f} ms, DMA {dma_ms:7.2f} ms "
          f"({nbytes / (dma_ms * 1e-3) / 1e9:.2f} GB/s)")
    print(f"    copy_       D2H: CPU {1000 * t_cpu_d2h:7.2f} ms")
    print(f"    memcpyAsync D2H: CPU {1000 * t_cpu_async_d2h:7.2f} ms")
    print(f"    data ok: {ok1} {ok2} {ok3} {ok4}")
    del d, h2, h3
    ext.cuda_host_unregister(base)
    shm.close(); shm.unlink()

if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        bench(i, f"cuda:{i}")
