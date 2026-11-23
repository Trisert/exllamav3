"""
m = 1 multi-matrix GEMV (exl3_mgemv_kernel), taken by exl3_mgemm on Pascal for 7 / 8 bpw mul1 bundles (EXL3_MGEMV):
the uniform form (gate + up, K + V) and the sliced form (Q / K / V of different widths, one input transform per
source) against the per-matrix single GEMM. On other devices the same calls take the tile MGEMM and the checks still
hold (fp16 accumulation-order noise only).
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def _weights(k, n, bits, device):
    B = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype = torch.int16, device = device)
    suh = torch.randn(k, device = device).sign().half()
    svh = torch.randn(n, device = device).sign().half()
    return B, suh, svh


def _ptrs(ts, device):
    return torch.tensor([t.data_ptr() if isinstance(t, torch.Tensor) else t for t in ts], dtype = torch.long, device = device)


def _single(x, B, suh, svh, dtype):
    xh = torch.empty_like(x)
    c = torch.empty((x.shape[0], B.shape[1] * 16), dtype = dtype, device = x.device)
    ext.exl3_gemm(x, B, c, suh, xh, svh, 0, False, True, 0)
    return c


def _close(a, b, tol = None):
    # 3e-3 where the multi-matrix GEMV runs (Pascal); elsewhere the tile MGEMM, whose kernel shapes differ from the
    # single GEMM in accumulation order by up to ~1e-2 on mul1 tensors (see test_exl3_mgemm_sliced.py)
    if tol is None:
        tol = 3e-3 if _pascal(b.device) else 1e-2
    rel = ((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt()).item()
    assert rel < tol, f"relative rms {rel:.2e}"


def _pascal(device):
    return torch.cuda.get_device_capability(device)[0] == 6


@pytest.mark.parametrize("bits", [7, 8])
@pytest.mark.parametrize("shape", [(2048, 7680, 2), (2048, 512, 2), (2048, 2048, 3), (7680, 2048, 2)])
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("m", [1, 2, 3, 8, 16])
@torch.inference_mode()
def test_uniform_bundle(device, bits, shape, dtype, m):
    k, n, nm = shape
    torch.manual_seed(bits * 7 + n + nm + m)
    mats = [_weights(k, n, bits, device) for _ in range(nm)]
    x = (torch.randn(m, k, device = device) * 0.5).half()
    ref = [_single(x, B, suh, svh, dtype) for B, suh, svh in mats]
    a_had = torch.empty((nm, m, k), dtype = torch.half, device = device)
    out = torch.full((nm, m, n), 7.0, dtype = dtype, device = device)
    ext.exl3_mgemm(
        x.view(1, m, k), _ptrs([B for B, _, _ in mats], device), out, _ptrs([s for _, s, _ in mats], device), a_had,
        _ptrs([v for _, _, v in mats], device), None, None, bits, -1, False, True, -1, -1, 0, 1, None, None,
    )
    for o, r in zip(out, ref):
        _close(o, r)
    # K / V-sized pairs run the narrow plain config, the single GEMV's for each matrix: same per-group k split and
    # summation order
    if _pascal(device) and n == 512 and m == 1:
        for o, r in zip(out, ref):
            assert torch.equal(o, r)


@pytest.mark.parametrize("bits", [7, 8])
@pytest.mark.parametrize("widths", [(2048, 512, 512), (4096, 1024, 1024), (2048, 512, 512, 2048)])
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("m", [1, 2, 4, 5, 16])
@torch.inference_mode()
def test_sliced_bundle(device, bits, widths, dtype, m):
    torch.manual_seed(bits + sum(widths) + m)
    k = 2048
    mats = [_weights(k, n, bits, device) for n in widths]
    W = min(widths)
    x = (torch.randn(m, k, device = device) * 0.5).half()
    ref = [_single(x, B, suh, svh, dtype) for B, suh, svh in mats]
    es = 2 if dtype == torch.half else 4
    outs = [torch.full((m, n), 7.0, dtype = dtype, device = device) for n in widths]
    tr, sv, cp, src, stride = [], [], [], [], []
    for i, ((B, suh, svh), n) in enumerate(zip(mats, widths)):
        for n0 in range(0, n, W):
            tr.append(B.data_ptr() + (n0 // 16) * 16 * bits * 2)
            sv.append(svh.data_ptr() + n0 * 2)
            cp.append(outs[i].data_ptr() + n0 * es)
            src.append(i)
            stride.append(n)
    S = len(tr)
    i32 = lambda v: torch.tensor(v, dtype = torch.int32, device = device)
    a_had = torch.empty((len(mats), m, k), dtype = torch.half, device = device)
    carrier = torch.empty((S, 1, W), dtype = dtype, device = device).expand(S, m, W)
    ext.exl3_mgemm(
        x.view(1, m, k), _ptrs(tr, device), carrier, _ptrs([suh for _, suh, _ in mats], device), a_had,
        _ptrs(sv, device), None, None, bits, -1, False, True, -1, -1, 0,
        1, i32([W] * S), _ptrs(cp, device), i32(stride), i32(src), len(mats),
    )
    for o, r in zip(outs, ref):
        _close(o, r)


@pytest.mark.parametrize("bits", [8])
@torch.inference_mode()
def test_uniform_bundle_in_graph(device, bits):
    """Captured once, replayed with a new input in the same static buffer"""
    k, n, nm = 2048, 7680, 2
    torch.manual_seed(5)
    mats = [_weights(k, n, bits, device) for _ in range(nm)]
    x = (torch.randn(1, k, device = device) * 0.5).half()
    a_had = torch.empty((nm, 1, k), dtype = torch.half, device = device)
    out = torch.empty((nm, 1, n), dtype = torch.half, device = device)
    args = (_ptrs([B for B, _, _ in mats], device), out, _ptrs([s for _, s, _ in mats], device), a_had,
            _ptrs([v for _, _, v in mats], device), None, None, bits, -1, False, True, -1, -1, 0, 1, None, None)
    ext.exl3_mgemm(x.view(1, 1, k), *args)
    torch.cuda.synchronize(device)
    s = torch.cuda.Stream(device)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream = s):
            ext.exl3_mgemm(x.view(1, 1, k), *args)
    x.copy_((torch.randn(1, k, device = device) * 0.5).half())
    g.replay()
    torch.cuda.synchronize(device)
    ref = [_single(x, B, suh, svh, torch.half) for B, suh, svh in mats]
    for o, r in zip(out, ref):
        _close(o, r)


@pytest.mark.parametrize("m", [1, 4, 16])
@torch.inference_mode()
def test_pascal_takes_mgemv(device, m):
    """On Pascal the 8 bpw mul1 bundle runs as one exl3_mgemv_kernel launch"""
    if not _pascal(device):
        pytest.skip("Pascal only")
    k, n = 2048, 1024
    mats = [_weights(k, n, 8, device) for _ in range(2)]
    x = (torch.randn(1, m, k, device = device) * 0.5).half()
    a_had = torch.empty((2, m, k), dtype = torch.half, device = device)
    out = torch.empty((2, m, n), dtype = torch.half, device = device)
    with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CUDA]) as prof:
        ext.exl3_mgemm(x, _ptrs([B for B, _, _ in mats], device), out, _ptrs([s for _, s, _ in mats], device), a_had,
                       _ptrs([v for _, _, v in mats], device), None, None, 8, -1, False, True, -1, -1, 0, 1, None, None)
        torch.cuda.synchronize(device)
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert any("exl3_mgemv_kernel" in nm for nm in names), names
