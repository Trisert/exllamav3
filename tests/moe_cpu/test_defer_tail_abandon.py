"""
EXL3_MOE_DEFER_TAIL hands the CPU tail's collect to the caller (submit_prefill(defer_tail = True) returns a pending, the
combine calls collect_deferred_tail). If the caller never gets there (an exception between issue and combine), the
tail's slot consumed flags stay unpublished and the next job on those slots waits for them forever. The host remembers
the one open tail and reads it back (dropping the result) before accepting the next submit.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.nogpu


@pytest.fixture
def host(monkeypatch):
    from exllamav3.model import moe_cpu_host as mch

    class CpuTorch:
        cuda = SimpleNamespace(device = lambda device: nullcontext())

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(mch, "torch", CpuTorch())
    h = mch.MoeCpuHost.__new__(mch.MoeCpuHost)
    h._deferred_open = None
    h.collected = []
    h._collect_compute = lambda jobs, out, rtmp, hh: h.collected.append(list(jobs))
    return h


def _pending(jobs, out):
    out_t = torch.zeros((2, 4))
    return (jobs, out_t, None, torch.tensor([0, 1]), 4, out, "cpu")


def test_collect_clears_the_open_tail(host):
    out = torch.zeros((2, 4))
    pending = _pending(["j0"], out)
    host._deferred_open = pending
    host.collect_deferred_tail(pending)
    assert host._deferred_open is None
    assert host.collected == [["j0"]]
    host._drain_abandoned_tail()              # nothing left to drain
    assert host.collected == [["j0"]]


def test_abandoned_tail_is_read_back_once_and_dropped(host):
    out = torch.zeros((2, 4))
    pending = _pending(["j0", "j1"], out)
    host._deferred_open = pending
    host._drain_abandoned_tail()
    assert host.collected == [["j0", "j1"]], "the flags must be released by reading the jobs back"
    assert host._deferred_open is None
    assert torch.count_nonzero(out) == 0, "an abandoned tail must not be merged into the output"
    host._drain_abandoned_tail()
    assert len(host.collected) == 1


@pytest.mark.parametrize("entry", ["submit", "submit_issue", "submit_prefill", "submit_issue_fused"])
def test_every_submit_drains_first(host, entry):
    out = torch.zeros((2, 4))
    host._deferred_open = _pending(["j0"], out)

    # Past the drain the entry points need a full host; the missing specs make them raise, which is enough to show
    # that the drain ran first
    host.specs = None
    y = torch.zeros((2, 4))
    args = {"submit": (0, y, y, y), "submit_issue": (0, y, y, y), "submit_prefill": (0, y, y, y),
            "submit_issue_fused": (0, y, y, y, None, None, 0)}[entry]
    with pytest.raises(Exception):
        getattr(host, entry)(*args)
    assert host.collected == [["j0"]]
    assert host._deferred_open is None
