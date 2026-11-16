"""Progressive first prefill chunks (EXL3_NGRAM_RAMP) and the next-chunk prefetch prediction
(Qwen3.8-Flash-Next PLE n-gram table streamed from disk).

Pure Python, no model: the Job helpers are arithmetic over the generator's chunk size, the prompt
length and the recurrent last-page rule. Every case simulates the prefill loop's own bound
computation and asserts that (a) the chunks tile the prompt exactly, page-aligned, doubling from
one page up to max_chunk_size, and (b) the end position predicted for the chunk after the current
one (params["ple_prefetch_next"]) is exactly the chunk the loop then runs. (b) is the regression
test for the broken look-ahead the Nov 7 doc reports: the old prediction ignored the recurrent
last-page split, so the staged entry was never the one the next forward asked for.
"""

from types import SimpleNamespace

import pytest

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator import job as jobmod

MAX_CHUNK = 2048


def _stub(embeddings = (), chunk_size = None, max_chunk_size = MAX_CHUNK,
          loaded_tp = False, prefetch_layers = True, recurrent_cache = True):
    model = SimpleNamespace(
        loaded_tp = loaded_tp,
        _get_prefetch_layers = [object()] if prefetch_layers else [],
    )
    gen = SimpleNamespace(
        max_chunk_size = max_chunk_size,
        recurrent_cache = object() if recurrent_cache else None,
        model = model,
    )
    j = SimpleNamespace(generator = gen, embeddings = list(embeddings), _ramp_next = None)
    # the Job helpers are ordinary methods; bind the ones the others call through self
    j._ramp_gate = lambda cs = chunk_size: jobmod.Job._ramp_gate(j, cs)
    j._ramp_peek = lambda cs = chunk_size: jobmod.Job._ramp_peek(j, cs)
    j._ramp_peek_next = lambda cs = chunk_size: jobmod.Job._ramp_peek_next(j, cs)
    return j


def _loop_end(j, start, prompt_len):
    """The prefill loop's own bound computation for one chunk (Job.prefill)."""
    end = (start + j.generator.max_chunk_size) // PAGE_SIZE * PAGE_SIZE
    end = min(end, prompt_len - 1)
    end = min(end, (start + j._ramp_peek()) // PAGE_SIZE * PAGE_SIZE)
    if j.generator.recurrent_cache is not None:
        last_page_b = (prompt_len - 1) // PAGE_SIZE * PAGE_SIZE
        if start < last_page_b <= end:
            end = last_page_b
    return end


def _plan(prompt_len, start0 = 0, **stub_kwargs):
    """Chunks the loop runs: (start, end, predicted end of the following chunk)."""
    j = _stub(**stub_kwargs)
    seq = SimpleNamespace(sequence_ids = list(range(prompt_len)))
    chunks = []
    start = start0
    while start < prompt_len - 1:
        end = _loop_end(j, start, prompt_len)
        pred = jobmod.Job._next_chunk_end(j, seq, end, None)
        chunks.append((start, end, pred))
        if j._ramp_gate():
            j._ramp_next = j._ramp_peek_next()
        start = end
    return chunks


def _sizes(chunks):
    return [e - s for s, e, _ in chunks]


@pytest.mark.parametrize("prompt_len, expected", [
    (4096, [PAGE_SIZE, 512, 1024, 2048, 255]),
    (8192, [PAGE_SIZE, 512, 1024, 2048, 2048, 2048, 255]),
    (4097, [PAGE_SIZE, 512, 1024, 2048, 256]),
    (3000, [PAGE_SIZE, 512, 1024, 1024, 183]),
    (512, [PAGE_SIZE, 255]),
    (257, [PAGE_SIZE]),
])
def test_ramp_schedule(monkeypatch, prompt_len, expected):
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", True)
    monkeypatch.setattr(jobmod, "NGRAM_RAMP_FIRST", PAGE_SIZE)
    chunks = _plan(prompt_len)
    assert _sizes(chunks) == expected
    assert sum(_sizes(chunks)) == prompt_len - 1
    for start, end, _ in chunks:
        assert start % PAGE_SIZE == 0
        assert end % PAGE_SIZE == 0 or end == prompt_len - 1


def test_ramp_prediction_matches_next_chunk(monkeypatch):
    # the look-ahead must name exactly the following chunk's ids, or lookup() can never take the
    # staged set (the failure the Nov 7 doc describes)
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", True)
    monkeypatch.setattr(jobmod, "NGRAM_RAMP_FIRST", PAGE_SIZE)
    for prompt_len in (257, 512, 3000, 4095, 4096, 4097, 8191, 8192, 20000):
        chunks = _plan(prompt_len)
        for (start, end, pred), (next_start, next_end, _) in zip(chunks, chunks[1:]):
            assert next_start == end
            assert pred == next_end, f"len {prompt_len}: chunk {start}..{end} predicts {pred}, next chunk ends {next_end}"
        assert chunks[-1][2] == prompt_len - 1


def test_ramp_off_keeps_old_chunking_and_predicts(monkeypatch):
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", False)
    chunks = _plan(4096)
    assert _sizes(chunks) == [2048, 1792, 255]
    for (start, end, pred), (next_start, next_end, _) in zip(chunks, chunks[1:]):
        assert pred == next_end
    chunks = _plan(8192)
    assert _sizes(chunks) == [2048, 2048, 2048, 1792, 255]
    for (start, end, pred), (next_start, next_end, _) in zip(chunks, chunks[1:]):
        assert pred == next_end


def test_last_page_split_is_predicted(monkeypatch):
    # a chunk that would straddle the last full page is cut there, so the final partial page is
    # its own forward; the look-ahead has to know that
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", False)
    chunks = _plan(4096)          # last full page at 3840, partial page 3840..4095
    assert _sizes(chunks) == [2048, 1792, 255]
    assert chunks[0][2] == 3840   # end of the following chunk, predicted during chunk 0
    assert chunks[1][1] == 3840
    assert chunks[2][1] == 4095
    chunks = _plan(8192)          # 2048 x 3, then the cut, then the partial page
    assert _sizes(chunks) == [2048, 2048, 2048, 1792, 255]
    assert chunks[2][2] == 7936
    assert chunks[3][1] == 7936
    assert chunks[4][1] == 8191


def test_cached_prefix_start(monkeypatch):
    # a job resuming a cached prefix starts mid-prompt; the ramp and the look-ahead still tile
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", True)
    monkeypatch.setattr(jobmod, "NGRAM_RAMP_FIRST", PAGE_SIZE)
    for prompt_len, start0, expected in (
        (4096, 2048, [PAGE_SIZE, 512, 1024, 255]),
        (8192, 4096, [PAGE_SIZE, 512, 1024, 2048, 255]),
    ):
        chunks = _plan(prompt_len, start0 = start0)
        assert chunks[0][0] == start0
        assert _sizes(chunks) == expected
        assert sum(_sizes(chunks)) == prompt_len - 1 - start0
        for (start, end, pred), (_, next_end, _) in zip(chunks, chunks[1:]):
            assert pred == next_end


def test_slower_growth_schedule(monkeypatch):
    # EXL3_NGRAM_RAMP_GROW=1.5 gives page-rounded intermediate steps; the plan still tiles and
    # the look-ahead still names the next chunk
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", True)
    monkeypatch.setattr(jobmod, "NGRAM_RAMP_FIRST", PAGE_SIZE)
    monkeypatch.setattr(jobmod, "NGRAM_RAMP_GROW", 1.5)
    chunks = _plan(4096)
    # the 1536-capped chunk is cut at the last full page (3840), then the partial page
    assert _sizes(chunks) == [PAGE_SIZE, 512, 768, 1024, 1280, 255]
    assert sum(_sizes(chunks)) == 4095
    for (start, end, pred), (_, next_end, _) in zip(chunks, chunks[1:]):
        assert pred == next_end


def test_gate_conditions(monkeypatch):
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", True)
    monkeypatch.setattr(jobmod, "NGRAM_RAMP_FIRST", PAGE_SIZE)
    off = [
        dict(chunk_size = 1024),            # prefill shared with a generating job
        dict(embeddings = [object()]),      # multimodal chunking
        dict(loaded_tp = True),             # TP prefill does no row prefetch
        dict(prefetch_layers = False),      # no table streamed for this model
        dict(max_chunk_size = PAGE_SIZE),   # nothing to ramp to
    ]
    for kwargs in off:
        j = _stub(**kwargs)
        assert j._ramp_gate() is False, kwargs
        cap = j._ramp_peek()
        assert cap == (kwargs.get("chunk_size") or j.generator.max_chunk_size)
    j = _stub()
    assert j._ramp_gate() is True
    assert j._ramp_peek() == PAGE_SIZE
    assert j._ramp_peek_next() == 2 * PAGE_SIZE
    j._ramp_next = MAX_CHUNK
    assert j._ramp_peek_next() == MAX_CHUNK
    # NGRAM_RAMP=0 disables it
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", False)
    assert _stub()._ramp_gate() is False


def test_no_recurrent_last_page_rule_without_recurrent_cache(monkeypatch):
    # dense models keep their chunk (no split at the last page)
    monkeypatch.setattr(jobmod, "NGRAM_RAMP", False)
    j = _stub(recurrent_cache = False)
    assert _loop_end(j, 2048, 4096) == 4095      # not cut at the last full page
    assert jobmod.Job._next_chunk_end(
        j, SimpleNamespace(sequence_ids = list(range(4096))), 2048, None) == 4095
    # the recurrent model in the same position is cut at the last full page
    j = _stub(recurrent_cache = True)
    assert _loop_end(j, 2048, 4096) == 3840
    assert jobmod.Job._next_chunk_end(
        j, SimpleNamespace(sequence_ids = list(range(4096))), 2048, None) == 3840
