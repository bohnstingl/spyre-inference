# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampled hidden-state rows are gathered on Spyre before output D2H.

The gather only pays off once the body copy it removes is large enough to cover
the gather kernel, so these use a realistic ``HIDDEN`` and body sizes: a toy
hidden size would fall under ``_OUTPUT_TRIM_MIN_BYTES`` and never trim.
"""

import torch
import torch.nn as nn

from spyre_inference.v1.worker import spyre_model_runner as mr
from spyre_inference.v1.worker.spyre_model_runner import (
    _OUTPUT_TRIM_MIN_BYTES,
    _SpyreModelWrapper,
    _worth_trimming,
)

HIDDEN = 4096
# fp16: one row is 8 KiB, so a 512-row body is 4 MiB and trimming clears the
# byte floor by a wide margin.
BIG_BODY = 512


class _Body(nn.Module):
    def forward(self, input_ids=None, **kwargs):
        rows = input_ids.shape[0]
        return torch.arange(rows, dtype=torch.float16).unsqueeze(1).expand(rows, HIDDEN)


def _wrapper(monkeypatch, *, buckets, copied):
    def fake_convert(t, device=None, dtype=None):
        if device is not None and str(device) == "cpu" and t.dim() == 2:
            copied.append(t.shape[0])
        return t

    monkeypatch.setattr(mr, "convert", fake_convert)
    model = _Body()
    return _SpyreModelWrapper(
        model,
        torch.device("cpu"),
        logits_row_buckets=buckets,
    )


def _run(wrapper, num_tokens, rows=None):
    ids = torch.zeros(num_tokens, dtype=torch.int64)
    if rows is not None:
        object.__setattr__(wrapper, "_sample_rows", torch.tensor(rows, dtype=torch.int64))
    return wrapper(input_ids=ids)


def test_copies_only_the_sampled_rows(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)

    out = _run(wrapper, BIG_BODY, rows=[BIG_BODY - 1])

    assert copied == [1]
    assert out.shape == (1, HIDDEN)
    torch.testing.assert_close(
        out[0], torch.full((HIDDEN,), float(BIG_BODY - 1), dtype=torch.float16)
    )


def test_sampled_rows_are_returned_in_order(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 2, 4], copied=copied)

    out = _run(wrapper, BIG_BODY, rows=[127, 300, BIG_BODY - 1])

    assert copied == [4]
    assert out.shape == (3, HIDDEN)
    for result, row in zip(out, (127, 300, BIG_BODY - 1)):
        torch.testing.assert_close(result, torch.full((HIDDEN,), float(row), dtype=torch.float16))


def test_unarmed_call_copies_the_whole_body(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)

    _run(wrapper, BIG_BODY)

    assert copied == [BIG_BODY]


def test_arming_does_not_leak_into_the_next_call(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)

    _run(wrapper, BIG_BODY, rows=[BIG_BODY - 1])
    _run(wrapper, BIG_BODY)

    assert copied == [1, BIG_BODY]


def test_full_copy_when_the_padded_gather_is_as_wide_as_the_body(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 8], copied=copied)

    _run(wrapper, 8, rows=list(range(8)))

    assert copied == [8]


class TestSavingsGuard:
    """The trim is skipped when the gather would cost more than the copy it saves.

    The decision lives in the runner, before ``logits_indices`` is rewritten, so
    these cover the predicate directly; the runner wiring is covered by
    ``TestArming`` in ``test_warmup_logits_widths.py``.
    """

    def test_threshold_is_on_bytes_saved_not_rows(self):
        # Same row count either side; only the wide body moves enough bytes to
        # pay for the gather. A rows-based threshold would decide these the same
        # way and be wrong on any model with a different hidden size.
        assert _worth_trimming(64, 4, 4096)
        assert not _worth_trimming(64, 4, 64)

    def test_no_narrowing_is_rejected(self):
        assert not _worth_trimming(4, 4, 4096)
        assert not _worth_trimming(4, 8, 4096)

    def test_floor_is_exact(self):
        exact_rows = _OUTPUT_TRIM_MIN_BYTES // (4096 * 2)

        assert _worth_trimming(1 + exact_rows, 1, 4096)
        assert not _worth_trimming(exact_rows, 1, 4096)

    def test_itemsize_is_honoured(self):
        # fp32 halves the rows needed to clear the floor.
        rows_fp16 = _OUTPUT_TRIM_MIN_BYTES // (4096 * 2)
        assert not _worth_trimming(rows_fp16, 1, 4096, itemsize=2)
        assert _worth_trimming(rows_fp16, 1, 4096, itemsize=4)


class TestNoRoundTrip:
    """The gathered rows stay on Spyre for the lm_head instead of going back H2D.

    The wrapper's "device" is CPU in these tests, so a convert cannot be labelled
    H2D by its target device. Detect the round trip structurally instead: the H2D
    path is the only one that calls ``convert`` inside ``compute_logits``, and it
    is the only one that hands the lm_head a tensor that is not the gathered one.
    """

    @staticmethod
    def _instrument(monkeypatch, wrapper):
        """Record converts and the tensor the lm_head receives, per projection."""
        state: dict[str, object] = {"converts": 0, "seen": None}
        inner = mr.convert

        def counting_convert(t, device=None, dtype=None):
            if state.get("in_logits"):
                state["converts"] = int(state["converts"]) + 1
            return inner(t, device=device, dtype=dtype)

        monkeypatch.setattr(mr, "convert", counting_convert)

        def fake_compute_logits(hs, *a, **k):
            state["seen"] = hs
            return torch.zeros(hs.shape[0], 3)

        monkeypatch.setattr(wrapper._model, "compute_logits", fake_compute_logits, raising=False)
        return state

    @staticmethod
    def _project(wrapper, state, hidden):
        state["converts"] = 0
        state["in_logits"] = True
        try:
            wrapper.compute_logits(hidden)
        finally:
            state["in_logits"] = False
        return int(state["converts"]), state["seen"]

    def test_compute_logits_reuses_the_device_gather(self, monkeypatch):
        copied: list[int] = []
        wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)
        state = self._instrument(monkeypatch, wrapper)

        sampled = _run(wrapper, BIG_BODY, rows=[BIG_BODY - 1])
        gathered = wrapper._gathered_rows[1]
        converts, seen = self._project(wrapper, state, sampled)

        assert converts == 0, "the rows were already on device; no H2D expected"
        assert seen is gathered, "the lm_head must consume the gathered device tensor"

    def test_cache_is_consumed_once(self, monkeypatch):
        copied: list[int] = []
        wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)
        state = self._instrument(monkeypatch, wrapper)

        sampled = _run(wrapper, BIG_BODY, rows=[BIG_BODY - 1])
        first, _ = self._project(wrapper, state, sampled)
        # A second projection must not reuse a consumed cache.
        second, _ = self._project(wrapper, state, sampled)

        assert first == 0
        assert second == 1, "the consumed cache must fall back to an H2D"

    def test_mismatched_rows_fall_back_to_h2d(self, monkeypatch):
        copied: list[int] = []
        wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)
        state = self._instrument(monkeypatch, wrapper)

        _run(wrapper, BIG_BODY, rows=[BIG_BODY - 1])
        # Something transformed the rows between the gather and the projection.
        converts, _ = self._project(wrapper, state, torch.ones(1, HIDDEN, dtype=torch.float16))

        assert converts == 1

    def test_untrimmed_step_still_h2ds(self, monkeypatch):
        copied: list[int] = []
        wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)
        state = self._instrument(monkeypatch, wrapper)

        body = _run(wrapper, BIG_BODY)
        converts, _ = self._project(wrapper, state, body[:1])

        assert converts == 1
