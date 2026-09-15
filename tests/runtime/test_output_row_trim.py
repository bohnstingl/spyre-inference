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

"""The model-output D2H copies only the rows sampling will read."""

import torch
import torch.nn as nn

from spyre_inference.v1.worker import spyre_model_runner as mr
from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

HIDDEN = 8


class _Body(nn.Module):
    """Returns a body whose row ``i`` is filled with the value ``i``."""

    def forward(self, input_ids=None, **kwargs):
        rows = input_ids.shape[0]
        return torch.arange(rows, dtype=torch.float16).unsqueeze(1).expand(rows, HIDDEN)


def _wrapper(monkeypatch, *, buckets, trim_body_tokens, copied):
    """Wrapper whose "device" is CPU, recording every row count copied host-ward."""

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
        trim_body_tokens=trim_body_tokens,
    )


def _run(wrapper, num_tokens, rows=None):
    ids = torch.zeros(num_tokens, dtype=torch.int64)
    if rows is None:
        return wrapper(input_ids=ids)
    with wrapper.sampled_rows(torch.tensor(rows, dtype=torch.int64)):
        return wrapper(input_ids=ids)


def test_copies_only_the_sampled_rows(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], trim_body_tokens=512, copied=copied)

    out = _run(wrapper, 512, rows=[511])

    # One row crosses to the host, not 512 ...
    assert copied == [1]
    # ... and it still reads back at its original body offset.
    assert out.shape == (512, HIDDEN)
    torch.testing.assert_close(out[511], torch.full((HIDDEN,), 511.0, dtype=torch.float16))


def test_sampled_rows_land_at_their_own_offsets(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 2, 4], trim_body_tokens=512, copied=copied)

    out = _run(wrapper, 512, rows=[127, 300, 511])

    # 3 rows padded up onto the next warmed row bucket (4), one copy.
    assert copied == [4]
    for row in (127, 300, 511):
        torch.testing.assert_close(out[row], torch.full((HIDDEN,), float(row), dtype=torch.float16))


def test_unarmed_call_copies_the_whole_body(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], trim_body_tokens=512, copied=copied)

    _run(wrapper, 512)

    assert copied == [512]


def test_arming_does_not_leak_into_the_next_call(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], trim_body_tokens=512, copied=copied)

    _run(wrapper, 512, rows=[511])
    _run(wrapper, 512)

    assert copied == [1, 512]


def test_other_body_widths_keep_the_full_copy(monkeypatch):
    """Only the warmed (prefill) width trims; anything else has no gather graph."""
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 2, 4], trim_body_tokens=512, copied=copied)

    _run(wrapper, 4, rows=[0])

    assert copied == [4]


def test_full_copy_when_the_padded_gather_is_as_wide_as_the_body(monkeypatch):
    """8 rows padded onto bucket 8 is the whole body — copying it directly is cheaper."""
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 8], trim_body_tokens=8, copied=copied)

    _run(wrapper, 8, rows=list(range(8)))

    assert copied == [8]
