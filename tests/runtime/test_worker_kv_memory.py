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

"""TorchSpyreWorker sizes the KV cache through upstream's Worker.determine_available_memory
(spyre-inference#901), capped at what a full batch at max_model_len can use. Only the
card is faked: torch.accelerator's memory view, with free bytes the test sets."""

from types import SimpleNamespace

import torch
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import request_memory

from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

GiB = 1 << 30
TOTAL = 96 * GiB
BLOCK_BYTES = 2 * 2 * 4 * 128 * 128  # K+V, fp16, 4 kv heads, head 128, block 128


def _worker(
    monkeypatch,
    *,
    free_after_load: int,
    max_num_seqs: int = 4,
    max_model_len: int = 1024,
    kv_cache_memory_bytes: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> TorchSpyreWorker:
    """A worker that started on an empty card and whose model now leaves free_after_load."""
    card = SimpleNamespace(free=TOTAL)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (card.free, TOTAL))
    monkeypatch.setattr(torch.accelerator, "memory_stats", lambda *_: {})
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda *_: 0)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda *_: None)

    spec = FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=dtype)
    worker = object.__new__(TorchSpyreWorker)
    worker.vllm_config = SimpleNamespace()
    worker.parallel_config = SimpleNamespace(data_parallel_size=1)
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=kv_cache_memory_bytes, gpu_memory_utilization=0.9, block_size=128
    )
    worker.model_config = SimpleNamespace(max_model_len=max_model_len, multimodal_config=None)
    worker.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    worker.model_runner = SimpleNamespace(
        get_kv_cache_spec=lambda: {f"layers.{i}.attn": spec for i in range(2)},
        profile_run=lambda **_: None,
        model_memory_usage=0,
    )
    worker.init_snapshot = MemorySnapshot(device=torch.device("cpu"))
    worker.requested_memory = request_memory(worker.init_snapshot, worker.cache_config)
    card.free = free_after_load
    return worker


def test_a_short_batch_is_capped_at_the_batch_area(monkeypatch):
    """Memory for far more pages than max_num_seqs × max_model_len is not handed out:
    the single block pool could never allocate them."""
    worker = _worker(monkeypatch, free_after_load=80 * GiB)

    budget = worker.determine_available_memory()

    assert budget // (2 * BLOCK_BYTES) == 4 * (1024 // 128) + 1


def test_the_budget_is_the_requested_share_minus_what_loading_consumed(monkeypatch):
    worker = _worker(monkeypatch, free_after_load=60 * GiB, max_num_seqs=1024, max_model_len=131072)

    assert worker.determine_available_memory() == worker.requested_memory - 36 * GiB


def test_an_explicit_kv_cache_memory_bytes_skips_the_measurement(monkeypatch):
    worker = _worker(
        monkeypatch,
        free_after_load=0,
        max_num_seqs=1024,
        max_model_len=131072,
        kv_cache_memory_bytes=3 * GiB,
    )

    assert worker.determine_available_memory() == 3 * GiB


def test_an_explicit_kv_cache_memory_bytes_is_still_capped(monkeypatch):
    """A leftover VLLM_CPU_KVCACHE_SPACE=64 must not turn into 64 GiB of pages."""
    worker = _worker(monkeypatch, free_after_load=0, kv_cache_memory_bytes=64 * GiB)

    assert worker.determine_available_memory() // (2 * BLOCK_BYTES) == 4 * (1024 // 128) + 1


def test_loading_past_the_requested_share_leaves_nothing(monkeypatch):
    """vLLM then fails at init, before any allocation, naming gpu_memory_utilization."""
    worker = _worker(monkeypatch, free_after_load=5 * GiB)

    assert worker.determine_available_memory() <= 0


def test_an_fp8_cache_fits_twice_the_blocks(monkeypatch):
    """The budget is bytes and vLLM divides it by the spec's page size, so a 1-byte
    cache holds twice the tokens in the same memory."""
    blocks = {}
    for dtype in (torch.float16, torch.uint8):
        worker = _worker(
            monkeypatch,
            free_after_load=60 * GiB,
            max_num_seqs=1024,
            max_model_len=131072,
            dtype=dtype,
        )
        page = next(iter(worker.model_runner.get_kv_cache_spec().values())).page_size_bytes
        blocks[dtype] = worker.determine_available_memory() // (2 * page)

    assert blocks[torch.uint8] == 2 * blocks[torch.float16]
