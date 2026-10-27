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

"""SPYRE_MAX_CONCURRENT_LOADS: at most N ranks hold their host copy of the weights at once.

A model larger than host RAM (MiniMax-M3 at TP16: 852 GB against 755 GB) only loads if the
ranks are staggered; vLLM's max_parallel_loading_workers is ignored upstream.
"""

import multiprocessing
import time

import pytest


def _rank(rank, world_size, log, lock):
    from spyre_inference.v1.worker import load_gate

    with load_gate.load_slot(rank, world_size):
        with lock:
            log.append(("in", rank, time.monotonic()))
        time.sleep(0.3)
        with lock:
            log.append(("out", rank, time.monotonic()))


def _max_concurrent(log):
    inside = peak = 0
    for kind, _, _ in sorted(log, key=lambda e: e[2]):
        inside += 1 if kind == "in" else -1
        peak = max(peak, inside)
    return peak


@pytest.mark.parametrize("limit,expected", [(2, 2), (0, 4)])
def test_load_slots_bound_concurrent_ranks(monkeypatch, limit, expected):
    monkeypatch.setenv("SPYRE_MAX_CONCURRENT_LOADS", str(limit))
    ctx = multiprocessing.get_context("fork")
    with ctx.Manager() as manager:
        log, lock = manager.list(), manager.Lock()
        procs = [ctx.Process(target=_rank, args=(r, 4, log, lock)) for r in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0
        assert _max_concurrent(list(log)) == expected
