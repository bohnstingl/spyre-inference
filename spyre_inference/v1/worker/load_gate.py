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

"""Bound how many TP ranks hold their weights in host RAM at once.

A rank loads its whole shard into host memory and only frees it once the weights are on its
card, so peak host RAM is about the model size when every rank loads together. vLLM's
``max_parallel_loading_workers`` would bound that, but it is ignored upstream. With
``SPYRE_MAX_CONCURRENT_LOADS=N``, rank ``r`` takes slot ``r % N``: one file lock per slot,
so ranks ``r`` and ``r + N`` load one after the other. The ranks of one engine are children
of the same executor process, whose pid keys the locks.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
from collections.abc import Iterator

from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)


@contextlib.contextmanager
def load_slot(rank: int, world_size: int) -> Iterator[None]:
    limit = envs.SPYRE_MAX_CONCURRENT_LOADS
    if limit <= 0 or limit >= world_size:
        yield
        return
    name = f"spyre_load_{os.getppid()}_{rank % limit}.lock"
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "w") as lock:
        logger.info("Rank %d waiting for weight-load slot %d of %d.", rank, rank % limit, limit)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
