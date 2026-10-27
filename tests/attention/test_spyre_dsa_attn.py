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

"""DSA top-k mask over a head-major latent cache, against the fp32 lightning indexer."""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops.utils import convert, row_outermost_layout
from spyre_inference.v1.attention.ops.dsa_head_major import dsa_mask_head_major_kernel
from spyre_inference.v1.attention.ops.layout import INT32_ELEMS_PER_STICK, head_major_kv_layout

pytestmark = pytest.mark.attention

# GLM-5.x: 576-wide latent, 32 indexer heads of 128, top-2048.
LATENT, INDEX_HEADS, INDEX_DIM = 576, 32, 128
HEAD = LATENT + INDEX_DIM
BLOCK = 128
FLOOR = torch.finfo(torch.float16).min


@pytest.fixture()
def device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


def _indexer_case(kv_len: int, query_len: int, num_blocks: int):
    """A sequence's latent pages (indexer key in columns 576:704), index inputs and mask."""
    torch.manual_seed(0)
    keys = torch.randn(num_blocks * BLOCK, INDEX_DIM) * 0.5
    rows = torch.cat([torch.randn(num_blocks * BLOCK, LATENT) * 0.5, keys], dim=-1)
    # Pages out of order, and page 0 left unused, as vLLM's null block is.
    page_ids = torch.randperm(num_blocks) + 1
    cache = torch.zeros(num_blocks + 1, 1, BLOCK, HEAD)
    for i, page in enumerate(page_ids):
        cache[page, 0] = rows[i * BLOCK : (i + 1) * BLOCK]
    q = torch.randn(query_len + 1, INDEX_HEADS, INDEX_DIM) * INDEX_DIM**-0.5
    w = torch.zeros(query_len + 1, 64)
    w[:, :INDEX_HEADS] = torch.randn(query_len + 1, INDEX_HEADS) * INDEX_HEADS**-0.5

    q_pos = kv_len - query_len + torch.arange(query_len)
    valid = torch.arange(num_blocks * BLOCK)[None, :] <= q_pos[:, None]
    base = torch.where(valid, 0.0, FLOOR).view(query_len, num_blocks, BLOCK).transpose(0, 1)
    table = torch.zeros(num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    table[:, 0] = page_ids
    return cache, q, w, keys, valid, base.contiguous(), table


@pytest.mark.parametrize(
    ("kv_len", "query_len", "index_topk"),
    [
        (4096, 64, 2048),  # prefill past GLM-5's index_topk
        (6000, 1, 2048),  # decode, ragged last page
        (1500, 64, 2048),  # context below index_topk keeps every valid token
        (1024, 64, 64),
    ],
)
def test_dsa_mask_keeps_the_fp32_topk(device, kv_len, query_len, index_topk):
    """The mask keeps the fp32 indexer's top-k per query row, plus at most a few fp16
    near-ties at the k-th score, and never a masked (future or padded) position."""
    num_blocks = -(-kv_len // BLOCK)
    cache, q, w, keys, valid, base, table = _indexer_case(kv_len, query_len, num_blocks)

    scores = torch.einsum(
        "qh,qhk->qk", w[:query_len, :INDEX_HEADS], torch.relu(q[:query_len] @ keys.t())
    )
    scores = scores.masked_fill(~valid, float("-inf"))
    k = min(index_topk, kv_len)
    want = torch.zeros_like(valid).scatter(1, scores.topk(k, -1).indices, True) & valid

    layout = head_major_kv_layout(num_blocks + 1, BLOCK, HEAD, torch.float16)
    pages = cache.half().to(device, device_layout=layout).view(num_blocks + 1, BLOCK, HEAD)
    q_dev = convert(q.half(), device, device_layout=row_outermost_layout(q.shape, torch.float16))
    w_dev = convert(w.half(), device, device_layout=row_outermost_layout(w.shape, torch.float16))
    mask = torch.compile(dsa_mask_head_major_kernel, dynamic=False, fullgraph=True)(
        q_dev,
        w_dev,
        convert(torch.arange(query_len, dtype=torch.int32), device),
        pages,
        convert(table, device),
        convert(base.half(), device),
        num_blocks,
        query_len,
        LATENT,
        INDEX_HEADS,
        index_topk,
        16,
    )
    got = (mask.cpu().float() > -1).transpose(0, 1).reshape(query_len, -1)

    assert not (got & ~valid).any(), "a masked position was kept"
    # The scores are fp16 on the device, so tokens tied with the k-th fp32 score within
    # fp16 resolution may swap; anything further from the boundary must be decided right.
    kth = scores.topk(k, -1).values[:, -1:]
    finite = scores.masked_fill(~valid, 0.0)
    tol = 2e-3 * finite.abs().amax(-1, keepdim=True)
    assert not (got & (scores < kth - tol)).any(), "kept a token well below the k-th score"
    assert not (~got & valid & (scores > kth + tol)).any(), "dropped a token well above it"
    assert (want & ~got).sum(-1).max() <= max(2, k // 500)
