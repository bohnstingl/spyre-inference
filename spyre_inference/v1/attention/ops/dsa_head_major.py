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

"""DSA top-k mask over a head-major latent KV cache. See ``spyre_inference.v1.attention.dsa``."""

import torch

from spyre_inference.v1.attention.ops import tile_loop


def dsa_mask_head_major_kernel(
    index_q,
    index_w,
    query_row_index,
    k_pages,
    page_index_table,
    mask_stack,
    num_blocks,
    padded_query_len,
    key_offset,
    index_heads,
    index_topk,
    steps,
):
    """One sequence's mask stack with everything outside its top ``index_topk`` masked.

    The lightning-indexer score of query ``t`` and cached token ``j`` is
    ``sum_h w[t, h] * relu(q[t, h] . k[j])``, ``k`` being columns
    ``key_offset : key_offset + index_head_dim`` of the latent row.

    Expected shapes:
        index_q: [rows, index_heads, index_head_dim], softmax scale folded in
        index_w: [rows, >= index_heads], head weights (zero past index_heads)
        query_row_index: [padded_query_len] int32, this sequence's absolute rows
        k_pages: [num_pages_total, block_size, head_size], the latent cache folded
        page_index_table, mask_stack: the paged walk's, [num_blocks, ...] on dim 0

    The k-th largest score per row is found by ``steps`` bisection steps rather than
    ``torch.topk``, which on Spyre takes k <= 128 and selects wrongly past 1k-wide rows.
    Rows that see no more than ``index_topk`` tokens keep every valid one.
    """
    head_size = k_pages.shape[-1]
    index_head_dim = index_q.shape[-1]
    # Zero outside the indexer key, so the whole latent row is the matmul operand: an
    # in-graph slice of the gathered page does not lower.
    q = index_q.index_select(0, query_row_index)
    q = torch.nn.functional.pad(q, (key_offset, head_size - key_offset - index_head_dim))
    w = index_w.index_select(0, query_row_index)[:, :index_heads]
    base = mask_stack[:num_blocks]

    def block_body(_, tiles):
        page_index, k_pages, q, w = tiles
        page = k_pages.index_select(0, page_index[0, 0:1]).squeeze(0)
        scores = torch.relu(torch.matmul(q, page.t().unsqueeze(0)))
        return None, torch.bmm(w.unsqueeze(1), scores).squeeze(1).unsqueeze(0)

    operands = (page_index_table[:num_blocks], k_pages, q, w)
    if tile_loop.USE_FOR_EACH_TILE:
        from torch_spyre._inductor.wsr import for_each_tile

        _, raw = for_each_tile(
            block_body, operands, dims=(0, None, None, None), tile_size=1, out_dim=0
        )
    else:
        raw = torch.cat(
            [
                block_body(None, (operands[0][b : b + 1], *operands[1:]))[1]
                for b in range(num_blocks)
            ]
        )

    # [num_blocks, padded_query_len, block_size]: each query row reduces over dims 0 and 2.
    scores = raw + base
    # The range of the valid scores only: a masked lane (stale page rows, future tokens)
    # sits near the fp16 floor in `scores` and far below it in `base - raw`.
    lo = -torch.amax(base - raw, dim=(0, 2), keepdim=True)
    hi = torch.amax(scores, dim=(0, 2), keepdim=True)
    for _ in range(steps):
        mid = (lo + hi) * 0.5
        # fp32: an fp16 sum of 0/1 flags stops being exact past 2048 (GLM-5's index_topk).
        count = (scores >= mid).to(torch.float32).sum(dim=(0, 2), keepdim=True)
        enough = (count >= index_topk).to(scores.dtype)
        lo = mid * enough + lo * (1 - enough)
        hi = hi * enough + mid * (1 - enough)
    # Arithmetic rather than torch.where, whose fused select does not lower here.
    keep = (scores >= lo).to(scores.dtype)
    floor = torch.finfo(scores.dtype).min
    return (base + (keep - 1.0) * -floor).clamp(min=floor)
