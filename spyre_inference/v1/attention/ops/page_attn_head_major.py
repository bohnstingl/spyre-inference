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

"""Per-sequence paged attention over a head-major KV cache, keeping the page LX-resident.

Two shape choices keep a gathered page in LX rather than round-tripping it through HBM,
over the cache folded to ``[pages * kv, block_size, D]``: the page is gathered on
(page, kv_head) so the gather's split lands per kv head, an output axis of ``probs @ V``
the consumer can mirror; and the query groups are unrolled, since the batched GQA form
leaves the page with two batch dims and Inductor clones it out to a query-group axis it
does not have (torch-spyre#4123).
"""

import torch

from spyre_inference.v1.attention.ops.online_softmax import (
    OnlineSoftmaxCarry,
    online_softmax_step,
)


def page_attn_head_major_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    kv_index_tables,
    head_index_tables,
    mask_tiles,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Online softmax attention over ``num_blocks`` KV pages.

    The unrolled reference for `page_attn_head_major_decode_kernel`, which folds the
    query groups into the row axis instead. Not dispatched to: the impl wires the
    folded kernel, and this one is the equivalence baseline its tests compare
    against, plus the LX residency probe. Keep the two in step.

    Under `dynamic=False` Dynamo specializes on every non-tensor argument, so the page
    loop is unrolled per variant.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: int32 device tensor whose first padded_query_len entries are this
            sequence's absolute query rows.
        k_pages / v_pages: [num_pages_total * num_kv_heads, block_size, head_size]
        kv_index_tables: per active block, a [num_kv_heads, 1] int32 device tensor of that
            block's ``page * num_kv_heads + kv`` rows. One real tensor per block, not a
            slice of a table: an index tensor reaches the hardware as a tensor argument,
            so a slice's nonzero storage offset is dropped (torch-spyre#3770).
        head_index_tables: per query group, a [num_kv_heads] int32 device tensor of that
            group's head ids (``kv * num_queries_per_kv + g``).
        mask_tiles: [num_blocks], each [padded_query_len, block_size]
        out: buffer to store into, or None to return the result instead.

    Returns [padded_query_len, num_heads, head_size], or ``out``.
    """
    num_queries_per_kv = num_heads // num_kv_heads

    # Gathered, not sliced: a compiled region reads a view from offset 0 and ignores its
    # strides (torch-spyre#3770).
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    # Rows before heads: selecting heads first keeps every staging row, so each group
    # would build a full-height intermediate and gather one row back out of it.
    q_groups = [
        q_rows.index_select(1, head_index_tables[g]).transpose(0, 1)
        for g in range(num_queries_per_kv)
    ]

    # One carry per query group; None until that group's first block folds in.
    carries: list[OnlineSoftmaxCarry | None] = [None] * num_queries_per_kv

    for i in range(num_blocks):
        # Subscripting, not index_select, which takes only a 1-D index: that puts the
        # entry axis on the index's own stick axis, splittable only in whole 32-entry
        # sticks. [num_kv_heads, 1] lets the split land per kv head.
        kv_rows = kv_index_tables[i]
        k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        k_t = k_page.permute(0, 2, 1)
        mask_tile = mask_tiles[i]

        for g in range(num_queries_per_kv):
            scores = torch.matmul(q_groups[g], k_t) * scale
            if logits_soft_cap > 0.0:
                # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so capping
                # after it would un-mask the padded lanes.
                scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
            scores = scores + mask_tile

            carries[g] = online_softmax_step(carries[g], scores, v_page)

    groups = []
    for carry in carries:
        assert carry is not None, "num_blocks must be at least 1"
        _, group_sum, group_out = carry
        groups.append(group_out / group_sum)
    attn = torch.stack(groups, dim=1)
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if out is not None:
        # Storing the full padded extent keeps this sequence's real query_len out of the
        # arguments, so it is not specialized on; rows past it duplicate the last row.
        out.index_copy_(0, query_row_index[:padded_query_len], attn[:padded_query_len])
        return out
    return attn


def page_attn_head_major_decode_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    kv_index_tables,
    mask_tiles,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Decode (Q=1) attention with the query groups folded into the row axis.

    Heads are kv-major, so the fold is a reshape: the page keeps a single batch dim, where
    the batched GQA form gives it a group axis Inductor clones it out to
    (torch-spyre#4123). The fold needs no head gather, so this takes no head index tables:
    passing them costs argument marshalling per call for tensors the graph never reads.
    """
    assert padded_query_len == 1, "decode kernel is specialized for a single query row"
    # The dispatch walks max(2, active blocks) so the token-major kernel's symbolic
    # count never takes the value Dynamo specializes; the pools must cover that,
    # not just the active count. Both sides are static here, so this is free.
    assert len(kv_index_tables) >= num_blocks, (
        f"{len(kv_index_tables)} index tables for a {num_blocks}-block walk"
    )
    # `len`, not `.shape[0]`: the pool is a stack from the impl and a list of tiles
    # from the equivalence tests.
    assert len(mask_tiles) >= num_blocks, (
        f"mask pool of height {len(mask_tiles)} for a {num_blocks}-block walk"
    )
    num_queries_per_kv = num_heads // num_kv_heads

    row = query_row_index[:1]
    q = query.index_select(0, row).reshape(num_kv_heads, num_queries_per_kv, head_size)

    carry: OnlineSoftmaxCarry | None = None

    for i in range(num_blocks):
        kv_rows = kv_index_tables[i]
        k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)

        scores = torch.matmul(q, k_page.permute(0, 2, 1)) * scale
        if logits_soft_cap > 0.0:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        # At one query row the mask is head-independent, so its [1, block_size] tile
        # broadcasts across the folded group axis.
        scores = scores + mask_tiles[i]

        carry = online_softmax_step(carry, scores, v_page)

    assert carry is not None, "num_blocks must be at least 1"
    _, tile_sum, tile_out = carry
    attn = (tile_out / tile_sum).reshape(1, num_heads, head_size)
    if out is not None:
        out.index_copy_(0, row, attn)
        return out
    return attn
