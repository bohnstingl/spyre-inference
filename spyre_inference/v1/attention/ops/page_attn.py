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

"""Per-sequence paged attention over the KV cache."""

import torch

from spyre_inference.v1.attention.ops.online_softmax import online_softmax_step
from spyre_inference.v1.attention.ops.tile_loop import walk_tiles


def page_attn_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_index_table,
    mask_stack,
    scale,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    logits_soft_cap=0.0,
    alibi_stack=None,
    out=None,
):
    """Online softmax attention over the KV pages the index tables name.

    Under `dynamic=False` Dynamo specializes on every non-tensor argument, so a Python
    page loop is unrolled per variant. `walk_tiles` holds one block body instead when
    SPYRE_ATTN_FOR_EACH_TILE is set.

    The block count is deliberately not an argument: it is the tables' dim 0, which
    the caller marks dynamic, and comparing it here against a Python int would
    install the very guard that makes one trace serve every count. The launch
    boundary cross-checks the two instead.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: [padded_query_len] int32 device tensor of this
            sequence's absolute query rows.
        k_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        v_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
            tensor, row i holding the i-th active block's page index at
            column 0.
        mask_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
            tensor, row i holding the mask pool row at column 0.
        mask_stack: [max_num_blocks, padded_query_len, block_size] additive
            mask pool, gathered through mask_index_table.
        alibi_stack: [max_num_blocks, num_kv_heads, num_queries_per_kv, 1,
            block_size], gathered like mask_stack,
            or None for no ALiBi. The query-axis dim is 1 because softmax absorbs
            per-query-row constants — see the derivation at the bias-tile
            construction site in _online_softmax_attention.
        out: buffer to store into, or None to return the result instead.

    Returns [padded_query_len, num_heads, head_size], or ``out`` when this
    kernel stored the result itself.
    """
    num_queries_per_kv = num_heads // num_kv_heads
    # Gathered, not sliced outside: since torch-spyre#4449 a view's storage_offset is a
    # Dynamo graph guard, and q_start varies, so a slice would compile one kernel per batch
    # layout -- test_spyre_compile_input_offset_specialises_the_graph. The builder now
    # creates this table at exactly padded_query_len rows.
    q_rows = query.index_select(0, query_row_index)
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )

    # Both walks tile tensor axes, so every per-block operand arrives stacked on dim 0.
    operands = [
        page_index_table,
        mask_index_table,
        k_pages,
        v_pages,
        mask_stack,
        q,
    ]
    dims: list[int | None] = [0, 0, None, None, None, None]
    if alibi_stack is not None:
        operands.append(alibi_stack)
        dims.append(None)

    def block_body(carry, tiles):
        page_index, mask_index, k_pages, v_pages, mask_pool, q, *rest = tiles
        page_idx = page_index[0, 0:1]
        mask_idx = mask_index[0, 0:1]
        # index_select, not `k_pages[page_idx]`: subscripting lowers to
        # aten.index, which upcasts the int32 index to int64 and fails eager.
        k_page = k_pages.index_select(0, page_idx)
        v_page = v_pages.index_select(0, page_idx)
        mask_tile = mask_pool.index_select(0, mask_idx)
        # Token-major page to head-major for the matmuls; permutes on device.
        k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
        v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Pull logits into (-cap, +cap) before the mask add so masked
            # positions still map cleanly to -inf. Applied before the ALiBi
            # bias so the positional term is not squashed by the tanh.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if rest:
            # ALiBi bias slope[h] * (kv_pos - context_len). The additive
            # mask_tile below uses finfo.min for masked positions, so this
            # bias cannot un-mask them.
            scores = scores + rest[0].index_select(0, mask_idx)[0]
        scores = scores + mask_tile[0]

        return online_softmax_step(carry, scores, v_page_4d), None

    state_shape = (num_kv_heads, num_queries_per_kv, padded_query_len, 1)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    (_, tile_sum, tile_output), _ = walk_tiles(
        block_body,
        tuple(operands),
        dims=tuple(dims),
        tile_size=1,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(
                (num_kv_heads, num_queries_per_kv, padded_query_len, head_size),
                **state_kwargs,
            ),
        ),
    )
    attn = tile_output / tile_sum
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if out is not None:
        # `out` and `query` are both indexed by absolute token row. Storing the
        # full padded extent keeps the sequence's real query_len out of the
        # arguments, so it is not specialized on; rows past it duplicate the
        # sequence's last row, so index_copy_'s undefined write order for
        # duplicate indices is harmless.
        out.index_copy_(0, query_row_index, attn)
        return out
    return attn
