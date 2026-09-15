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


def page_attn_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_stack,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    logits_soft_cap=0.0,
    alibi_stack=None,
    out=None,
):
    """Online softmax attention over ``num_blocks`` KV pages.

    The page walk is one `for_each_tile` level, not a Python loop, so the traced
    graph holds a single block body instead of `num_blocks` copies of one. Under
    `dynamic=False` Dynamo still specializes on every non-tensor argument, so
    there is a variant per (num_blocks, padded_query_len, ...) combination, but
    each variant's graph — and its compile time — no longer grows with
    num_blocks.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: int32 device tensor whose first padded_query_len
            entries are this sequence's absolute query rows.
        k_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        v_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
            tensor, row i holding the i-th active block's page index at
            column 0; the other 31 columns are stick padding. Extra rows are
            tolerated and ignored, as the unrolled loop tolerated them.
        mask_stack: [num_blocks, padded_query_len, block_size] additive mask,
            block-major. One stacked tensor, not a list: for_each_tile tiles a
            tensor axis. See _mirror_mask_stacks.
        alibi_stack: [num_blocks, num_kv_heads, num_queries_per_kv, 1, block_size],
            or None for no ALiBi. The query-axis dim is 1 because softmax absorbs
            per-query-row constants — see the derivation at the bias-tile
            construction site in _online_softmax_attention.
        out: buffer to store into, or None to return the result instead.

    Returns [padded_query_len, num_heads, head_size], or ``out`` when this
    kernel stored the result itself.
    """
    from torch_spyre._inductor.wsr import for_each_tile

    # num_blocks is the tiled operands' dim-0 extent now, not a Python range, so a
    # mask_stack that disagrees runs a different number of blocks than was asked
    # for, rather than the extra rows being ignored.
    assert mask_stack.shape[0] == num_blocks, (
        f"mask_stack has {mask_stack.shape[0]} block rows, expected {num_blocks}"
    )
    if alibi_stack is not None:
        assert alibi_stack.shape[0] == num_blocks, (
            f"alibi_stack has {alibi_stack.shape[0]} block rows, expected {num_blocks}"
        )

    num_queries_per_kv = num_heads // num_kv_heads
    # A compiled region reads a view from offset 0, ignoring storage_offset
    # (torch-spyre#3770), so the rows are gathered here rather than sliced outside.
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )

    # The pages are gathered one per trip inside the body, exactly as the unrolled
    # loop did: the block table is the tiled operand, the two caches are passed
    # whole, and only the sequence's current K and V page is live at a time. The
    # per-trip page index is an address that moves with the loop var and carries no
    # iteration dim of its own, which coarse tiling handles via
    # squeezed_advance_per_read (torch-spyre's _point_splice_advance_for_dep).
    #
    # The row narrow is a no-op for both callers; it is here so an over-long table
    # is ignored rather than silently adding trips.
    operands = [page_index_table[:num_blocks], k_pages, v_pages, mask_stack, q]
    dims: list[int | None] = [0, None, None, 0, None]
    use_alibi = alibi_stack is not None
    if use_alibi:
        # A trace-time branch, so the body is specialized either way and a
        # non-ALiBi layer pays nothing. for_each_tile operands cannot be None,
        # which is why this is a conditional operand and not a zero tensor.
        operands.append(alibi_stack)
        dims.append(0)

    def block_body(carry, tiles):
        tile_max, tile_sum, tile_output = carry
        # Tiles are rank-preserving (narrow, not select), so each keeps a leading
        # 1: the table row is [1, INT32_ELEMS_PER_STICK] and the mask
        # [1, padded_query_len, block_size]. The two caches are invariant, so
        # they arrive whole.
        if use_alibi:
            table_row, k_all, v_all, mask_row, q_whole, alibi_row = tiles
        else:
            table_row, k_all, v_all, mask_row, q_whole = tiles
            alibi_row = None

        # index_select, not `k_all[page_idx]`: subscripting lowers to aten.index,
        # which upcasts the int32 index to int64 and fails eager. The 0:1 narrow
        # keeps the index a 1-element tensor, which is what index_select wants.
        page_idx = table_row[0, 0:1]
        # [1, block_size, num_kv_heads, head_size] -> the matmul's
        # [num_kv_heads, 1, *, *], with the query-group axis the matmuls
        # broadcast over opened up next to the KV-head axis. K is transposed
        # here so scores is a plain matmul.
        k_page = k_all.index_select(0, page_idx).squeeze(0)
        v_page = v_all.index_select(0, page_idx).squeeze(0)
        k_page_t = k_page.permute(1, 2, 0).unsqueeze(1)
        v_page_4d = v_page.permute(1, 0, 2).unsqueeze(1)
        mask_tile = mask_row[0]

        scores = torch.matmul(q_whole, k_page_t) * scale
        if logits_soft_cap > 0.0:
            # Pull logits into (-cap, +cap) before the mask add so masked
            # positions still map cleanly to -inf. Applied before the ALiBi
            # bias so the positional term is not squashed by the tanh.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if alibi_row is not None:
            # ALiBi bias slope[h] * (kv_pos - context_len). The additive
            # mask_tile below uses finfo.min for masked positions, so this
            # bias cannot un-mask them.
            scores = scores + alibi_row[0]
        scores = scores + mask_tile

        new_max = torch.maximum(tile_max, torch.amax(scores, dim=-1, keepdim=True))
        rescale = torch.exp(tile_max - new_max)
        tile_probs = torch.exp(scores - new_max)
        new_sum = tile_sum * rescale + tile_probs.sum(dim=-1, keepdim=True)
        new_output = tile_output * rescale + torch.matmul(tile_probs, v_page_4d)
        return (new_max, new_sum, new_output), None

    # A (-inf, 0, 0) start makes the general update reproduce the block-0 one, so
    # no iteration is peeled: exp(-inf - m) == 0 zeroes the rescale on the first
    # trip. Safe for every mask this builder emits, all-masked rows included,
    # because masked positions carry finfo.min rather than -inf — so this is
    # always -inf minus something finite, never the undefined -inf - -inf.
    state_shape = (num_kv_heads, num_queries_per_kv, padded_query_len, 1)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    (_, tile_sum, tile_output), _ = for_each_tile(
        block_body,
        tuple(operands),
        # Blocks slice the block table, the mask and the bias; the caches and
        # the query are read whole on every step.
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
        out.index_copy_(0, query_row_index[:padded_query_len], attn[:padded_query_len])
        return out
    return attn
