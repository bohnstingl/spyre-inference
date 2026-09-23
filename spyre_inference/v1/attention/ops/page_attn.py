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

from spyre_inference.v1.attention.ops import tile_loop
from spyre_inference.v1.attention.ops.tile_loop import walk_tiles


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
    page_group=1,
    alibi_stack=None,
    out=None,
):
    """Online softmax attention over ``num_blocks`` KV pages.

    Under `dynamic=False` Dynamo specializes on every non-tensor argument, so a Python
    page loop is unrolled per variant. `walk_tiles` holds one block body instead when
    SPYRE_ATTN_FOR_EACH_TILE is set.

    ``page_group`` is the tile width: that many adjacent pages are gathered and reduced by
    one online-softmax update, which is one update and one gather per group instead of per
    page. The group stays a batch axis of the matmuls and is reduced away at the end of the
    body, so no two axes are ever merged -- the same shape batched decode gives its chunk.
    It must divide ``num_blocks``: a tile walk has no ragged final tile.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: [padded_query_len] int32 device tensor of this
            sequence's absolute query rows.
        k_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        v_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
            tensor, row i holding the i-th active block's page index at
            column 0.
        mask_stack: [num_blocks, padded_query_len, block_size], tiled on dim 0.
        page_group: adjacent pages per online-softmax update; must divide num_blocks.
        alibi_stack: [num_blocks, num_kv_heads, num_queries_per_kv, 1, block_size],
            or None for no ALiBi. The query-axis dim is 1 because softmax absorbs
            per-query-row constants — see the derivation at the bias-tile
            construction site in _online_softmax_attention.
        out: buffer to store into, or None to return the result instead.

    Returns [padded_query_len, num_heads, head_size], or ``out`` when this
    kernel stored the result itself.
    """
    num_queries_per_kv = num_heads // num_kv_heads
    if page_group < 1 or num_blocks % page_group:
        raise ValueError(
            f"page_group={page_group} must be a positive divisor of num_blocks={num_blocks}"
        )
    grouped = page_group > 1

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
    if grouped:
        # The group axis the tile's pages arrive on. Broadcast, not materialized: the
        # same query rows attend every page of the group.
        q = q.unsqueeze(0)

    # Both walks tile tensor axes, so every per-block operand arrives stacked on dim 0.
    operands = [
        page_index_table[:num_blocks],
        k_pages,
        v_pages,
        mask_stack[:num_blocks],
        q,
    ]
    dims: list[int | None] = [0, None, None, 0, None]
    if alibi_stack is not None:
        operands.append(alibi_stack[:num_blocks])
        dims.append(0)

    def fold_group(tile, reduce):
        """Reduce a per-page result over the group axis; identity at width 1."""
        return reduce(tile, dim=0, keepdim=True) if grouped else tile

    def block_body(carry, tiles):
        page_index, k_pages, v_pages, mask_tile, q, *rest = tiles
        # One gather for the whole group. The column slice runs inside the body, so it is
        # a real op rather than a view whose offset is dropped at the argument boundary
        # (torch-spyre#3770). index_select, not `k_pages[page_idx]`: subscripting lowers to
        # aten.index, which upcasts the int32 index to int64 and fails eager.
        page_idx = page_index[:, 0] if grouped else page_index[0, 0:1]
        k_page = k_pages.index_select(0, page_idx)
        v_page = v_pages.index_select(0, page_idx)
        if grouped:
            # Token-major pages to head-major, the group left as a leading batch axis:
            # [group, kv, 1, block_size, head_size]. Merging group x block_size into one
            # token axis would widen the score matmul, but that merge is not a view on
            # every cache layout and lowers to a strided copy the backend cannot address.
            k_page_4d = k_page.permute(0, 2, 1, 3).unsqueeze(2)
            v_page_4d = v_page.permute(0, 2, 1, 3).unsqueeze(2)
            # [group, 1, 1, padded_query_len, block_size], broadcast over the heads.
            mask = mask_tile.unsqueeze(1).unsqueeze(1)
        else:
            k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            mask = mask_tile[0]

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Pull logits into (-cap, +cap) before the mask add so masked
            # positions still map cleanly to -inf. Applied before the ALiBi
            # bias so the positional term is not squashed by the tanh.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if rest:
            # ALiBi bias slope[h] * (kv_pos - context_len). The additive
            # mask below uses finfo.min for masked positions, so this
            # bias cannot un-mask them.
            scores = scores + (rest[0] if grouped else rest[0][0])
        scores = scores + mask

        # One maximum for the whole group, over its pages as well as its keys, so the
        # group needs no rescale within itself -- as batched decode's chunk does not.
        scores_max = fold_group(torch.amax(scores, dim=-1, keepdim=True), torch.amax)

        # `carry is None` is required for SPYRE_ATTN_FOR_EACH_TILE=0
        if carry is None:
            tile_probs = torch.exp(scores - scores_max)
            return (
                scores_max,
                fold_group(tile_probs.sum(dim=-1, keepdim=True), torch.sum),
                fold_group(torch.matmul(tile_probs, v_page_4d), torch.sum),
            ), None

        tile_max, tile_sum, tile_output = carry
        new_max = torch.maximum(tile_max, scores_max)
        if tile_loop.USE_FOR_EACH_TILE:
            # Read tile_max before the maximum that supersedes it, or the tiled
            # lowering copies the whole carry every trip.
            rescale = torch.exp(-torch.relu(scores_max - tile_max))
        else:
            rescale = torch.exp(tile_max - new_max)
        tile_probs = torch.exp(scores - new_max)
        new_sum = tile_sum * rescale + fold_group(tile_probs.sum(dim=-1, keepdim=True), torch.sum)
        new_output = tile_output * rescale + fold_group(
            torch.matmul(tile_probs, v_page_4d), torch.sum
        )
        return (new_max, new_sum, new_output), None

    # The carry keeps the group axis the body reduces onto, at extent 1.
    group_axis = (1,) if grouped else ()
    state_shape = (*group_axis, num_kv_heads, num_queries_per_kv, padded_query_len, 1)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    (_, tile_sum, tile_output), _ = walk_tiles(
        block_body,
        tuple(operands),
        dims=tuple(dims),
        tile_size=page_group,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(
                (*group_axis, num_kv_heads, num_queries_per_kv, padded_query_len, head_size),
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
