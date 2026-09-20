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

"""Batched multi-sequence decode, behind ``SPYRE_BATCHED_DECODE``."""

import torch

from spyre_inference.v1.attention.ops.tile_loop import walk_tiles


def batched_decode_kernel(
    query,
    rep_row_ids,
    k_pages,
    v_pages,
    chunk_page_ids,
    mask_by_chunk,
    scale,
    num_seqs,
    blocks_per_chunk,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Batched decode kernel; gathers K/V and the query in-graph.

    Gathers blocks_per_chunk blocks per sequence per step, so the gather's entry
    axis is entries = num_seqs * blocks_per_chunk. A gather is core-split only on
    that axis, and behind a 1-D index it is counted in whole 32-entry sticks, so
    a narrow 1-D gather has no splittable unit and runs on one core.

    k/v_pages: [num_pages_total, block_size, KV, D] (the raw page cache).
    chunk_page_ids: [padded_blocks, num_seqs] int32, row b holding each sequence's
    b-th active page. mask_by_chunk: [padded_blocks, num_seqs, KV or 1, 1,
    block_size]; both walks tile blocks_per_chunk-wide chunks off those logical
    block axes, and the trailing extent-1 axes broadcast over the query groups in
    the mask add. rep_row_ids: [entries] int32, all query rows repeated as one
    block-major group per block slot. ``out`` None returns the result instead of
    storing it.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    entries = num_seqs * blocks_per_chunk
    q = query.index_select(0, rep_row_ids).reshape(
        blocks_per_chunk,
        num_seqs,
        num_kv_heads,
        num_queries_per_kv,
        head_size,
    )

    def chunk_body(carry, tiles):
        page_ids, mask_rows, k_pages, v_pages, q = tiles
        # Keep the real [block-slot, sequence] tile through the indirect read.
        # Flattening it first forces an unsupported int32 staging layout.
        # Token-major cache page to head-major; a view, so do not add
        # .contiguous() -- merging these axes is what materializes the page.
        k_page = (
            k_pages[page_ids]
            .reshape(entries, block_size, num_kv_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        v_page = (
            v_pages[page_ids]
            .reshape(entries, block_size, num_kv_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        scores = (
            torch.matmul(
                q.reshape(entries, num_kv_heads, num_queries_per_kv, head_size),
                k_page.transpose(-2, -1),
            )
            * scale
        )
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so
            # capping after it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        # Restore the tile coordinates before adding the mask. Keeping the mask
        # in [K, B, KV, S] avoids flattening its advancing read window.
        sc = scores.reshape(
            blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, block_size
        )
        sc = sc + mask_rows
        chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=0, keepdim=True)

        # `carry is None` is required for SPYRE_ATTN_FOR_EACH_TILE=1
        if carry is None:
            new_max = chunk_max
        else:
            tile_max, tile_sum, tile_output = carry
            rescale = torch.exp(-torch.relu(chunk_max - tile_max))
            new_max = torch.maximum(tile_max, chunk_max)
        probs = torch.exp(sc - new_max)
        # The chunk's slots share one max, so summing them needs no rescale.
        chunk_sum = torch.sum(torch.sum(probs, dim=-1, keepdim=True), dim=0, keepdim=True)
        chunk_out = torch.sum(
            torch.matmul(
                probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size),
                v_page,
            ).reshape(blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, head_size),
            dim=0,
            keepdim=True,
        )

        # `carry is None` is required for SPYRE_ATTN_FOR_EACH_TILE=1
        if carry is None:
            return (new_max, chunk_sum, chunk_out), None

        return (
            new_max,
            tile_sum * rescale + chunk_sum,
            tile_output * rescale + chunk_out,
        ), None

    state_shape = (1, num_seqs, num_kv_heads, num_queries_per_kv, 1)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    (_, tile_sum, tile_output), _ = walk_tiles(
        chunk_body,
        (chunk_page_ids, mask_by_chunk, k_pages, v_pages, q),
        dims=(0, 0, None, None, None),
        tile_size=blocks_per_chunk,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(
                (1, num_seqs, num_kv_heads, num_queries_per_kv, head_size),
                **state_kwargs,
            ),
        ),
    )
    attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
    if out is not None:
        # The destination prefix starts at offset 0, so torch-spyre#3770 does not
        # apply; rows past the batch are don't-care and kept finite by the builder.
        out[:num_seqs].copy_(attn)
        return out
    return attn
