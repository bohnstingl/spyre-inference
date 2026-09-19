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

"""Batched multi-sequence decode over a head-major KV cache.

The reduction is ``batched_decode``'s, chunk for chunk; only the page read differs.
One index row per page, as token-major, but this layout stores the page head-major
already, so the permute token-major does per chunk disappears.

Not the folded ``(page, kv_head)`` rows the per-sequence kernel gathers: the two move the
same bytes, but folding costs ``num_kv_heads`` times the index entries, and gather time
scales with entries rather than bytes -- measured at ~2x the kernel time for 8 kv heads.
"""

import torch

from spyre_inference.v1.attention.ops.tile_loop import walk_tiles


def batched_decode_head_major_kernel(
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
    """Shapes as in ``batched_decode_kernel``, except k/v_pages are the unfolded
    head-major cache, [num_pages_total, num_kv_heads, block_size, head_size].

    chunk_page_ids and mask_by_chunk keep the same logical block-major
    layout as ``batched_decode_kernel``; each walk tile is one chunk.

    One index row per page rather than per (page, kv_head): the gather then splits on the
    axis that stays the matmul's batch dim 0, as token-major's does, and the page still
    arrives head-major so there is no permute either.
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
        # Preserve the real [block-slot, sequence] tile through the indirect
        # page read; flattening the int32 tile first requires an unsupported
        # staging layout.
        k_page = k_pages[page_ids].reshape(entries, num_kv_heads, block_size, head_size)
        v_page = v_pages[page_ids].reshape(entries, num_kv_heads, block_size, head_size)
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
        sc = scores.reshape(
            blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, block_size
        )
        sc = sc + mask_rows
        chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=0, keepdim=True)

        # The running max drives exp(), not the chunk's own: a chunk wholly past a
        # sequence's length is -inf throughout and exp(-inf - -inf) is NaN.
        tile_max, tile_sum, tile_output = carry
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

        rescale = torch.exp(tile_max - new_max)
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
        use_init_in_fallback=True,
    )
    attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
    if out is not None:
        # Offset 0, so torch-spyre#3770 does not apply; rows past the batch are
        # don't-care and kept finite by the builder.
        out[:num_seqs].copy_(attn)
        return out
    return attn
