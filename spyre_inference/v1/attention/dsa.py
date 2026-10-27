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

"""DeepSeek sparse attention (DSA, as GLM-5.x runs it) on Spyre's latent MLA attention.

DSA's lightning indexer scores every cached token per query,
``sum_h w_h * relu(q_h . k)``, and the query attends only to its ``index_topk`` best.
Upstream runs that as fp8 DeepGEMM logits and a CUDA top-k feeding a sparse MLA kernel.
Here the selection is an additive mask on the dense paged walk, the "masked MHA" form
DeepSeek uses for short prefill:

- ``SpyreDsaIndexer`` runs upstream's ``Indexer`` projections (``wq_b``, the fused
  ``wk_weights_proj``, ``k_norm``) and its RoPE, so the checkpoint loads as upstream
  loads it, minus the fp8 quantization and the DeepGEMM op.
- The indexer key is stored in the latent cache row after ``[kv_c | k_pe]``, so the
  existing KV store writes it and no second cache, spec or metadata group is needed. The
  attention query is zero in those columns, so they add nothing to the attention scores.
- ``SpyreDsaAttentionImpl`` builds each sequence's top-k mask from the cache
  (``ops.dsa_head_major``) and hands it to the unchanged head-major attention kernels.
  A ``"shared"`` layer (GLM-5's IndexCache) reuses the mask of the full layer before it,
  as upstream reuses that layer's top-k indices.

A walk of at most ``index_topk`` positions keeps every token, which is the dense model,
so it skips the indexer scoring altogether.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert, row_outermost_layout
from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
    SpyreHeadMajorAttentionBackend,
    SpyreHeadMajorAttentionImpl,
)
from spyre_inference.v1.attention.ops import tile_loop
from spyre_inference.v1.attention.ops.dsa_head_major import dsa_mask_head_major_kernel
from spyre_inference.v1.worker import compile_guard

if TYPE_CHECKING:
    from torch import nn
    from vllm.model_executor.models.deepseek_v2 import Indexer

    from spyre_inference.v1.attention.backends.spyre_attn import (
        SpyreAttentionMetadata,
    )

logger = init_logger(__name__)

_dsa_mask_compiled = torch.compile(
    dsa_mask_head_major_kernel, dynamic=False, fullgraph=tile_loop.USE_FOR_EACH_TILE
)
compile_guard.watch(dsa_mask_head_major_kernel, "DSA top-k mask kernel (head-major)")

# Each step halves a row's score range; 16 leave only fp16 near-ties at the k-th score.
_BISECT_STEPS = 16
_STICK = 64


@dataclass(frozen=True)
class SpyreDsaConfig:
    index_topk: int
    n_heads: int
    head_dim: int
    rope_dim: int
    # Column of the latent row where the indexer key starts.
    key_offset: int

    @classmethod
    def from_hf(cls, config, key_offset: int) -> SpyreDsaConfig:
        return cls(
            index_topk=config.index_topk,
            n_heads=config.index_n_heads,
            head_dim=config.index_head_dim,
            rope_dim=config.qk_rope_head_dim,
            key_offset=key_offset,
        )


class DsaStaging:
    """Constant-shaped buffers the block graph writes a step's index queries and weights to.

    One set serves every layer: each full layer writes them and its attention reads them
    before the next layer runs. Rows are absolute token rows, as in the query staging.
    """

    def __init__(self, rows: int, n_heads: int, head_dim: int, dtype: torch.dtype):
        # Head weights padded to a whole stick; the zero lanes weigh nothing.
        self.shapes = ((rows, n_heads, head_dim), (rows, -(-n_heads // _STICK) * _STICK))
        self.dtype = dtype
        self._buffers: tuple[torch.Tensor, torch.Tensor] | None = None

    @staticmethod
    @cache
    def shared(rows: int, n_heads: int, head_dim: int, dtype: torch.dtype) -> DsaStaging:
        return DsaStaging(rows, n_heads, head_dim, dtype)

    def buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._buffers is None:
            q, w = (
                convert(
                    torch.zeros(shape, dtype=self.dtype),
                    device,
                    device_layout=(
                        row_outermost_layout(shape, self.dtype) if device.type == "spyre" else None
                    ),
                )
                for shape in self.shapes
            )
            self._buffers = (q, w)
        return self._buffers


class SpyreDsaIndexer(torch.nn.Module):
    """Upstream's indexer projections: the key to cache, and the staged query and weights."""

    def __init__(
        self, indexer: Indexer, rope: nn.Module, config: SpyreDsaConfig, staging: DsaStaging
    ):
        super().__init__()
        self.indexer = indexer
        self.rope = rope
        self.config = config
        self.staging = staging
        self.register_buffer("wk_t", None, persistent=False)
        self.register_buffer("weights_t", None, persistent=False)
        # Upstream's indexer key cache is a layer of its own, with a DeepGEMM backend and an
        # fp8 page; the key lives in the latent row instead, so it must not get a KV cache.
        from vllm.config import get_current_vllm_config

        static_ctx = get_current_vllm_config().compilation_config.static_forward_context
        static_ctx.pop(indexer.k_cache.prefix, None)
        # An fp32 LayerNorm meets fp16 activations; fp16 lowers ("Multi-arg pointwise").
        dtype = torch.get_default_dtype()
        for param in indexer.k_norm.parameters():
            param.data = param.data.to(dtype)

    def split_projections(self) -> None:
        """Split the fused ``[wk | weights_proj]`` weight once it is loaded.

        Its output puts the head weights half a stick into a row, which no store lowers
        from. ``wk`` is kept alone, ``weights_proj`` zero-padded to a stick with the head
        scale folded in. Called from ``SpyreLatentAttention.process_weights_after_loading``,
        which runs after every linear layer's: the fused weight is then Wᵀ [in, out].
        """
        layer = self.indexer.wk_weights_proj
        cfg = self.config
        w_t = layer.weight.data
        assert w_t.shape == (layer.input_size, cfg.head_dim + cfg.n_heads), tuple(w_t.shape)
        wk_t, wp_t = w_t.split([cfg.head_dim, cfg.n_heads], dim=-1)
        wp_t = torch.nn.functional.pad(
            wp_t * cfg.n_heads**-0.5, (0, self.staging.shapes[1][1] - cfg.n_heads)
        )
        self.wk_t = wk_t.contiguous()
        self.weights_t = wp_t.contiguous()

    def forward(
        self, hidden_states: torch.Tensor, q_c: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``Indexer.forward`` without the fp8 quantization and the DeepGEMM op."""
        cfg, indexer = self.config, self.indexer
        q = indexer.wq_b(q_c)[0].view(-1, cfg.n_heads, cfg.head_dim)
        q_pe, q_nope = q.split([cfg.rope_dim, cfg.head_dim - cfg.rope_dim], dim=-1)
        assert self.wk_t is not None and self.weights_t is not None, "split_projections() not run"
        k = indexer.k_norm(torch.matmul(hidden_states, self.wk_t))
        weights = torch.matmul(hidden_states, self.weights_t)
        k_pe, k_nope = k.split([cfg.rope_dim, cfg.head_dim - cfg.rope_dim], dim=-1)
        q_pe, k_pe = self.rope(positions, q_pe, k_pe.unsqueeze(1))
        # Upstream folds both scales into the weights; relu commutes with a positive scale.
        q = torch.cat([q_pe.reshape(-1, cfg.n_heads, cfg.rope_dim), q_nope], dim=-1)
        q = q * cfg.head_dim**-0.5
        k = torch.cat([k_pe.reshape(-1, cfg.rope_dim), k_nope], dim=-1)

        q_buf, w_buf = self.staging.buffers(q.device)
        rows = q.shape[0]
        q_buf[:rows] = q
        w_buf[:rows] = weights
        return k, [q_buf, w_buf]


class SpyreDsaAttentionImpl(SpyreHeadMajorAttentionImpl):
    """Head-major paged attention under the DSA top-k mask.

    ``dsa_staging`` is set on layers with an indexer, which compute the mask; a layer
    without one reuses the mask the last full layer stored on the step's metadata.
    """

    def __init__(
        self, *args, dsa: SpyreDsaConfig, dsa_staging: DsaStaging | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.kv_cache_fp8:
            raise NotImplementedError("Spyre DSA does not support an fp8 KV cache.")
        self.dsa = dsa
        self.dsa_staging = dsa_staging
        self._call: tuple[SpyreAttentionMetadata, tuple[torch.Tensor, ...] | None] | None = None
        self._recording = False

    def _batched_decode_supported(self) -> bool:
        # The batched kernel builds its mask on the host; decode takes the per-sequence walk.
        return False

    def record_graphs(self, layer, kv_cache, builder) -> int:
        # Warmup records each layer alone, so a shared layer has no full layer's mask.
        self._recording = True
        try:
            return super().record_graphs(layer, kv_cache, builder)
        finally:
            self._recording = False

    def staging_buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Allocated with the query staging, before any trace (see allocate_staging_buffers).
        if self.dsa_staging is not None:
            self.dsa_staging.buffers(device)
        return super().staging_buffers(device)

    def forward(  # ty: ignore[invalid-method-override]
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
        extra_inputs: list[torch.Tensor] | None = None,
    ):
        if attn_metadata is None:
            return output
        index_inputs = None
        if self.dsa_staging is not None:
            # Warmup's recorder calls without them; the staging buffers stand in.
            index_inputs = (
                tuple(extra_inputs)
                if extra_inputs
                else self.dsa_staging.buffers(kv_cache[0].device)
            )
        self._call = (attn_metadata, index_inputs)
        try:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        finally:
            self._call = None

    def _run_page_attn(
        self,
        query: torch.Tensor,
        row_table: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        index_table: torch.Tensor,
        mask_stack: torch.Tensor,
        num_blocks: int,
        padded_query_len: int,
        alibi_stack: torch.Tensor | None,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self._call is not None
        attn_metadata, index_inputs = self._call
        # A walk of at most index_topk positions keeps every valid one: the dense mask.
        if num_blocks * self.block_size <= self.dsa.index_topk:
            pass
        elif index_inputs is not None:
            index_q, index_w = index_inputs
            k_folded, _ = self._folded_pages(k_pages, v_pages)
            mask_stack = _dsa_mask_compiled(
                index_q,
                index_w,
                row_table,
                k_folded,
                index_table,
                mask_stack,
                num_blocks,
                padded_query_len,
                self.dsa.key_offset,
                self.dsa.n_heads,
                self.dsa.index_topk,
                _BISECT_STEPS,
            )
            # Keyed by the sequence's row table, which the step builds once on the
            # metadata every layer of the group shares.
            attn_metadata.dsa_mask_stacks[id(row_table)] = mask_stack
        elif (shared := attn_metadata.dsa_mask_stacks.get(id(row_table))) is not None:
            mask_stack = shared
        elif not self._recording:
            raise RuntimeError(
                "DSA: a layer sharing the top-k selection ran before any layer that "
                "computes it, so it would attend densely."
            )
        return super()._run_page_attn(
            query,
            row_table,
            k_pages,
            v_pages,
            index_table,
            mask_stack,
            num_blocks,
            padded_query_len,
            alibi_stack,
            out,
        )


class SpyreDsaAttentionBackend(SpyreHeadMajorAttentionBackend):
    """Head-major backend whose impl applies the DSA mask; selected by the MLA layer."""

    @staticmethod
    def get_impl_cls() -> type[SpyreDsaAttentionImpl]:
        return SpyreDsaAttentionImpl
