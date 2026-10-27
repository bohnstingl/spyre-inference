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

"""MiniMax-M3 (text backbone) on Spyre.

Upstream's model (``vllm.models.minimax_m3``) is NVIDIA code: a FlashInfer Gemma norm,
a fused CUDA QK-norm + RoPE + KV-insert prologue, and a block-sparse attention layer with
its own Triton backend and index-key cache. Everything else is reused unchanged: the
decoder layer, the MoE block, the dense MLP, the loaders (including the fused
``[q | k | v | index_q | index_k]`` projection) and the all-reduce/norm fusion's
non-FlashInfer fallback. This module replaces only those three pieces:

* ``MiniMAXGemmaRMSNorm`` becomes vLLM's ``GemmaRMSNorm``, which Spyre overrides;
* the prologue is the same math in torch: per-head Gemma QK-norm and partial NeoX RoPE;
* attention is a plain vLLM ``Attention`` on the Spyre paged backend. The sparse layers
  keep the index projections and norms so the checkpoint loads as upstream loads it, but
  attend densely: the lightning indexer keeps the top 16 blocks of 128 tokens, so up to
  2,048 tokens of context this is the trained model exactly, and past it an
  approximation.

The MoE experts use the gpt-oss ``swigluoai`` recipe (the same math, gate and up already
in halves) and GLM's TEMPORARY CPU FALLBACK router (``glm_cpu_router``): sigmoid scores,
a selection-only bias, top-4 renormalized, computed in fp32 on the host. The routed scale
(x2.0) is folded into the down weights.
"""

from __future__ import annotations

import contextlib
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import MinimaxM3QKVParallelLinearWithIndexer
from vllm.models.minimax_m3.nvidia import model as m3

from spyre_inference.models.glm_cpu_router import capture_host_routers, install_host_router
from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _scale_down_weight(down_weight: torch.Tensor, scale: float) -> torch.Tensor:
    return down_weight * scale


class _SpyreMiniMaxM3Attention(m3.MiniMaxM3Attention):
    """Upstream's dense attention with the fused CUDA prologue written out in torch."""

    def _prologue(
        self, positions: torch.Tensor, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sizes = [self.q_size, self.kv_size, self.kv_size]
        # A sparse layer's projection also emits [index_q | index_k], unused here.
        if rest := qkv.shape[-1] - sum(sizes):
            sizes.append(rest)
        q, k, v = qkv.split(sizes, dim=-1)[:3]
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary_emb(positions, q.view(-1, self.q_size), k.view(-1, self.kv_size))
        return q, k, v

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self._prologue(positions, qkv)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class _SpyreMiniMaxM3SparseAttention(_SpyreMiniMaxM3Attention):
    """A sparse layer, attending densely; see the module docstring.

    Builds upstream's fused projection with the index heads (so the checkpoint's
    ``index_{q,k}_proj`` load into it) and the index norms, plus the dense layer's
    attention. The index columns of the projection are computed and unused.
    """

    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        cache_config=None,
        topk_indices_buffer=None,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config=quant_config, prefix=prefix, cache_config=cache_config
        )
        sparse_cfg = config.sparse_attention_config
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            sparse_cfg["sparse_num_index_heads"],
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)


@contextlib.contextmanager
def _spyre_layers():
    """Build upstream's model with the Spyre-runnable attention and norm classes.

    The NVIDIA module's attention and norm are plain ``nn.Module``s, not ``PluggableLayer``s,
    so they cannot be replaced through ``register_oot``; the decoder layer and model look them
    up as module globals at construction.
    """
    saved = (m3.MiniMAXGemmaRMSNorm, m3.MiniMaxM3Attention, m3.MiniMaxM3SparseAttention)
    m3.MiniMAXGemmaRMSNorm = GemmaRMSNorm
    m3.MiniMaxM3Attention = _SpyreMiniMaxM3Attention
    m3.MiniMaxM3SparseAttention = _SpyreMiniMaxM3SparseAttention
    try:
        yield
    finally:
        m3.MiniMAXGemmaRMSNorm, m3.MiniMaxM3Attention, m3.MiniMaxM3SparseAttention = saved


def configure_minimax_m3_moe_layers(layers: Iterable[nn.Module]) -> int:
    """Opt every MiniMax-M3 MoE block into the Spyre MoE backend; returns the count."""
    configured = 0
    for decoder in layers:
        moe = getattr(decoder, "block_sparse_moe", None)
        if not isinstance(moe, m3.MiniMaxM3MoE):
            continue
        # Built without apply_routed_scale_to_output, so the scale sits on the routed
        # experts, where upstream's router would apply it; fold it into the down stack.
        experts = moe.experts.routed_experts
        prepare_down_weight = None
        if experts.routed_scaling_factor != 1.0:
            prepare_down_weight = partial(_scale_down_weight, scale=experts.routed_scaling_factor)
            experts.routed_scaling_factor = 1.0
        configure_spyre_moe_layer(
            experts,
            SpyreMoERecipe(
                activation="swigluoai",
                routing="dense_routes",
                prepare_down_weight=prepare_down_weight,
            ),
        )
        install_host_router(experts, moe.gate)
        configured += 1
    if configured:
        logger.info("Spyre: configured %d MiniMax-M3 MoE layers.", configured)
    return configured


class SpyreMiniMaxM3SparseForCausalLM(m3.MiniMaxM3SparseForCausalLM):
    """MiniMax-M3's text backbone on Spyre."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        with _spyre_layers():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        layers = self.model.layers[self.model.start_layer : self.model.end_layer]
        moe_layers = sum(
            isinstance(getattr(d, "block_sparse_moe", None), m3.MiniMaxM3MoE) for d in layers
        )
        if configure_minimax_m3_moe_layers(layers) != moe_layers:
            raise RuntimeError("MiniMax-M3: not every MoE layer opted into the Spyre backend.")
        # Each layer leaves its FFN un-reduced for the next layer's norm, inside the next
        # block's graph. The final norm runs outside every block, where the eager all-reduce
        # feeding it does not lower ("broadcast read ... view covers 768 cores but op runs
        # 4"), so the last layer reduces its own FFN, as in every other model.
        if self.model.fuse_final_norm_allreduce:
            last = self.model.layers[self.model.end_layer - 1]
            if last.is_moe_layer:
                last.block_sparse_moe.experts.moe_config.skip_final_all_reduce = False
            else:
                last.mlp.down_proj.reduce_results = True
            self.model.fuse_final_norm_allreduce = False
        logger.warning_once(
            "MiniMax-M3 on Spyre attends densely in its sparse layers: exact up to 2,048 "
            "tokens of context (16 blocks of 128), an approximation of the trained "
            "block-sparse attention beyond."
        )

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # Before process_weights_after_loading transposes and moves the gate weights.
        capture_host_routers()
        return loaded
