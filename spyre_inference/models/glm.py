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

"""Spyre adaptations for vLLM's GLM MoE models with DeepSeek-style attention.

GLM-4.7-Flash (``Glm4MoeLiteForCausalLM``) and GLM-5.x (``GlmMoeDsaForCausalLM``) share
DeepSeek-V2's MLA attention and a noaux_tc router: sigmoid scores, a selection-only
correction bias, renormalized top-k and a routed scale.
"""

from __future__ import annotations

import types
from functools import partial
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2MoE,
    GlmMoeDsaForCausalLM,
)
from vllm.model_executor.models.glm4_moe import Glm4MoE
from vllm.model_executor.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer

from spyre_inference.models.glm_cpu_router import capture_host_routers, install_host_router
from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _scale_down_weight(down_weight: torch.Tensor, scale: float) -> torch.Tensor:
    return down_weight * scale


def _spyre_mha_forward(
    self: DeepseekV2Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
) -> torch.Tensor:
    """``DeepseekV2Attention.forward`` with K and Q assembled by ``torch.cat``.

    Upstream writes both halves into ``torch.empty_like(q)``, which Inductor lowers to a
    NaN-filled buffer Spyre cannot store ("store value of unexpected type Constant").
    """
    if self.q_lora_rank is not None:
        q = self.q_a_proj(hidden_states)[0]
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
    else:
        q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
    kv_a, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_a = self.kv_a_layernorm(kv_a)
    kv = self.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))

    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, self.num_local_heads, -1)], dim=-1)
    if llama_4_scaling is not None:
        q = q * llama_4_scaling

    if self.v_head_dim != self.qk_head_dim:
        v = torch.nn.functional.pad(v, [0, self.qk_head_dim - self.v_head_dim], value=0)
    attn_output = self.attn(q, k, v.reshape(-1, self.num_local_heads * self.qk_head_dim))
    attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
        ..., : self.v_head_dim
    ].reshape(-1, self.num_local_heads * self.v_head_dim)
    output, _ = self.o_proj(attn_output)
    return output


def patch_mha_attention(layers: Iterable[nn.Module]) -> None:
    """Bind ``_spyre_mha_forward`` onto the MLA-disabled attention of each layer."""
    for decoder in layers:
        attn = getattr(decoder, "self_attn", None)
        if isinstance(attn, DeepseekV2Attention):
            attn.forward = types.MethodType(_spyre_mha_forward, attn)


def configure_glm_moe_layers(
    layers: Iterable[nn.Module], *, fold_routed_scale: bool, host_router: bool = True
) -> int:
    """Opt every GLM MoE layer into the Spyre MoE backend; returns the count.

    ``host_router`` (default) selects the TEMPORARY CPU FALLBACK in ``glm_cpu_router``:
    the router runs on the host in fp32 and the experts take its routing table
    (``dense_routes``). Without it the router runs on Spyre in fp16 (``sigmoid_bias``),
    which picks a different expert set for about a third of GLM-5.2's real tokens.

    ``fold_routed_scale`` moves the routed scale from MoERunner into the down stack. In
    fp16 the runner divides the shared expert's output by the scale instead of scaling
    the routed one, which is only right when the decoder layer divides the residual
    stream to match: DeepseekV2DecoderLayer (GLM-5.x) does, Glm4MoeLiteDecoderLayer
    (GLM-4.7-Flash) does not.
    """
    configured = 0
    for decoder in layers:
        mlp = getattr(decoder, "mlp", None)
        if not isinstance(mlp, (Glm4MoE, DeepseekV2MoE)):
            continue
        runner = mlp.experts
        prepare_down_weight = None
        if fold_routed_scale and runner.routed_scaling_factor != 1.0:
            prepare_down_weight = partial(_scale_down_weight, scale=runner.routed_scaling_factor)
            runner.routed_scaling_factor = 1.0
        experts = runner.routed_experts
        configure_spyre_moe_layer(
            experts,
            SpyreMoERecipe(
                activation="silu",
                routing="dense_routes" if host_router else "sigmoid_bias",
                prepare_down_weight=prepare_down_weight,
            ),
        )
        if host_router:
            install_host_router(experts.layer_name, experts, mlp.gate)
        configured += 1
    if configured:
        logger.info("Spyre: configured %d GLM MoE layers.", configured)
    return configured


def compensate_fp16_residual_scale(model: nn.Module, scale: float) -> None:
    """Undo the epsilon shift of DeepseekV2DecoderLayer's fp16 overflow guard.

    In fp16 the decoder divides the residual stream by ``routed_scaling_factor`` and
    relies on RMSNorm being scale invariant. Up to epsilon it is: ``norm(x / s)`` with
    ``eps`` is ``norm(x)`` with ``eps * s**2``. GLM-5.2's early residual stream has a
    mean square near 5e-5, the same order as ``1e-5 * 2.5**2``, which moved its logits
    by about 1.6 nats against HF. Every norm reading the scaled stream gets ``eps / s**2``;
    layer 0's input norm reads the embedding before the guard scales it.
    """
    if scale == 1.0:
        return
    norms = [model.norm]
    for layer in model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        norms.append(layer.post_attention_layernorm)
        if layer.layer_idx != 0:
            norms.append(layer.input_layernorm)
    for norm in norms:
        if not isinstance(norm, PPMissingLayer):
            norm.variance_epsilon /= scale * scale


class SpyreGlm4MoeLiteForCausalLM(Glm4MoeLiteForCausalLM):
    """GLM-4.7-Flash on Spyre with Spyre MoE expert dispatch."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        patch_mha_attention(self.model.layers)
        configure_glm_moe_layers(self.model.layers, fold_routed_scale=True)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # Before process_weights_after_loading transposes and moves the gate weights.
        capture_host_routers()
        return loaded


class SpyreGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    """GLM-5.x on Spyre with Spyre MoE expert dispatch and Spyre MLA + DSA attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        patch_mha_attention(self.model.layers)
        configure_glm_moe_layers(self.model.layers, fold_routed_scale=False)
        if vllm_config.model_config.dtype == torch.float16:
            compensate_fp16_residual_scale(self.model, self.config.routed_scaling_factor)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # The loader runs process_weights_after_loading after load_weights returns, so
        # the indexer split is taken from the [out, in] checkpoint weight here.
        from spyre_inference.v1.attention.dsa import SpyreDsaIndexer

        for module in self.modules():
            if isinstance(module, SpyreDsaIndexer):
                module.split_projections()
        capture_host_routers()
        return loaded
