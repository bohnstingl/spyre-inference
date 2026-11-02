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

"""Muse Spark Open (text backbone) on Spyre.

Upstream's model (``vllm.models.muse_spark``) is NVIDIA or ROCm code throughout: FlashInfer
and Triton norms, CUDA events and aux streams, a fused QK-norm + RoPE + KV-insert, and
TRT-LLM / AITER experts with their own finalize. This module is the same text model written
for Spyre, against the HF checkpoint (``MuseSparkForConditionalGeneration``); the vision and
audio towers are not built, so it serves text only.

* Norms are vLLM's: ``GemmaRMSNorm`` for the ``1 + w`` norms, ``RMSNorm`` for the rest, and
  ``_InputScaledRMSNorm`` (``rms(x * (1 + w))``) after the FFN and after each expert.
* Attention: QK-norm before NeoX RoPE on the checkpoint's own Q/K layout (HF rotates it as
  stored), ``qk_scale_factor`` folded into the softmax scale, per-head o-norm and a sigmoid
  gate. Every fourth layer is full attention without RoPE, the rest slide over 2,048 tokens.
* The residual stream is fp16 (fp32 on GPUs). Its blends, ``alpha * stream + beta * branch``,
  take alpha and beta precomputed from the learned gates after loading.
* MoE: a pooler squeezes each token to 4,096, 256 experts (top 8) run there, and every
  routed expert's output gets its own RMSNorm before the weighted sum. The experts are
  split over ranks whole (expert parallel inside the TP group), so that norm sees full rows
  and needs no all-reduce of its own. Each rank runs its local experts densely over all
  tokens, weighting the unselected ones by zero, then the partial sums are all-reduced.
* TEMPORARY CPU FALLBACK: the router (fp32 GEMM, sigmoid, a selection-only bias, top-8,
  renormalized) runs on the host, as GLM's does (``glm_cpu_router``): the model routes in
  fp32, and Spyre's matmul accumulates in fp16 only.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from itertools import islice
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import extract_layer_index, make_layers, maybe_prefix
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from spyre_inference.custom_ops.utils import convert

if TYPE_CHECKING:
    from vllm.config import CacheConfig, VllmConfig
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

# The routing table goes to the card one stick wide per (expert, token); the experts read
# it back as its max over the stick, as a [E, T, 1] tensor does not lower.
_ROUTE_STICK = 64
_TEXT_PREFIX = "model.language_model."


class _InputScaledRMSNorm(nn.Module):
    """``rms(x * (1 + w))``: the weight scales the input, the norm has none of its own."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.norm = RMSNorm(hidden_size, eps=eps, has_weight=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x * (1.0 + self.weight))


class _ResidualGate(nn.Module):
    """Norm-preserving blend ``alpha * stream + beta * branch`` with per-channel gates.

    ``beta = sigmoid(w / T)`` and ``alpha = sqrt(max(eps, 1 - beta^2))``, computed once in
    fp32 by ``materialize`` after loading; the blend runs in the stream's dtype, as an fp32
    one does not lower.
    """

    def __init__(self, hidden_size: int, temperature: float, eps: float = 1e-3) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.temperature = temperature
        self.eps = eps

    def materialize(self) -> None:
        w = self.gate.detach().float() / self.temperature
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=self.eps))
        dtype = self.gate.dtype
        self.register_buffer("alpha", alpha.to(dtype), persistent=False)
        self.register_buffer("beta", beta.to(dtype), persistent=False)

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return self.alpha * stream + self.beta * branch


class _HostRouter:
    """TEMPORARY CPU FALLBACK: one MoE layer's router in fp32 on the host.

    Returns this rank's slice of the routing table, ``[local experts, tokens, stick]``.
    """

    def __init__(self, num_experts: int, top_k: int, first: int, count: int, config) -> None:
        self.top_k = top_k
        self.first, self.count = first, count
        self.multiplier = config.output_multiplier
        self.route_scale = config.routed_scaling_factor
        self.weight_t = torch.empty(0)
        self.bias = torch.zeros(num_experts)

    def load_weight(self, weight: torch.Tensor) -> None:
        self.weight_t = weight.to("cpu", torch.float32).t().contiguous()

    def load_bias(self, bias: torch.Tensor) -> None:
        self.bias = bias.to("cpu", torch.float32).clone()

    def dense(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[tokens, experts]`` routing weights in fp32, zero off the top-k."""
        scores = torch.sigmoid((hidden @ self.weight_t) * self.multiplier)
        _, ids = torch.topk(scores + self.bias, self.top_k, dim=-1)
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-15) * self.route_scale
        return torch.zeros_like(scores).scatter(1, ids, weights)

    def route(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # .cpu() before the upcast: an fp32 tensor read back from Spyre may be staggered.
        dense = self.dense(hidden_states.cpu().float())
        local = dense[:, self.first : self.first + self.count].t()
        return local.to(hidden_states.dtype).unsqueeze(-1).expand(-1, -1, _ROUTE_STICK)


# Keyed by MoE layer prefix; registered at construction, filled while the weights load.
_HOST_ROUTERS: dict[str, _HostRouter] = {}


def _host_route(
    hidden_states: torch.Tensor, layer_name: LayerNameType, num_local: int
) -> torch.Tensor:
    table = _HOST_ROUTERS[_resolve_layer_name(layer_name)].route(hidden_states)
    return convert(table.contiguous(), device=hidden_states.device)


def _host_route_fake(
    hidden_states: torch.Tensor, layer_name: LayerNameType, num_local: int
) -> torch.Tensor:
    return hidden_states.new_empty((num_local, hidden_states.shape[0], _ROUTE_STICK))


@lru_cache(maxsize=1)
def _register_host_route_op() -> None:
    from vllm.platforms import current_platform

    direct_register_custom_op(
        op_name="spyre_muse_spark_host_route",
        op_func=_host_route,
        fake_impl=_host_route_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def local_experts(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    routes: torch.Tensor,
) -> torch.Tensor:
    """This rank's experts over every token, each output normalized, then weighted and summed.

    ``w13`` is ``[E, H, 2I]`` (gate, then up), ``w2`` ``[E, I, H]``, ``routes`` the host's
    ``[E, T, stick]`` table; the result is this rank's partial sum, ``[T, H]``.
    """
    h = torch.matmul(x.unsqueeze(0).expand(w13.shape[0], -1, -1), w13)
    gate, up = h.split(w2.shape[1], dim=-1)
    y = (torch.matmul(F.silu(gate) * up, w2) * (1.0 + norm_weight)).float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y * routes.float().amax(-1, keepdim=True)).sum(0).to(x.dtype)


class SpyreMuseSparkMoE(nn.Module):
    """Pooler, host-routed local experts with their post-expert norm, all-reduce, unpooler."""

    def __init__(self, config, prefix: str) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        num_experts = config.num_local_experts
        if num_experts % tp_size:
            raise NotImplementedError(
                f"Muse Spark on Spyre splits its {num_experts} experts over the TP ranks "
                f"whole; tensor_parallel_size={tp_size} does not divide them."
            )
        hidden, inter = config.moe_hidden_size, config.moe_intermediate_size
        self.tp_size = tp_size
        self.num_local = num_experts // tp_size
        self.first = tp_rank * self.num_local
        self.post_norm_eps = config.post_norm_eps
        self.pooler = ReplicatedLinear(
            config.hidden_size, hidden, bias=False, prefix=f"{prefix}.pre_expert_proj"
        )
        self.pre_expert_norm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.w13 = nn.Parameter(torch.empty(self.num_local, hidden, 2 * inter), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(self.num_local, inter, hidden), requires_grad=False)
        self.post_expert_norm_weight = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self.unpooler = ReplicatedLinear(
            hidden, config.hidden_size, bias=False, prefix=f"{prefix}.post_expert_proj"
        )
        self.router = _HostRouter(
            num_experts, config.num_experts_per_tok, self.first, self.num_local, config
        )
        _register_host_route_op()
        _HOST_ROUTERS[prefix] = self.router
        self.route_name = prefix

    def load_expert_weight(self, name: str, loaded: torch.Tensor) -> None:
        """Copy this rank's experts out of a checkpoint stack, transposed for ``x @ w``."""
        target = self.w13 if name == "gate_up_proj" else self.w2
        for i in range(self.num_local):
            target.data[i].copy_(loaded[self.first + i].t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routes = torch.ops.vllm.spyre_muse_spark_host_route(
            x,  # ty: ignore[invalid-argument-type]
            _encode_layer_name(self.route_name),
            self.num_local,
        )
        pooled, _ = self.pooler(x)
        pooled = self.pre_expert_norm(pooled)
        partial = local_experts(
            pooled, self.w13, self.w2, self.post_expert_norm_weight, self.post_norm_eps, routes
        )
        if self.tp_size > 1:
            partial = tensor_model_parallel_all_reduce(partial)
        out, _ = self.unpooler(partial)
        return out


class SpyreMuseSparkAttention(nn.Module):
    def __init__(
        self,
        config,
        max_position: int,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.gate_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )
        self.qk_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
        self.o_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
        theta = config.layer_rope_theta[layer_idx]
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=max_position,
                is_neox_style=True,
                rope_parameters={"rope_theta": theta, "rope_type": "default"},
            )
            if theta
            else None
        )
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            # The query's post-norm scale, exact before RoPE since RoPE is a rotation.
            self.head_dim**-0.5 * config.qk_scale_factor,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            per_layer_sliding_window=config.sliding_window if sliding else None,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.qk_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.q_size)
        k = self.qk_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.kv_size)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)
        out = self.o_norm(out.view(-1, self.num_heads, self.head_dim)).view(-1, self.q_size)
        gate, _ = self.gate_proj(hidden_states)
        out, _ = self.o_proj(out * torch.sigmoid(gate))
        return out


class SpyreMuseSparkDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        if config.mlp_layer_types[extract_layer_index(prefix)] != "sparse":
            raise NotImplementedError("Muse Spark on Spyre implements MoE layers only.")
        self.self_attn = SpyreMuseSparkAttention(
            config,
            vllm_config.model_config.max_model_len,
            vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = SpyreMuseSparkMoE(config, prefix=f"{prefix}.mlp")
        eps, post_eps = config.rms_norm_eps, config.post_norm_eps
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=post_eps, has_weight=False)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = _InputScaledRMSNorm(config.hidden_size, eps=post_eps)
        temperature = config.residual_gate_temperature
        self.post_attention_residual_gate = _ResidualGate(config.hidden_size, temperature)
        self.post_feedforward_residual_gate = _ResidualGate(config.hidden_size, temperature)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        branch = self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states = self.post_attention_residual_gate(
            hidden_states, self.post_attention_layernorm(branch)
        )
        branch = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return self.post_feedforward_residual_gate(
            hidden_states, self.post_feedforward_layernorm(branch)
        )


class SpyreMuseSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, has_weight=False)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SpyreMuseSparkDecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class SpyreMuseSparkForCausalLM(nn.Module):
    """Muse Spark Open's text backbone on Spyre; see the module docstring."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("Muse Spark on Spyre does not support pipeline parallel.")
        if vllm_config.quant_config is not None:
            raise NotImplementedError("Muse Spark on Spyre runs the BF16 checkpoint only.")
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = SpyreMuseSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.output_multiplier = config.output_multiplier
        self.final_logit_softcapping = config.final_logit_softcapping
        logger.warning_once(
            "TEMPORARY CPU FALLBACK: Muse Spark MoE routing (fp32 router GEMM, sigmoid, "
            "bias, top-8) runs on the host for every MoE layer; the experts stay on Spyre."
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        logits = logits.float() * self.output_multiplier
        cap = self.final_logit_softcapping
        return cap * torch.tanh(logits / cap) if cap else logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        layers = self.model.layers
        loaded: set[str] = set()
        for name, weight in weights:
            if name == "lm_head.weight":
                pass
            elif name.startswith(_TEXT_PREFIX):
                name = "model." + name[len(_TEXT_PREFIX) :]
            else:
                continue  # the vision and audio towers
            parts = name.split(".")
            if parts[1] == "layers":
                layer = int(parts[2])
                if not self.model.start_layer <= layer < self.model.end_layer:
                    continue
                moe = layers[layer].mlp
                rest = ".".join(parts[3:])
                if rest == "mlp.gate.weight":
                    moe.router.load_weight(weight)
                    continue
                if rest == "mlp.gate.e_score_correction_bias":
                    moe.router.load_bias(weight)
                    continue
                if rest in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                    moe.load_expert_weight(parts[-1], weight)
                    loaded.add(
                        f"model.layers.{layer}.mlp." + ("w13" if "gate_up" in rest else "w2")
                    )
                    continue
                name = _remap_layer_weight(name, rest)
            stacked = _QKV_SHARDS.get(name.rsplit(".", 2)[-2])
            if stacked is not None:
                param_name = name.replace(f"self_attn.{stacked[0]}", "self_attn.qkv_proj")
                param = params[param_name]
                param.weight_loader(param, weight, stacked[1])
                loaded.add(param_name)
                continue
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded

    def process_weights_after_loading(self) -> None:
        for module in self.modules():
            if isinstance(module, _ResidualGate):
                module.materialize()


_QKV_SHARDS = {"q_proj": ("q_proj", "q"), "k_proj": ("k_proj", "k"), "v_proj": ("v_proj", "v")}

# Checkpoint names under ``mlp.`` that this module names differently.
_MLP_RENAMES = {
    "mlp.pre_expert_proj.weight": "mlp.pooler.weight",
    "mlp.post_expert_proj.weight": "mlp.unpooler.weight",
    "mlp.experts.post_expert_norm.weight": "mlp.post_expert_norm_weight",
}


def _remap_layer_weight(name: str, rest: str) -> str:
    renamed = _MLP_RENAMES.get(rest)
    return name if renamed is None else name[: -len(rest)] + renamed
