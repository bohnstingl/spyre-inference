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

"""TEMPORARY CPU FALLBACK: GLM / DeepSeek-V3 (noaux_tc) MoE router expert selection on the host.

This is a deliberate exception to spyre-inference's rule of no CPU fallbacks, taken to get
correct GLM-5.x routing until Spyre can run the router GEMM with fp32 accumulation and an
fp32 result. Remove it once that lands; the repro filed with torch-spyre is
``vllm_spyre_next/ZRL/claude/repro_spyre_fp32_accum_router_gemm.py``.

Why. The router computes ``sigmoid(x @ W_gate.T) + e_score_correction_bias`` and keeps the
top-k experts. The model is trained with that GEMM in fp32 (GLM-5.2: ``moe_router_dtype:
float32``; vLLM forces fp32 for ``glm_moe_dsa``), and on real GLM-5.2 inputs the k-th and
(k+1)-th biased scores are a median 1.6e-4 apart at a value near 35. Spyre's matmul
accumulates and returns fp16 only, so on device only 69% of real tokens get the expert set
vLLM's fp32 router picks (96% with fp32 accumulation rounded to fp16, 100% in fp32).

What runs on the host: the gate GEMM, sigmoid, bias, top-k and renormalization, as vLLM's
own CPU router (``cpu_router._grouped_topk``) computes them, in fp32. The experts and
everything else stay on Spyre. Cost per MoE layer per step: the [tokens, hidden] fp16
activation goes to the host and a [tokens, experts] fp16 routing table comes back, and the
op is opaque to the block graph.
"""

from __future__ import annotations

import types
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.cpu_router import _grouped_topk
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from spyre_inference.custom_ops.utils import convert

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

# Per MoE layer, keyed by layer name: the gate weight and selection bias, kept on the host
# in fp32. Registered at model construction, filled after weight loading.
_HOST_ROUTERS: dict[str, HostRouter] = {}


class HostRouter:
    """fp32 host copies of one MoE layer's gate and routing settings."""

    def __init__(self, experts: RoutedExperts, gate: torch.nn.Module) -> None:
        self.gate = gate
        self.top_k = experts.top_k
        self.renormalize = experts.renormalize
        self.num_expert_group = experts.num_expert_group or 1
        self.topk_group = experts.topk_group or 1
        self.bias_param = experts.e_score_correction_bias
        self.weight_t: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def capture(self) -> None:
        """Take fp32 host copies once the checkpoint is loaded, before the device move."""
        weight = self.gate.weight.detach()
        # SpyreTransposedWeightMethod stores the gate as Wᵀ [hidden, experts].
        if weight.shape[0] != self.gate.input_size:
            weight = weight.t()
        self.weight_t = weight.to("cpu", torch.float32).contiguous()
        assert self.bias_param is not None
        self.bias = self.bias_param.detach().to("cpu", torch.float32).clone()

    def route(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dense [tokens, experts] renormalized routing weights, zero off the top-k."""
        assert self.weight_t is not None and self.bias is not None, "capture() not run"
        # .cpu() before the upcast: an fp32 tensor read back from Spyre may be staggered.
        logits = hidden_states.cpu().float() @ self.weight_t
        weights, ids = _grouped_topk(
            logits,
            self.top_k,
            self.renormalize,
            self.num_expert_group,
            self.topk_group,
            "sigmoid",
            1.0,
            self.bias,
        )
        return torch.zeros_like(logits).scatter(1, ids.long(), weights)


def _host_route(
    hidden_states: torch.Tensor, layer_name: LayerNameType, num_experts: int
) -> torch.Tensor:
    router = _HOST_ROUTERS[_resolve_layer_name(layer_name)]
    dense = router.route(hidden_states)
    return convert(dense.to(hidden_states.dtype), device=hidden_states.device)


def _host_route_fake(
    hidden_states: torch.Tensor, layer_name: LayerNameType, num_experts: int
) -> torch.Tensor:
    # A hoisted LayerName is opaque to a fake kernel, hence the explicit width.
    return hidden_states.new_empty((hidden_states.shape[0], num_experts))


@lru_cache(maxsize=1)
def _register_op() -> None:
    from vllm.platforms import current_platform

    direct_register_custom_op(
        op_name="spyre_glm_host_route",
        op_func=_host_route,
        fake_impl=_host_route_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def _host_route_forward(self: torch.nn.Module, hidden_states: torch.Tensor):
    """The gate's forward under the host router: the routing table in place of logits."""
    dense = torch.ops.vllm.spyre_glm_host_route(
        hidden_states,  # ty: ignore[invalid-argument-type]
        _encode_layer_name(self.spyre_route_name),  # ty: ignore[invalid-argument-type]
        self.output_size,  # ty: ignore[invalid-argument-type]
    )
    return dense, None


def install_host_router(name: str, experts: RoutedExperts, gate: torch.nn.Module) -> None:
    """TEMPORARY CPU FALLBACK: make ``gate`` return the host-computed dense routing table.

    Both GLM MoE classes call the gate where upstream expects router logits
    (``Glm4MoE.forward`` itself, ``DeepseekV2MoE`` through ``MoERunner``), so replacing the
    gate's forward reaches both. Its output then is the routing table the Spyre MoE's
    ``dense_routes`` recipe consumes, not logits.
    """
    _register_op()
    _HOST_ROUTERS[name] = HostRouter(experts, gate)
    # A bound method reading the name off the module, as attn_layer does: one code object
    # and a hoisted LayerName keep the per-block compile shared across layers.
    gate.spyre_route_name = name
    gate.forward = types.MethodType(_host_route_forward, gate)
    logger.warning_once(
        "TEMPORARY CPU FALLBACK: GLM MoE router expert selection (gate GEMM, sigmoid, "
        "bias, top-k) runs on the host in fp32 for every MoE layer, because Spyre's "
        "matmul has no fp32 accumulation; the experts stay on Spyre. "
        "See spyre_inference/models/glm_cpu_router.py."
    )


def capture_host_routers() -> None:
    for router in _HOST_ROUTERS.values():
        router.capture()
