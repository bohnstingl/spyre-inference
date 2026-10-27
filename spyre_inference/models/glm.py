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
correction bias, renormalized top-k and a routed scale. Their attention needs no adapter
(``spyre_inference.v1.attention.mla``); their MoE layers opt in here.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE, GlmMoeDsaForCausalLM
from vllm.model_executor.models.glm4_moe import Glm4MoE
from vllm.model_executor.models.glm4_moe_lite import Glm4MoeLiteForCausalLM

from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch
    from torch import nn
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _scale_down_weight(down_weight: torch.Tensor, scale: float) -> torch.Tensor:
    return down_weight * scale


def configure_glm_moe_layers(layers: Iterable[nn.Module], *, fold_routed_scale: bool) -> int:
    """Opt every GLM MoE layer into the Spyre MoE backend; returns the count.

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
        configure_spyre_moe_layer(
            runner.routed_experts,
            SpyreMoERecipe(
                activation="silu",
                routing="sigmoid_bias",
                prepare_down_weight=prepare_down_weight,
            ),
        )
        configured += 1
    if configured:
        logger.info("Spyre: configured %d GLM MoE layers.", configured)
    return configured


class SpyreGlm4MoeLiteForCausalLM(Glm4MoeLiteForCausalLM):
    """GLM-4.7-Flash on Spyre with Spyre MoE expert dispatch."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        configure_glm_moe_layers(self.model.layers, fold_routed_scale=True)


class SpyreGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    """GLM-5.x on Spyre with Spyre MoE expert dispatch and Spyre MLA + DSA attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        configure_glm_moe_layers(self.model.layers, fold_routed_scale=False)
