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

"""Nemotron-H MoE models (Nemotron-3 Nano/Super) on the Spyre MoE backend.

Their routed experts are non-gated ``relu2`` MLPs (``w13`` is the up projection alone),
routed by DeepSeek-V3's noaux_tc (sigmoid scores, a selection-only correction bias,
renormalized top-k, no groups). Super's experts run in a latent space: upstream's
``fc1_latent_proj`` / ``fc2_latent_proj`` wrap them as MoERunner's routed input/output
transforms, which SpyreMoERunner inherits unchanged.

The routed scale is folded into the down weights. In fp16 MoERunner divides the shared
expert's output by it instead, which only a decoder that divides its residual stream to
match would undo; ``NemotronHMoEDecoderLayer`` does not.
"""

from __future__ import annotations

import contextlib
from functools import partial
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import config as fused_moe_config
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM, NemotronHMoE

from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from torch import nn
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _scale_down_weight(down_weight: torch.Tensor, scale: float) -> torch.Tensor:
    return down_weight * scale


class _AcceptsNonGatedExperts:
    """``current_platform`` as ``FusedMoEConfig.__post_init__`` sees it: also XPU-like.

    Only that check reads ``is_xpu`` there, so nothing else takes an XPU path.
    """

    def __init__(self, platform) -> None:
        self._platform = platform

    def __getattr__(self, name: str):
        return getattr(self._platform, name)

    def is_xpu(self) -> bool:
        return True


@contextlib.contextmanager
def _non_gated_experts_allowed() -> Iterator[None]:
    """Let ``FusedMoEConfig`` accept a non-gated activation on Spyre while the model builds.

    Upstream refuses ``is_act_and_mul=False`` outside CUDA, ROCm and XPU because only their
    kernels implement it; the Spyre MoE backend implements ``relu2_no_mul`` itself, and
    ``configure_spyre_moe_layer`` rejects every layer it does not take. Only the config
    module's ``current_platform`` is swapped, so no other layer sees XPU.
    """
    saved = fused_moe_config.current_platform
    fused_moe_config.current_platform = _AcceptsNonGatedExperts(saved)
    try:
        yield
    finally:
        fused_moe_config.current_platform = saved


def configure_nemotron_h_moe_layers(layers: Iterable[nn.Module]) -> int:
    """Opt every Nemotron-H MoE block into the Spyre MoE backend; returns the count."""
    configured = 0
    for decoder in layers:
        moe = getattr(decoder, "mixer", None)
        if not isinstance(moe, NemotronHMoE):
            continue
        runner = moe.experts
        prepare_down_weight = None
        if runner.routed_scaling_factor != 1.0:
            prepare_down_weight = partial(_scale_down_weight, scale=runner.routed_scaling_factor)
            runner.routed_scaling_factor = 1.0
        configure_spyre_moe_layer(
            runner.routed_experts,
            SpyreMoERecipe(
                activation="relu2",
                routing="sigmoid_bias",
                prepare_down_weight=prepare_down_weight,
            ),
        )
        configured += 1
    if configured:
        logger.info("Spyre: configured %d Nemotron-H MoE layers.", configured)
    return configured


class SpyreNemotronHForCausalLM(NemotronHForCausalLM):
    """Nemotron-H on Spyre; MoE variants route their experts through the Spyre backend."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        with _non_gated_experts_allowed():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        configure_nemotron_h_moe_layers(self.model.layers)
