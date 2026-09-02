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

"""Spyre adaptations for vLLM's Gemma-4 model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

if TYPE_CHECKING:
    from torch import nn
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

# Scalar buffers that ``Gemma4Model`` owns and ``Gemma4SelfDecoderLayers``
# re-exposes as plain attributes.
_ALIASED_SCALARS = (
    "normalizer",
    "embed_scale_per_layer",
    "per_layer_input_scale",
    "per_layer_projection_scale",
)


def force_text_backbone(engine_args: EngineArgs) -> None:
    """Default gemma-4 to its text-only backbone (it ships multimodal).

    Sets ``hf_overrides["architectures"]`` so ``create_model_config`` resolves
    ``Gemma4ForCausalLM`` instead of the multimodal default. Skipped when the user
    already pinned an architecture (dict or callable ``hf_overrides``).
    """
    ov = engine_args.hf_overrides
    user_arch = callable(ov) or (isinstance(ov, dict) and "architectures" in ov)
    if "gemma-4" in (engine_args.model or "").lower() and not user_arch and isinstance(ov, dict):
        overrides = cast("dict[str, Any]", ov)
        overrides["architectures"] = ["Gemma4ForCausalLM"]
        logger.info("gemma-4: loading text-only backbone Gemma4ForCausalLM.")


def register_aliased_scalars(decoder: nn.Module) -> None:
    """Turn the self-decoder's aliased scalar attributes into buffers."""
    buffers = dict(decoder.named_buffers(recurse=False))
    for name in _ALIASED_SCALARS:
        scalar = getattr(decoder, name, None)
        if scalar is None or name in buffers:
            continue
        delattr(decoder, name)
        decoder.register_buffer(name, scalar, persistent=False)


class SpyreGemma4ForCausalLM(Gemma4ForCausalLM):
    """Gemma-4 with the self-decoder's aliased scalars registered as buffers.

    ``Gemma4SelfDecoderLayers`` holds four scalar buffers owned by ``Gemma4Model``
    as plain tensor attributes. ``model.to("spyre")`` rebinds the parent's buffers
    but leaves the aliases on CPU, so the compiled ``embed_input_ids`` feeds a 0-d
    CPU tensor into Inductor, which has no notion of a live CPU graph input.
    Re-registering the aliases restores the parent's stated intent (move with the
    model, interact with torch.compile) and needs no change to the embedding math:
    a device-side 0-d scalar lowers fine.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        register_aliased_scalars(self.model.self_decoder)
