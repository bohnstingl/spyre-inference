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

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PretrainedConfig
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

# Where the text stack sits inside the multimodal checkpoint, read by
# SpyreTransformersForCausalLM to rebase the weight names onto the text-only module
# tree. vLLM's own Gemma4ForCausalLM rebases the same prefix for the same reason.
_CHECKPOINT_TEXT_PREFIX = "model.language_model."


def force_text_backbone(engine_args: EngineArgs) -> None:
    """Default gemma-4 to its text-only backbone (it ships multimodal).

    Replaces the nested ``Gemma4Config`` with its ``Gemma4TextConfig``, which is what
    both model paths key off:

    * The Transformers backend picks its class from whether ``hf_config`` *is*
      ``hf_text_config`` (``ModelConfig._get_transformers_backend_cls``) and then builds
      the HF model with ``AutoModel.from_config(config=hf_config)``. Overriding
      ``architectures`` alone therefore changes nothing there: the run stays on
      ``TransformersMultiModalForCausalLM`` wrapping HF's multimodal ``Gemma4Model``,
      whose ``forward`` unconditionally calls ``get_placeholder_mask`` — three
      ``input_ids == <token id>`` compares, and Spyre's layout propagation rejects a
      ``torch.bool`` result from an int32 operand.
    * vLLM's native path resolves ``architectures``, which the text config carries.

    Skipped when the user pinned ``architectures`` themselves.
    """
    if "gemma-4" not in (engine_args.model or "").lower():
        return

    user_overrides = engine_args.hf_overrides
    if isinstance(user_overrides, dict) and "architectures" in user_overrides:
        return

    engine_args.hf_overrides = _TextBackboneOverride(user_overrides)


class _TextBackboneOverride:
    """``hf_overrides`` callable that swaps a config for its text config.

    A class rather than a closure because ``VllmConfig`` is pickled to the engine-core
    process, and a local function is not picklable.
    """

    def __init__(self, user_overrides: dict[str, Any] | Callable | None) -> None:
        self.user_overrides = user_overrides

    def __call__(self, config: PretrainedConfig) -> PretrainedConfig:
        # Installing a callable takes vLLM's own dict handling out of the loop
        # (hf_overrides is either a dict or a callable, never both), so apply whatever
        # the user asked for to the config we hand back.
        user_overrides = self.user_overrides
        if callable(user_overrides):
            config = user_overrides(config)
        text_config = config.get_text_config()
        if text_config is config:
            return config  # already text-only, e.g. a text-only re-upload
        if isinstance(user_overrides, dict):
            _apply_overrides(text_config, cast("dict[str, Any]", user_overrides))
        if not getattr(text_config, "architectures", None):
            text_config.architectures = ["Gemma4ForCausalLM"]
        text_config._spyre_text_backbone_prefix = _CHECKPOINT_TEXT_PREFIX
        logger.info("gemma-4: loading the text-only backbone (%s).", type(text_config).__name__)
        return text_config


def _apply_overrides(config: PretrainedConfig, overrides: dict[str, Any]) -> None:
    """Apply the dict form of ``hf_overrides`` to *config*, as ``ModelConfig`` would."""
    from transformers import PretrainedConfig

    for key, value in overrides.items():
        attr = getattr(config, key, None)
        if isinstance(value, dict) and isinstance(attr, PretrainedConfig):
            attr.update(value)
        else:
            setattr(config, key, value)


def install_spyre_patches() -> None:
    """Register Gemma-4's aliased ``normalizer`` as a buffer so it follows the model.

    ``Gemma4SelfDecoderLayers`` stores ``normalizer`` as a plain tensor attribute
    aliased from ``Gemma4Model``'s buffer. ``model.to("spyre")`` rebinds the parent's
    buffer to a device tensor but leaves this alias on CPU, so the compiled
    ``embed_input_ids`` feeds a 0-d CPU tensor into Inductor, which has no notion of a
    live CPU graph input. Re-registering it as a buffer restores the parent's documented
    intent (move with the model, interact with torch.compile) and needs no change to the
    embedding math. A device-side 0-d scalar lowers fine.
    """
    from vllm.model_executor.models.gemma4 import Gemma4SelfDecoderLayers

    if getattr(Gemma4SelfDecoderLayers, "_spyre_patched", False):
        return

    orig_init = Gemma4SelfDecoderLayers.__init__

    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        normalizer = self.normalizer
        del self.normalizer
        self.register_buffer("normalizer", normalizer, persistent=False)

    Gemma4SelfDecoderLayers.__init__ = __init__  # ty: ignore[invalid-assignment]
    Gemma4SelfDecoderLayers._spyre_patched = True
    logger.info(
        "Spyre: Gemma-4 normalizer registered as a buffer so it follows the model to "
        "device (upstream aliases it as a plain CPU attribute)."
    )
