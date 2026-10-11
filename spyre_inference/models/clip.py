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

"""CLIP on Spyre: ``SpyreCLIPEmbeddingModel``, and the EngineArgs override that must
land before ModelConfig is built.

The LayerNorm boundary-norm swap lives in ``multimodal/clip.py`` instead
(an instance-level, post-load patch)."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import vllm.model_executor.models.clip as vllm_clip
from vllm.logger import init_logger
from vllm.model_executor.models.clip import (
    CLIPEmbeddingModel,
    _get_vision_feature_select_strategy,
)
from vllm.v1.attention.backend import AttentionType

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.spyre_pooler import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import expand_packed_embeds_to_encoder_grid

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)


def force_disable_chunked_prefill(engine_args: EngineArgs) -> None:
    """SpyreAllPool (CLIP's token-level pooler) rejects chunked prefill outright,
    but ModelConfig.is_chunked_prefill_supported doesn't exclude ALL-type token
    pooling, so it isn't disabled automatically. Mirrors
    models.gemma4.force_text_backbone's early get_config() probe.
    """
    if engine_args.enable_chunked_prefill is not None:
        return
    from vllm.transformers_utils.config import get_config

    try:
        hf_config = get_config(
            engine_args.hf_config_path or engine_args.model,
            engine_args.trust_remote_code,
            engine_args.revision,
            engine_args.code_revision,
            engine_args.config_format,
            token=engine_args.hf_token,
        )
    except Exception:
        return
    if getattr(hf_config, "model_type", None) != "clip":
        return
    engine_args.enable_chunked_prefill = False
    logger.info("CLIP: disabling chunked prefill (unsupported with token-level pooling).")


# Mixed image+text steps: vLLM 0.28's CLIPEmbeddingModel treats a step holding any
# image as image-only, so its text requests skip the text tower. Backport of
# vllm-project/vllm#53165 (in vLLM >= 0.29). Remove this, and the spyre_align_token_mask
# call in TorchSpyreModelRunner._preprocess, once the vLLM pin reaches 0.29.


def _has_mm(multimodal_embeddings) -> bool:
    return multimodal_embeddings is not None and len(multimodal_embeddings) > 0


def has_text_tokens(multimodal_embeddings, is_multimodal: torch.Tensor | None) -> bool:
    """Whether the step still has text rows to run through the text tower."""
    has_mm = _has_mm(multimodal_embeddings)
    if not has_mm or is_multimodal is None:
        return not has_mm
    return bool((~convert(is_multimodal, device="cpu")).any())


def merge_text_and_vision(
    text: torch.Tensor, vision: torch.Tensor, is_multimodal: torch.Tensor | None
) -> torch.Tensor:
    """Text-tower output on text rows, the vision embedding on image rows."""
    if is_multimodal is None:
        return text
    src = convert(is_multimodal, device="cpu")[: text.shape[0]]
    if not bool(src.any()):
        return text
    mask = torch.zeros(text.shape[0], dtype=torch.bool)
    mask[: src.shape[0]] = src
    # Unsqueezed on the host: a device-side [N, 1] view gives the result a layout that
    # index_select faults the card on past row 63 (torch-spyre#5230).
    return torch.where(convert(mask.unsqueeze(-1), device=text.device), vision, text)


def select_class_rows(feats: torch.Tensor) -> torch.Tensor:
    """Each image's class token, ``feats[:, :1, :]``, gathered rather than sliced.

    Past one image the slice is a strided view, and the on-card post-norm reads it as
    contiguous: every image after the first got rows of the first image instead.
    """
    b, s, h = feats.shape
    rows = torch.arange(b, dtype=torch.int64) * s
    return select_rows(feats.reshape(b * s, h), rows).reshape(b, 1, h)


class SpyreCLIPEmbeddingModel(CLIPEmbeddingModel):
    """CLIP with its text tower on the encoder backend, and correct mixed and batched
    image steps."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The text tower takes its layers from the clip module's ``Attention``. ENCODER_ONLY
        # skips the KV cache a pooling tower never reads; it stays causal, since only
        # ``EncoderOnlyAttention`` clears the metadata's causal flag.
        attention = vllm_clip.Attention
        encoder_only = partial(attention, attn_type=AttentionType.ENCODER_ONLY)
        vllm_clip.Attention = encoder_only  # ty: ignore[invalid-assignment]
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_clip.Attention = attention
        self._spyre_mm_token_mask: torch.Tensor | None = None

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        out = super().embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)
        self._spyre_mm_token_mask = is_multimodal if _has_mm(multimodal_embeddings) else None
        self._is_text_input = has_text_tokens(multimodal_embeddings, is_multimodal)
        return out

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
        out = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kw)
        mask = self._spyre_mm_token_mask
        if self._is_text_input and mask is not None and inputs_embeds is not None:
            out = merge_text_and_vision(out, inputs_embeds, mask)
        return out

    def spyre_align_token_mask(self, num_tokens: int, grid) -> None:
        """Trim the mask to the step's real rows, then lay it out as the grid, if any.

        Padding rows are not text: an image-only step then skips the text tower.
        """
        mask = self._spyre_mm_token_mask
        if mask is None:
            return
        mask = convert(mask, device="cpu")[:num_tokens]
        self._is_text_input = bool((~mask).any())
        if grid is not None:
            extent, width, query_lens = grid
            grid_mask = expand_packed_embeds_to_encoder_grid(
                mask.unsqueeze(-1), query_lens, width, extent
            )
            mask = grid_mask.squeeze(-1)
        self._spyre_mm_token_mask = mask

    def get_image_features(self, pixel_values, feature_select_strategy=None):
        if feature_select_strategy is None and pixel_values.shape[0] > 1:
            pooling_type = self.pooler_config.seq_pooling_type
            assert pooling_type is not None
            if _get_vision_feature_select_strategy(pooling_type) == "class":
                feature_select_strategy = select_class_rows
        return super().get_image_features(pixel_values, feature_select_strategy)
