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

"""Spyre adaptation of vLLM's Transformers backend.

Upstream's fusers replace HF's linear/norm/GLU modules with vLLM layers, which the Spyre
OOT registrations pick up on their own. Two things are left to HF's module code:

* RoPE — there is no RoPE fuser, so HF's ``rotary_emb`` survives and derives cos/sin
  inside the forward from int64 ``position_ids``, a cast torch-spyre cannot lower.
  Replaced here with the native path, ``SpyreRotaryEmbedding``,
  which is built based on the HF's frequencies. This file contains the required
  interface between the native path and the HF path.
* Models shipping both ``config.json`` and ``params.json`` parse into a bare
  ``PretrainedConfig``, which HF cannot build a model from.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from vllm.logger import init_logger
from vllm.model_executor.models.transformers import TransformersForCausalLM

from spyre_inference.custom_ops.head_pad import original_head_dim
from spyre_inference.custom_ops.rotary_embedding import (
    SpyreRotaryEmbedding,
    rotary_from_inv_freq,
    rotate_neox_2x2,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _rope_matmul_tokens_major(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[*tokens, H, D]`` by ``rot`` ``[*tokens, 2, 2, D // 2]``, shape kept.

    Multiply-and-reduce rather than HF's ``rotate_half`` cat: Spyre cannot restickify the
    halves that slicing a stick-aligned head_dim produces. This is the same rotation the
    native path's RoPE custom op runs, so it delegates to that kernel; the kernel takes a
    single flat token axis, hence the fold and unfold around it.
    """
    rope_half = inv_freq.shape[0]
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
    cos, sin = torch.cos(freqs) * scaling, torch.sin(freqs) * scaling
    rot = torch.stack([cos, -sin, sin, cos], dim=1).view(max_position, 2, 2, rope_half)

    if padded_head_dim is not None and padded_head_dim // 2 > rope_half:
        identity = torch.zeros(max_position, 2, 2, padded_head_dim // 2 - rope_half)
        identity[:, 0, 0, :] = 1.0
        identity[:, 1, 1, :] = 1.0
        rot = torch.cat([rot, identity], dim=-1)

    return rot.to(dtype)


def _apply_rope_matmul(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, L, D]`` by ``rot`` ``[B, L, 2, 2, D // 2]``.

    The heads-major layout most HF attention modules rotate in; the models that rotate
    pre-transpose go straight to ``_rope_matmul_tokens_major``.
    """
    return _rope_matmul_tokens_major(x.transpose(1, 2), rot).transpose(1, 2)


def _rope_frequencies(original: nn.Module) -> dict[str | None, tuple[torch.Tensor, float]]:
    """``{layer_type: (inv_freq, attention_scaling)}`` for the rope module being replaced.

    Models mixing global and sliding-window attention (Gemma 3, Olmo 3, ...) register one
    ``{layer_type}_inv_freq`` buffer per type and select between them on a third
    ``layer_type`` argument to ``forward``; a single rope is keyed under ``None``, its
    default.
    """
    layer_types = getattr(original, "layer_types", None)
    if not layer_types:
        scaling = float(getattr(original, "attention_scaling", 1.0))
        return {None: (original.get_buffer("inv_freq"), scaling)}

    freqs = {}
    for layer_type in layer_types:
        try:
            inv_freq = original.get_buffer(f"{layer_type}_inv_freq")
        except AttributeError:
            continue  # a layer type that does not rotate registers no buffer
        scaling = float(getattr(original, f"{layer_type}_attention_scaling", 1.0))
        freqs[layer_type] = (inv_freq, scaling)
    return freqs


# ModuleDict keys must be strings, so the single-rope case needs a stand-in for the
# ``layer_type=None`` default. Not a name any HF layer_type uses.
_DEFAULT_LAYER_TYPE = "__default__"


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for an HF rotary embedding.

    This routing layer to the native-path ``SpyreRotaryEmbedding``, so both backends
    run the same RoPE implementation.
    """

    def __init__(
        self,
        original: nn.Module,
        max_position: int,
        padded_head_dim: int | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        self._ropes = nn.ModuleDict(
            {
                (layer_type or _DEFAULT_LAYER_TYPE): rotary_from_inv_freq(
                    inv_freq,
                    scaling,
                    # The width Q/K actually arrive at: the padded head_dim when the
                    # platform widened it, else whatever this layer type rotates -- which
                    # can be per-type, where full-attention layers are one width and
                    # sliding ones another.
                    padded_head_dim if padded_head_dim is not None else 2 * inv_freq.shape[0],
                    max_position,
                    dtype,
                )
                for layer_type, (inv_freq, scaling) in _rope_frequencies(original).items()
            }
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None):
        rope = cast(SpyreRotaryEmbedding, self._ropes[layer_type or _DEFAULT_LAYER_TYPE])
        rot = rope.gather_rotation(position_ids)
        return rot.view(*position_ids.shape, *rot.shape[1:]), None


def _spyre_apply_rotary(q, k, rot, *args, **kwargs):
    """Rotate Q and K by the matrices ``_SpyreRotaryEmbedding`` returned."""
    return _apply_rope_matmul(q, rot), _apply_rope_matmul(k, rot)


def _rope_dispatch(original: Callable) -> Callable:
    """``apply_rotary_pos_emb`` replacement that hands stock HF calls to *original*.

    The patch lands on a modeling module in ``sys.modules`` and is never removed, so an HF
    model built later in the process has to keep working. Spyre's calls are the ones whose
    ``sin`` is the ``None`` standing in for ``_SpyreRotaryEmbedding``'s second return.

    HF spells this function two ways, and which one is in front of us is decided once, off
    the signature:

    * the common one takes the Q/K pair at ``[B, H, L, D]`` and returns both rotated;
    * some models take one tensor at a time at ``[B, L, H, D]``, before attention
      transposes it, and return just that tensor. Their ``unsqueeze_dim`` says where the
      head axis is, so it also tells us the layout is the one we can rotate.
    """
    params = inspect.signature(original).parameters

    if "k" in params or "key" in params:

        @functools.wraps(original)
        def apply_rotary_pos_emb(q, k, cos, sin=None, *args, **kwargs):
            if sin is None:
                return _spyre_apply_rotary(q, k, cos)
            return original(q, k, cos, sin, *args, **kwargs)

    else:

        @functools.wraps(original)
        def apply_rotary_pos_emb(x, cos, sin=None, unsqueeze_dim=1, **kwargs):
            if sin is None:
                if unsqueeze_dim != x.ndim - 2:
                    raise NotImplementedError(
                        f"Spyre RoPE expects the heads axis at {x.ndim - 2} for a "
                        f"{x.ndim}D tensor, got unsqueeze_dim={unsqueeze_dim}."
                    )
                return _rope_matmul_tokens_major(x, cos)
            return original(x, cos, sin, unsqueeze_dim=unsqueeze_dim, **kwargs)

    apply_rotary_pos_emb._spyre_patched = True
    return apply_rotary_pos_emb


def _rope_at_original_head_dim(cfg, rope: nn.Module, orig_head_dim: int) -> nn.Module:
    """Rebuild *rope* at the pre-pad head_dim.

    HF derived ``inv_freq`` from the widened ``config.head_dim``, giving one frequency
    per padded pair instead of per real pair.
    """
    padded = cfg.head_dim
    cfg.head_dim = orig_head_dim
    try:
        return type(rope)(config=cfg)
    finally:
        cfg.head_dim = padded


def _rebase_onto_text_backbone(
    weights: Iterable[tuple[str, torch.Tensor]],
    prefix: str,
    dropped: Counter[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Re-address a multimodal checkpoint onto its text backbone, dropping the towers.

    ``force_text_backbone`` hands the backend the text config, so the module tree is the
    text stack alone while the checkpoint still nests it under *prefix*
    (``model.language_model.``). Rebasing here — ahead of ``TransformersBase``'s derived
    ``hf_to_vllm_mapper``, whose catch-all ``^(model\\.)((?!<children>).+)`` rule would
    otherwise strip ``model.`` off every nested name and report the lot as unexpected —
    is what vLLM's native ``Gemma4ForCausalLM`` does with an ``orig_to_new_prefix`` entry.

    Anything else under the prefix's root (vision/audio towers and their projectors) has
    no home in a text-only tree and is counted into *dropped*; names outside that root
    (``lm_head`` and friends) pass through untouched.
    """
    root = prefix.partition(".")[0] + "."
    for name, weight in weights:
        if name.startswith(prefix):
            yield root + name[len(prefix) :], weight
        elif name.startswith(root):
            dropped[name[len(root) :].partition(".")[0]] += 1
        else:
            yield name, weight


class SpyreTransformersForCausalLM(TransformersForCausalLM):
    """Transformers backend with the Spyre RoPE replacement wired in."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        self._max_position = vllm_config.model_config.max_model_len
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("SpyreTransformersForCausalLM ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        prefix = getattr(self.config, "_spyre_text_backbone_prefix", None)
        if prefix:
            dropped: Counter[str] = Counter()
            weights = _rebase_onto_text_backbone(weights, prefix, dropped)
            result = super().load_weights(weights)
            if dropped:
                logger.info(
                    "Text-only backbone: rebased checkpoint weights off %r and skipped "
                    "%d non-text weight(s) (%s).",
                    prefix,
                    sum(dropped.values()),
                    ", ".join(f"{name}: {count}" for name, count in sorted(dropped.items())),
                )
        else:
            result = super().load_weights(weights)
        self._patch_rope()
        return result

    @staticmethod
    def _fix_generic_config(vllm_config: VllmConfig) -> None:
        """Re-resolve the bare PretrainedConfig that vLLM's Mistral parser produces for
        repos shipping both config.json and params.json, which AutoModel.from_config
        rejects, and force HF-format weight loading. ``--config-format hf`` skips it."""
        hf_config = vllm_config.model_config.hf_config
        if type(hf_config) is not PretrainedConfig:
            return

        model_id = vllm_config.model_config.hf_config_path or vllm_config.model_config.model
        try:
            resolved = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.revision,
            )
        except Exception:
            logger.warning("AutoConfig re-resolve failed for %s", model_id, exc_info=True)
            return

        skip = {"model_type", "_name_or_path", "transformers_version", "auto_map", "architectures"}
        for key, val in hf_config.to_dict().items():
            if key not in skip and val is not None:
                setattr(resolved, key, val)

        vllm_config.model_config.hf_config = resolved
        vllm_config.model_config.hf_text_config = resolved.get_text_config()
        if vllm_config.load_config.load_format in ("auto", "mistral"):
            vllm_config.load_config.load_format = "hf"
        logger.debug(
            "Re-resolved config: %s (model_type=%s), load_format=hf",
            type(resolved).__name__,
            resolved.model_type,
        )

    def _patch_rope(self):
        """Swap HF's rotary embedding and ``apply_rotary_pos_emb`` for the Spyre ones.

        Partial rotary dimensions (e.g. Phi-3) are unsupported — the cache would cover
        only the rotated dims. ``_maybe_pad_head_dim`` rejects them whenever padding is
        needed; for the rest, ``_rope_matmul_tokens_major`` raises on the width mismatch
        at the first rotation.
        """
        # The text backbone holding rotary_emb; multimodal models nest it one level
        # deeper, at model.model.language_model, and carry the rope config on its own
        # config rather than the top-level one.
        inner = self.model.model if hasattr(self.model, "model") else self.model
        backbone = cast(nn.Module, getattr(inner, "language_model", inner))
        cfg = getattr(backbone, "config", self.model.config)

        # Not every backbone rotates: BERT-family encoders position with learned
        # embeddings and register no rotary_emb.
        try:
            rope_source = backbone.get_submodule("rotary_emb")
        except AttributeError:
            logger.debug("%s has no rotary_emb, leaving rope alone", type(backbone).__name__)
            return

        # head_dim is already stick-aligned (the platform pads it, and the weight pass
        # pads Q/K interleaved to match), so the rotation only needs the pre-pad
        # frequencies identity-padded back out to the widened width.
        orig_head_dim = original_head_dim(cfg)
        padded_head_dim = None
        if orig_head_dim is not None:
            padded_head_dim = cfg.head_dim
            rope_source = _rope_at_original_head_dim(cfg, rope_source, orig_head_dim)

        spyre_rope = _SpyreRotaryEmbedding(
            rope_source,
            self._max_position,
            padded_head_dim,
            next(self.model.parameters()).dtype,
        )
        backbone.rotary_emb = spyre_rope

        patched_mods: set[int] = set()
        repointed = 1  # backbone.rotary_emb, above
        # spyre_rope's own children are SpyreRotaryEmbedding instances, so they match the
        # "*RotaryEmbedding" test below; repointing one at its own parent would nest the
        # wrapper inside itself.
        own = {id(module) for module in spyre_rope.modules()}
        # Scoped to the text backbone: a multimodal model's vision and audio towers carry
        # their own rotary embeddings over their own coordinates (2-D over patch positions,
        # say), which a text rotation cache cannot stand in for.
        # Materialised because attaching spyre_rope below adds submodules as we walk.
        for name, module in list(backbone.named_modules()):
            if id(module) in own:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding"):
                parent_name, _, attr = name.rpartition(".")
                parent = backbone.get_submodule(parent_name) if parent_name else backbone
                setattr(parent, attr, spyre_rope)
                repointed += 1
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = spyre_rope
                repointed += 1

            # apply_rotary_pos_emb is a module-level function in HF modeling files, so it
            # is patched once per modeling module rather than per layer.
            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            existing = getattr(mod, "apply_rotary_pos_emb", None)
            if existing is None or getattr(existing, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _rope_dispatch(existing)
            patched_mods.add(id(mod))

        # Logged because the swap is invisible in the module dump the backend prints, and
        # its absence is the failure mode: HF's rope stays in the graph and the model
        # aborts deep in the compiler on an fp32 outer product instead of here.
        logger.info(
            "Spyre RoPE installed on %s: %d rotary reference(s) repointed, "
            "apply_rotary_pos_emb patched in %d modeling module(s)",
            type(backbone).__name__,
            repointed,
            len(patched_mods),
        )


# using_transformers_backend() compares _ModelInfo.architecture, which is model_cls.__name__,
# against "TransformersForCausalLM", so the subclass has to keep answering to that name.
SpyreTransformersForCausalLM.__name__ = "TransformersForCausalLM"
