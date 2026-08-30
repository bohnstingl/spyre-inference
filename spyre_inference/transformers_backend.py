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
OOT registrations pick up on their own. Three things are left to HF's module code:

* RoPE — there is no RoPE fuser, so HF's ``rotary_emb`` survives and derives cos/sin
  inside the forward from int64 ``position_ids``, a cast torch-spyre cannot lower.
  Replaced here with a precomputed rotation cache and a matmul-only rotation.
* Attention — upstream keeps the vLLM ``Attention`` layers in a plain dict and indexes it
  by ``layer_idx`` from inside the HF attention forward. ``attach_attention_instances``
  turns them into submodules of the layers that use them, which is what lets the runner
  compile one decoder layer and reuse it (see the module comment there).
* Models shipping both ``config.json`` and ``params.json`` parse into a bare
  ``PretrainedConfig``, which HF cannot build a model from.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from vllm.logger import init_logger
from vllm.model_executor.models.transformers import (
    TransformersForCausalLM,
    vllm_attention_forward,
)

from spyre_inference.custom_ops.head_pad import original_head_dim

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# HF's registry name for upstream's attention interface, and for the Spyre one below.
_VLLM_ATTN_IMPL = "vllm"
SPYRE_ATTN_IMPL = "vllm_spyre"

# Attribute each layer's vLLM ``Attention`` is attached to. Deliberately the name in-tree
# vLLM models use (``LlamaAttention.attn``), so both paths present the same module tree to
# block discovery, to ``named_modules()`` walks and to a state dict.
VLLM_ATTN_ATTR = "attn"


def _spyre_vllm_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    attention_instances: dict[int, nn.Module] | None = None,
    **kwargs,
):
    """``vllm_attention_forward`` reading the layer off *module* rather than out of a dict.

    Upstream resolves it as ``attention_instances[module.layer_idx]``. Inside a compiled
    decoder layer that ``int`` becomes a Dynamo guard, so layer 1 fails layer 0's guards
    and every layer traces to a graph of its own — which is exactly the artifact reuse
    ``SPYRE_COMPILE_GRANULARITY=block`` exists to get. Reading a submodule is the same
    expression in every layer, which is why in-tree vLLM models share one artifact.

    The reshaping below mirrors upstream's ``vllm_attention_forward``
    (``vllm/model_executor/models/transformers/__init__.py``) — keep the two in step on a
    vLLM bump. Two deliberate differences beyond the lookup: a layer
    ``attach_attention_instances`` could not reach has no attribute to read and is handed
    back to upstream, dict lookup and all; and *scaling* is ignored, because
    ``attach_attention_instances`` has already copied it onto ``impl.scale`` (see
    ``_copy_attention_scale``).
    """
    self_attn = getattr(module, VLLM_ATTN_ATTR, None)
    if self_attn is None:
        return vllm_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            attention_instances=attention_instances,
            **kwargs,
        )

    hidden = query.shape[-2]
    head_dim_qk = query.shape[-1]
    head_dim_v = value.shape[-1]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    # Pad `value` up to the query/key head size when it is smaller (expanded MLA). A
    # larger last dim just means `value` isn't split per head, e.g. packed
    # grouped/multi-query projections, and needs no padding.
    pad_value = head_dim_v < head_dim_qk
    if pad_value:
        value = F.pad(value.view(-1, head_dim_v), (0, head_dim_qk - head_dim_v))
        value = value.reshape(hidden, -1)
    attn_output = self_attn.forward(query, key, value)
    if pad_value:
        attn_output = attn_output.view(-1, head_dim_qk)[..., :head_dim_v]
        attn_output = attn_output.reshape(hidden, -1)
    return attn_output, None


ALL_ATTENTION_FUNCTIONS.register(SPYRE_ATTN_IMPL, _spyre_vllm_attention_forward)


def _dispatches_through_vllm(module: nn.Module) -> Any | None:
    """The config *module* dispatches attention on, when that is upstream's vLLM interface.

    HF reads ``self.config._attn_implementation`` per call, so this is also the switch that
    routes a layer to the Spyre interface. ``vllm_mla`` is left alone: its interface takes
    ``(query, kv_c_normed, k_pe)`` rather than ``(query, key, value)``, and upstream
    already attaches those layers as ``_vllm_mla_attn`` submodules, so block discovery
    finds them without help from here.
    """
    config = getattr(module, "config", None)
    impl = getattr(config, "_attn_implementation", None)
    return config if impl in (_VLLM_ATTN_IMPL, SPYRE_ATTN_IMPL) else None


def _copy_attention_scale(module: nn.Module, instance: nn.Module) -> None:
    """Copy the HF module's softmax scale onto the vLLM layer, once, outside the forward.

    ``create_attention_instances`` builds every layer with the Llama default
    ``head_size ** -0.5`` and lets ``vllm_attention_forward`` correct it from the HF
    module's own ``scaling`` on every call. That correction cannot survive per-block
    sharing: with one artifact serving the whole stack, only the first layer's Python
    frame ever runs, so layers 1..N would keep the default — silently wrong for any model
    that does not derive its scale from ``head_size`` (Gemma 3's
    ``query_pre_attn_scalar``, Granite's ``attention_multiplier``). Doing it here instead
    is per-layer and runs before anything traces.

    ``fix_padded_attention_scale`` already keeps ``module.scaling`` and ``impl.scale`` in
    step for the head_dim-padded case, for exactly the same reason, so it is safe to run
    after: the two values it wrote are equal.
    """
    scaling = getattr(module, "scaling", None)
    if scaling is None:
        scaling = getattr(module, "scale", None)
    impl = getattr(instance, "impl", None)
    if impl is not None and isinstance(scaling, (int, float)) and not isinstance(scaling, bool):
        impl.scale = float(scaling)


def attach_attention_instances(model: nn.Module) -> int:
    """Make the Transformers backend's vLLM ``Attention`` layers submodules of their layer.

    Upstream builds them into ``model.attention_instances``, a plain ``dict`` that
    ``nn.Module.__setattr__`` never registers, and reaches them from the HF attention
    forward by ``attention_instances[module.layer_idx]``. Both halves of that cost the
    runner its per-layer compile granularity:

    * nothing under a decoder layer is a vLLM ``Attention``, so ``_repeated_block_lists``
      finds no block stack at all and ``_compile_for_spyre`` falls back to one whole-model
      graph whose compile cost grows with layer count;
    * the ``layer_idx`` index guards each layer's graph separately, so even a block stack
      it *did* find would compile once per layer instead of once per layer class.

    Attaching each layer to the HF attention module that uses it fixes both: discovery
    matches structurally, and ``_spyre_vllm_attention_forward`` reads the submodule
    instead of indexing. Called once per load, after weight loading and the device move —
    an ``Attention`` attached earlier would be pulled onto Spyre with its parent, and the
    runner deliberately keeps its buffers on the host.

    Sharing one artifact also means only the first layer's Python frame ever runs, so
    everything upstream did per forward has to move out here; ``_copy_attention_scale``
    is the one such thing.

    Returns the number of layers attached, 0 when *model* carries no
    ``attention_instances`` (not a Transformers-backend model, nothing to do).
    """
    instances = getattr(model, "attention_instances", None)
    if not instances:
        return 0

    attached: list[str] = []
    configs: dict[int, Any] = {}
    # Materialized: attaching mutates the ``_modules`` dicts this would otherwise walk.
    for name, module in list(model.named_modules()):
        index = getattr(module, "layer_idx", None)
        # HF names the module that dispatches attention ``<Model>Attention`` exactly;
        # ``endswith`` rather than ``in`` so a wrapper such as Zamba2's
        # ``Zamba2AttentionDecoderLayer`` -- which also carries a ``layer_idx`` and a
        # ``config`` -- does not collect a second, redundant ``.attn``. A vision tower's
        # attention passes this but is dropped by the interface check below, because it
        # dispatches on its own sub-config.
        if not type(module).__name__.endswith("Attention") or not isinstance(index, int):
            continue
        instance = instances.get(index)
        config = _dispatches_through_vllm(module)
        if instance is None or config is None:
            continue

        existing = getattr(module, VLLM_ATTN_ATTR, None)
        if existing is not None and existing is not instance:
            logger.warning(
                "%s already owns a .%s; leaving it on upstream's dict lookup.",
                name,
                VLLM_ATTN_ATTR,
            )
            continue
        if existing is None:
            setattr(module, VLLM_ATTN_ATTR, instance)
        _copy_attention_scale(module, instance)
        configs[id(config)] = config
        attached.append(name)

    if not attached:
        logger.warning(
            "Transformers backend: none of the %d vLLM Attention layers could be matched "
            "to an HF attention module, so they stay in attention_instances. Per-block "
            "compile will fall back to a whole-model graph.",
            len(instances),
        )
        return 0

    for config in configs.values():
        config._attn_implementation = SPYRE_ATTN_IMPL

    if len(attached) != len(instances):
        logger.warning(
            "Transformers backend: attached %d of %d vLLM Attention layers; the rest keep "
            "upstream's per-layer dict lookup and compile a graph each.",
            len(attached),
            len(instances),
        )
    logger.info(
        "Transformers backend: attached %d vLLM Attention layers as .%s submodules; "
        "attention dispatches through %r.",
        len(attached),
        VLLM_ATTN_ATTR,
        SPYRE_ATTN_IMPL,
    )
    return len(attached)


def _build_rotation_cache(
    inv_freq: torch.Tensor,
    scaling: float,
    max_position: int,
    padded_head_dim: int | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[max_position, 2, 2, head_dim // 2]`` rotation matrices ``[[cos, -sin], [sin, cos]]``.

    Working from ``inv_freq``/``attention_scaling`` inherits whatever rope scaling the
    module being replaced had baked into them. *padded_head_dim* extends the cache with
    identity blocks, so a Q/K padded up to it passes its trailing dimensions through.
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

    return rot.contiguous().to(dtype)


def _apply_rope_matmul(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, L, D]`` by ``rot`` ``[B, L, 2, 2, D // 2]``.

    Multiply-and-reduce rather than HF's ``rotate_half`` cat: Spyre cannot restickify the
    halves that slicing a stick-aligned head_dim produces.
    """
    b, h, seq, head_dim = x.shape
    pairs = x.transpose(1, 2).reshape(b, seq, h, 2, head_dim // 2)
    out = rot.unsqueeze(2).mul(pairs.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


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


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for an HF rotary embedding, returning ``(rot, None)`` in place of
    ``(cos, sin)``; the patched ``apply_rotary_pos_emb`` ignores the second element.

    The cache covers ``max_position`` up front rather than growing on demand: sizing it
    from the batch's positions needs an ``.item()``, so a host sync per step and a
    data-dependent guard ``torch.compile`` cannot trace.
    """

    def __init__(
        self,
        original: nn.Module,
        max_position: int,
        padded_head_dim: int | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        self._cpu_caches = {
            layer_type: _build_rotation_cache(
                inv_freq.to("cpu", torch.float32),
                scaling,
                max_position,
                padded_head_dim,
                dtype,
            )
            for layer_type, (inv_freq, scaling) in _rope_frequencies(original).items()
        }
        self._caches = self._cpu_caches

    def _apply(self, fn, recurse=True):
        # Prime the device caches when the model moves to Spyre, i.e. before compile, so only
        # the index_select is traced. They are not buffers (they are built after weight
        # loading) and there are no children, so super() has nothing to do.
        self._caches = {k: fn(v) for k, v in self._cpu_caches.items()}
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None):
        cache = self._caches[layer_type]
        rot = cache.index_select(0, position_ids.flatten())
        return rot.view(*position_ids.shape, *cache.shape[1:]), None


def _spyre_apply_rotary(q, k, rot, *args, **kwargs):
    """Rotate Q and K by the matrices ``_SpyreRotaryEmbedding`` returned."""
    return _apply_rope_matmul(q, rot), _apply_rope_matmul(k, rot)


def _rope_dispatch(original: Callable) -> Callable:
    """``apply_rotary_pos_emb`` replacement that hands stock HF calls to *original*.

    The patch lands on a modeling module in ``sys.modules`` and is never removed, so an HF
    model built later in the process has to keep working. Spyre's calls are the ones whose
    ``sin`` is the ``None`` standing in for ``_SpyreRotaryEmbedding``'s second return.
    """

    @functools.wraps(original)
    def apply_rotary_pos_emb(q, k, cos, sin=None, *args, **kwargs):
        if sin is None:
            return _spyre_apply_rotary(q, k, cos)
        return original(q, k, cos, sin, *args, **kwargs)

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


class SpyreTransformersForCausalLM(TransformersForCausalLM):
    """Transformers backend with the Spyre RoPE replacement wired in."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        self._max_position = vllm_config.model_config.max_model_len
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("SpyreTransformersForCausalLM ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
        only the rotated dims — but reach a shape mismatch here rather than a check:
        ``_maybe_pad_head_dim`` already rejects them whenever padding is needed.
        """
        # The text backbone holding rotary_emb; multimodal models nest it one level
        # deeper, at model.model.language_model, and carry the rope config on its own
        # config rather than the top-level one.
        inner = self.model.model if hasattr(self.model, "model") else self.model
        backbone = cast(nn.Module, getattr(inner, "language_model", inner))
        cfg = getattr(backbone, "config", self.model.config)

        # head_dim is already stick-aligned (the platform pads it, and the weight pass
        # pads Q/K interleaved to match), so the rotation only needs the pre-pad
        # frequencies identity-padded back out to the widened width.
        rope_source = backbone.get_submodule("rotary_emb")
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
        for name, module in self.model.named_modules():
            if module is spyre_rope:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding"):
                parent_name, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, attr, spyre_rope)
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = spyre_rope

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


# using_transformers_backend() compares _ModelInfo.architecture, which is model_cls.__name__,
# against "TransformersForCausalLM", so the subclass has to keep answering to that name.
SpyreTransformersForCausalLM.__name__ = "TransformersForCausalLM"
