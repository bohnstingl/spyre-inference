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

"""Spyre OOT replacement for RotaryEmbedding.

Applies rotary position embeddings on the Spyre device via a complex-free 2x2
rotation-matrix formulation (ported from foundation-model-stack). The 2x2 rotation
cache is held device-resident; ``forward_oot`` gathers this pass's per-token slice on
Spyre with ``index_select`` (torch-spyre#3418 gave single-row gather a kernel), then
applies the rotation with ``_rotate_neox_2x2``. Both run directly in the full-model
compile graph — no opaque op wraps them.

The one requirement for the in-graph gather: the device-resident cache must be
**built before compile**, not lazily inside the traced forward. Building the host
rotation cache (chunk/stack/view over ``cos_sin_cache``) and moving it to Spyre for
the first time *inside* the traced graph segfaults libsenlib during warmup; a cache
that is already materialized on-device before tracing indexes cleanly. ``_apply``
therefore primes the device cache when the module is moved to Spyre (which happens
before ``torch.compile`` wraps the model), so ``forward_oot`` only traces the
``index_select`` over an existing device tensor.

Only neox-style full rotary is supported; other configs raise
``NotImplementedError`` at construction instead of silently falling back to CPU.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)

from .utils import convert

logger = init_logger(__name__)

# Spyre stick size = 64 float16 elements. The 2x2 layout's inner dim is rotary_dim // 2
# and must be a stick multiple, otherwise the split-half view has a sub-stick stride the
# inductor rejects. The platform pads head_dim to a 128-multiple before RoPE is built
# (see head_pad.py), which guarantees this; a sub-stick inner is rejected at construction.
_SPYRE_STICK = 64


def _rotate_neox_2x2(
    x: torch.Tensor,
    rot: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Apply full neox RoPE via per-token 2x2 rotation matrices.

    ``x`` is [T, H*head_size] or [T, H, head_size]; ``rot`` is [T, 2, 2, head_size // 2].
    The inner dim head_size // 2 is stick-aligned (the platform pads head_dim to a
    128-multiple before RoPE is built), so the split-half pairing is a pure view.
    Returns the rotated tensor with ``x``'s shape.
    """
    num_tokens = x.shape[0]
    inner = head_size // 2
    x_pairs = x.view(num_tokens, -1, 2, inner)
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)
    return out.flatten(-2).view(x.shape)


class _SpyreRotaryMixin:
    """Spyre RoPE wiring shared by the base and llama3 OOT classes.

    Runs the 2x2 rotation on Spyre for supported configs; unsupported configs raise
    ``NotImplementedError`` at construction. The rotation cache is derived lazily from
    the base ``cos_sin_cache`` (inheriting all rope-scaling variants) and kept on CPU.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only neox full rotary has a Spyre kernel; gptj/interleaved and partial
        # rotary are rejected here rather than run on CPU.
        if not (self.is_neox_style and self.rotary_dim == self.head_size):
            raise NotImplementedError(
                "SpyreRoPE supports only neox-style full rotary (rotary_dim == "
                f"head_size); got is_neox_style={self.is_neox_style}, "
                f"rotary_dim={self.rotary_dim}, head_size={self.head_size}."
            )
        # The 2x2 rotation's inner dim (rotary_dim // 2) must be stick-aligned so the
        # split-half pairing is a pure view. The platform pads head_dim to a 128-multiple
        # before RoPE is built (see head_pad.py), which guarantees this; a sub-stick inner
        # here (e.g. head_size=64) means head padding did not run, so reject rather than
        # silently fall back to CPU.
        if (self.rotary_dim // 2) % _SPYRE_STICK != 0:
            raise NotImplementedError(
                f"SpyreRoPE needs the 2x2 rotation inner dim rotary_dim // 2 = "
                f"{self.rotary_dim // 2} to be a multiple of the {_SPYRE_STICK}-element "
                f"stick; got rotary_dim={self.rotary_dim}. The platform pads head_dim to a "
                "128-multiple before RoPE is built, so this indicates head padding did not "
                "run for this model."
            )
        self._padded_inner = self.rotary_dim // 2
        self._rotation_cache: torch.Tensor | None = None
        self._device_rotation_cache: torch.Tensor | None = None

    def _apply(self, fn, recurse=True):
        # cos_sin_cache has no Spyre kernel, so it is deliberately kept on CPU (we skip
        # super()._apply, which would move it to the device). But a move to Spyre must
        # PRIME the device-resident rotation cache here: forward_oot indexes it inside
        # the compiled full-model graph, and building it lazily during that first traced
        # forward (host chunk/stack/view -> device transfer) segfaults libsenlib. A cache
        # already materialized on-device before torch.compile traces indexes cleanly.
        # fn is the .to()/.cuda()/... convert closure; probe it to learn the target device.
        device = fn(torch.zeros(1, dtype=self.dtype)).device
        if device.type != "cpu":
            self._get_device_rotation_cache(device)
        return self

    def _get_rotation_cache(self) -> torch.Tensor:
        """Lazily build the CPU 2x2 rotation cache [max_pos, 2, 2, _padded_inner] from
        cos_sin_cache ([[cos, -sin], [sin, cos]]), zero-padding the inner dim up to
        _padded_inner when a padded head injected a narrower original-frequency cache."""
        if self._rotation_cache is None:
            # Derive inner from the cache actually present, not rotary_dim: when a
            # head is padded (head_size=64 -> 128), fix_padded_rope injects the
            # original narrower cos_sin_cache so the real frequencies survive; the
            # trailing dims are then zero-padded to _padded_inner (harmless because
            # the matching x pair dims are zero from weight padding).
            inner = self.cos_sin_cache.shape[-1] // 2
            cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
            cache = torch.stack([cos, -sin, sin, cos], dim=1).view(
                self.cos_sin_cache.shape[0], 2, 2, inner
            )
            if self._padded_inner != inner:
                cache = torch.nn.functional.pad(cache, (0, self._padded_inner - inner))
            self._rotation_cache = cache
        return self._rotation_cache

    def _get_device_rotation_cache(self, device: torch.device) -> torch.Tensor:
        """Device-resident copy of the 4D rotation cache ``[max_pos, 2, 2, padded]``,
        built once from the CPU cache so the per-pass gather runs on-device via
        ``index_select`` (single-row gather has a kernel since torch-spyre#3418)."""
        if self._device_rotation_cache is None:
            self._device_rotation_cache = convert(
                self._get_rotation_cache().contiguous(), device=device, dtype=self.dtype
            )
        return self._device_rotation_cache

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Gather this pass's per-token 2x2 rotation slice on-device with index_select,
        # then apply the rotation — both directly in the full-model compile graph, no
        # opaque op. The device cache was primed in _apply before compile (see module
        # docstring), so only the index_select over an existing device tensor is traced.
        cache = self._get_device_rotation_cache(query.device)
        rot = cache.index_select(0, positions.flatten())
        out_query = _rotate_neox_2x2(
            query,  # ty: ignore[invalid-argument-type]
            rot,
            self.head_size,
        )
        out_key = (
            _rotate_neox_2x2(
                key,  # ty: ignore[invalid-argument-type]
                rot,
                self.head_size,
            )
            if key is not None
            else None
        )
        return out_query, out_key


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(_SpyreRotaryMixin, RotaryEmbedding):
    """OOT RotaryEmbedding that applies the rotation on Spyre."""

    pass


@RotaryEmbeddingBase.register_oot(name="Llama3RotaryEmbedding")
class SpyreLlama3RotaryEmbedding(_SpyreRotaryMixin, Llama3RotaryEmbedding):
    """OOT Llama3RotaryEmbedding that applies the rotation on Spyre."""

    pass


@RotaryEmbeddingBase.register_oot(name="YaRNScalingRotaryEmbedding")
class SpyreYaRNScalingRotaryEmbedding(_SpyreRotaryMixin, YaRNScalingRotaryEmbedding):
    """OOT YaRNScalingRotaryEmbedding that applies the rotation on Spyre."""

    pass
