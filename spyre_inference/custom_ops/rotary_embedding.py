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

"""Spyre OOT replacement for RotaryEmbedding (CPU fallback).

Routes execution through an opaque custom op so that the H2D / D2H `.to(...)`
calls (the spyre `torch.Tensor.to` monkey-patch is opaque to Dynamo) are
not traced by the outer torch.compile graph.

Remove this file once Spyre natively supports rotary embedding ops.
"""

import torch
from functools import lru_cache

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert, register_layer, get_layer

logger = init_logger(__name__)


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """OOT RotaryEmbedding that falls back to CPU execution.

    Keeps cos_sin_cache on CPU via an _apply no-op. Inputs are moved to
    CPU for computation, and outputs are copied back to the original device.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_name = register_layer(self, "spyre_rotary_embedding")

    def _apply(self, fn, recurse=True):
        # Keep cos_sin_cache on CPU so forward_native can use it directly.
        return self

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The opaque op cannot return Optional[Tensor] (infer_schema limitation),
        # so pass a placeholder when key is None and discard the second return.
        key_in = key if key is not None else query.new_empty(0)
        out_query, out_key = torch.ops.vllm.spyre_rotary_embedding(
            positions, query, key_in, self._layer_name
        )
        return out_query, (out_key if key is not None else None)

    # def _forward_spyre_impl(
    #     self,
    #     positions: torch.Tensor,
    #     query: torch.Tensor,
    #     key: torch.Tensor | None,
    # ) -> tuple[torch.Tensor, torch.Tensor | None]:
    #     """D2H -> CPU rotary -> H2D back to original device."""
    #     target_device = query.device
    #     target_dtype = query.dtype

    #     cpu_positions = convert(positions, device="cpu")
    #     cpu_query = convert(query, device="cpu")
    #     cpu_key = convert(key, device="cpu")

    #     result_query, result_key = RotaryEmbedding.forward_native(
    #         self,
    #         cpu_positions,
    #         cpu_query,
    #         cpu_key,
    #     )

    #     out_query = convert(result_query, device=target_device, dtype=target_dtype)
    #     out_key = (
    #         convert(result_key, device=target_device, dtype=target_dtype)
    #         if result_key is not None
    #         else None
    #     )
    #     return out_query, out_key


def _op_func(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom op implementation — runs outside torch.compile graph.

    `key` may be a 0-element placeholder when the caller has no key tensor
    (the op schema cannot express Optional[Tensor] as a return). In that
    case the placeholder is forwarded through and the caller discards it.
    """
    layer = get_layer(layer_name)
    has_key = key.numel() > 0

    target_device = positions.device
    target_dtype = query.dtype

    cpu_positions = convert(positions, device="cpu")
    cpu_query = convert(query, device="cpu")
    cpu_key = convert(key, device="cpu") if has_key else None

    result_query, result_key = layer.forward_native(
        cpu_positions,
        cpu_query,
        cpu_key,
    )

    out_query = convert(result_query, device=target_device, dtype=target_dtype)
    if has_key:
        out_key = convert(result_key, device=target_device, dtype=target_dtype)
    else:
        out_key = key
    return out_query, out_key


def _op_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake impl: shape-correct empty tensors for Dynamo tracing."""
    return torch.empty_like(query), torch.empty_like(key)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rotary_embedding custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_rotary_embedding",
        op_func=_op_func,
        fake_impl=_op_fake,
    )
    logger.info("Registered custom op: SpyreRotaryEmbedding")
