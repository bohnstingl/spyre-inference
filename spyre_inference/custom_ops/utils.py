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

"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _convert_op_func(
    tensor: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Opaque-op body: device/dtype conversion with the Spyre dtype detour.

    Hidden behind `torch.ops.vllm.spyre_convert` so the device transfers and
    the spyre `torch.Tensor.to` monkey-patch are not traced into outer
    torch.compile graphs (no DeviceCopy nodes leak into the Inductor IR).
    """
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype

    if tensor.device.type == target_device.type and tensor.dtype == target_dtype:
        raise RuntimeError(
            f"Trying to convert a tensor to the same device ({tensor.device.type}) "
            + f"and same dtype ({tensor.dtype}), should never happen!"
        )

    # Spyre requires CPU for dtype changes
    if tensor.device.type == "spyre" and tensor.dtype != target_dtype:
        tensor = tensor.to(device="cpu")

    if tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)

    if tensor.device.type != target_device.type:
        tensor = tensor.to(device=target_device)

    return tensor


def _convert_op_fake(
    tensor: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype
    return torch.empty(tensor.shape, dtype=target_dtype, device=target_device)


def convert(tensor, device=None, dtype=None):
    """Convert tensor device and/or dtype. No-op when both are None.

    Routes through the opaque custom op `torch.ops.vllm.spyre_convert` so the
    transfer is invisible to torch.compile / Dynamo. None tensors are
    short-circuited at the Python boundary because `infer_schema` does not
    accept Optional[Tensor] returns.

    Args:
        tensor: Input tensor, or None (passed through as None).
        device: Target device as `str` or `torch.device` (None = keep current).
        dtype: Target dtype (None = keep current).

    Returns:
        Converted tensor, or None if input is None.
    """
    if tensor is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    # Short-circuit a true no-op at the call site so Inductor never emits a
    # same-device/dtype spyre_convert FallbackKernel into the graph.
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype
    if tensor.device.type == target_device.type and tensor.dtype == target_dtype:
        return tensor
    return torch.ops.vllm.spyre_convert(
        tensor,
        device,  # ty: ignore[invalid-argument-type]
        dtype,  # ty: ignore[invalid-argument-type]
    )


class SpyreGatherEmbeddingMixin:
    """Move a large-vocab embedding `weight` to Spyre with the gather-optimal
    indirect-access layout.

    vLLM's `VocabParallelEmbedding` is a plain `nn.Module`, not an `nn.Embedding`,
    so torch-spyre's optimal-layout loader never gives its `weight` the
    indirect-access layout an on-device gather needs; a default-layout gather over
    a big vocab (e.g. Gemma-4's 262k rows) overflows the Spyre per-core span limit.
    This mixin intercepts the `.to(spyre)` recursion and DMAs `weight` with the
    indirect-access layout (vocab outermost, hidden split into 128-byte sticks) via
    torch-spyre's `_dma_to_spyre_indirect_access`, so the gather fits and the
    embedding runs fully on-device. Mix in before the vLLM base class.
    """

    def _apply(self, fn, recurse=True):
        weight = self._parameters.get("weight")
        if weight is None or weight.ndim != 2 or weight.device.type == "spyre":
            return super()._apply(fn, recurse=recurse)
        # `_apply` hides the destination device; probe `fn` on a scalar to detect Spyre.
        probe = fn(torch.zeros(1, dtype=weight.dtype))
        if probe.device.type != "spyre":
            return super()._apply(fn, recurse=recurse)

        from torch_spyre.model_utils import _dma_to_spyre_indirect_access

        # Hold `weight` out of the recursion (super() would give it the overflowing
        # default layout), move everything else, then DMA it with the indirect layout.
        weight = self._parameters.pop("weight")
        super()._apply(fn, recurse=recurse)
        dev = _dma_to_spyre_indirect_access(weight.data, target_dtype=probe.dtype)
        assert dev is not None, (
            f"{self.__class__.__name__}: hidden dim {weight.shape[1]} does not tile into "
            "Spyre sticks (D % elems_per_stick != 0); indirect-access layout unavailable."
        )
        self._parameters["weight"] = torch.nn.Parameter(dev, requires_grad=weight.requires_grad)
        return self


@lru_cache(maxsize=1)
def register():
    """Register the spyre_convert custom op with vLLM."""
    # CompositeExplicitAutograd so the op dispatches regardless of input device
    # (convert is called with both CPU and Spyre input tensors).
    direct_register_custom_op(
        op_name="spyre_convert",
        op_func=_convert_op_func,
        fake_impl=_convert_op_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_convert")
