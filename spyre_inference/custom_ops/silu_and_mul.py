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

"""Spyre-specific SiluAndMul implementation using out-of-tree (OOT) registration.

This module provides a custom SiluAndMul (SwiGLU) activation layer for
IBM's Spyre device, replacing the upstream vLLM implementation from
vllm/model_executor/layers/activation.py when instantiated.

Architecture:
    - OOT Registration: @SiluAndMul.register_oot() replaces upstream at instantiation
    - forward_oot(): Entry point for OOT dispatch. When called inside an outer
      torch.compile graph, delegates to the opaque custom op
      torch.ops.vllm.spyre_siluandmul so that Dynamo never traces the
      device-transferring body. When called eagerly with Spyre inputs (no
      compile graph), runs _forward_spyre_impl directly because the custom op
      would otherwise need to fabricate an output via copy_, which Spyre
      cannot do for in-device tensors.
    - Custom Op Boundary: torch.ops.vllm.spyre_siluandmul is opaque to
      torch.compile and returns the result tensor directly (no mutates_args),
      avoiding any copy_ on the Spyre device.
    - CPU slicing workaround: Slice on CPU, transfer to Spyre separately to
      avoid memory corruption from slicing Spyre tensors directly.

Output Shape Note:
    input shape: [..., 2*d] -> output shape: [..., d]

References:
    - Upstream SiluAndMul: vllm/model_executor/layers/activation.py
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.activation import SiluAndMul
from functools import lru_cache

from .utils import convert, register_layer, get_layer

logger = init_logger(__name__)


@SiluAndMul.register_oot(name="SiluAndMul")
class SpyreSiluAndMul(SiluAndMul):
    """Out-of-tree (OOT) SiluAndMul implementation for IBM's Spyre device.

    This replaces the upstream vLLM SiluAndMul (vllm/model_executor/layers/activation.py)
    when instantiated, providing Spyre-specific optimizations and device handling.

    Computes: x -> silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2

    Preserves input dtype and device. Slices on CPU to avoid Spyre slicing bugs.
    """

    def __init__(self, *args, **kwargs):
        """Initialize SpyreSiluAndMul layer."""
        super().__init__(*args, **kwargs)
        self._layer_name = register_layer(self, "spyre_siluandmul")

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """OOT forward pass.

        Routes through the opaque custom op so that the device transfers in
        _forward_spyre_impl are never traced by Dynamo. The custom op returns
        the result tensor directly (no in-place copy_, which Spyre cannot
        execute between two Spyre tensors).

        Args:
            x: Input tensor of shape [..., 2*d]

        Returns:
            Activated output tensor of shape [..., d] with same device and dtype as input.
        """
        return torch.ops.vllm.spyre_siluandmul(x, self._layer_name)

    def _forward_spyre_impl(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Spyre device execution: CPU slicing workaround, kernel call.

        Computes silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2.

        Preserves the input's device and dtype. Slices on CPU to work around
        Spyre's slicing bug which causes memory corruption and crashes.

        Args:
            x: Input tensor of shape [..., 2*d] containing concatenated gate halves.

        Returns:
            Activated output tensor of shape [..., d] with same device and dtype as input.
        """
        return F.silu(x1) * x2


def _op_func(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Custom op implementation — runs outside torch.compile graph."""
    layer = get_layer(layer_name)
    
    # Slice and make contiguous before transferring to Spyre.
    # Non-contiguous slices get corrupted during transfer to Spyre!
    x_device = x.device
    x = convert(x, device="cpu")
    d = x.shape[-1] // 2
    x1 = x[..., :d].contiguous()
    x2 = x[..., d:].contiguous()

    x1 = convert(x1, device=x_device)
    x2 = convert(x2, device=x_device)
    
    return layer._forward_spyre_impl(x1, x2)


def _op_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Fake impl: shape-correct empty tensor for Dynamo tracing."""
    d = x.shape[-1] // 2
    return torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_siluandmul custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_siluandmul",
        op_func=_op_func,
        fake_impl=_op_fake,
    )
    logger.info("Registered custom op: SpyreSiluAndMul")
