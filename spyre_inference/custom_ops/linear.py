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

"""Spyre-specific linear layer implementations using out-of-tree (OOT) registration.

This module provides Spyre-device-specific replacements for the parallel linear
layer classes used inside MLP blocks:

    - SpyreMergedColumnParallelLinear  — replaces MergedColumnParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreQKVParallelLinear          — replaces QKVParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreRowParallelLinear          — replaces RowParallelLinear
      (vllm/model_executor/layers/linear.py)

At TP=1, the upstream forward() methods reduce to quant_method.apply() + bias
handling.  We inject a custom quant_method (SpyreUnquantizedLinearMethod) that
performs F.linear directly, QKV and RowParallel still override forward()
for device placement (D2H after GEMM, H2D before GEMM).

QKV's forward routes through the opaque custom op `torch.ops.vllm.spyre_qkv_linear`
so that the D2H `.to(device="cpu")` after the GEMM is not traced by Dynamo
(spyre's `torch.Tensor.to` monkey-patch is opaque to Dynamo).

Spyre Device Constraints:
    - Computations performed in torch.float16:
      Input (dtype defined by model / user) converted to torch.float16 for
      operations on spyre and then converted back to original dtype for cpu.
    - Tensor parallelism: TP=1 assumed (single Spyre device)

References:
    - Upstream linear layers:   vllm/model_executor/layers/linear.py
"""

import torch
import torch.nn.functional as F
from functools import lru_cache

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert, register_layer, get_layer

logger = init_logger(__name__)


class SpyreUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Spyre-specific linear method: F.linear without platform GEMM dispatch.

    Replaces the default UnquantizedLinearMethod so that upstream forward()
    methods work unchanged on Spyre at TP=1.

    - create_weights() is inherited — standard ModelWeightParameter works.
    - apply() does F.linear directly (no platform-specific GEMM dispatch).
    - process_weights_after_loading() is a no-op (skips CPU GEMM dispatch).
    """

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight.data, bias)

    def process_weights_after_loading(self, layer):
        pass


class SpyreLinearBase:
    """Shared initialization for Spyre linear layers at TP=1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tp_size > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports TP=1, got TP={self.tp_size}"
            )

        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = SpyreUnquantizedLinearMethod()


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class SpyreMergedColumnParallelLinear(SpyreLinearBase, MergedColumnParallelLinear):
    """Spyre MergedColumnParallelLinear (TP=1 only)."""


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(SpyreLinearBase, QKVParallelLinear):
    """Spyre QKVParallelLinear (TP=1 only).

    Routes through the opaque custom op `torch.ops.vllm.spyre_qkv_linear`
    so that the D2H `.to(device="cpu")` after the GEMM (needed because
    downstream `.split()` cannot operate on a strided Spyre view) is not
    traced by Dynamo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_name = register_layer(self, "spyre_qkv_linear")

    def forward(self, input_):
        out_dev = super().forward(input_)
        if self.return_bias:
            return torch.ops.vllm.spyre_qkv_linear(out_dev[0], self._layer_name), out_dev[1]
        return torch.ops.vllm.spyre_qkv_linear(out_dev, self._layer_name)

@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(SpyreLinearBase, RowParallelLinear):
    """Spyre RowParallelLinear (TP=1 only).
    RowParallelLinear is currently invoked from `GraniteAttention` where
    `input_` is on `cpu` and from `GraniteMLP` where `input_` is on spyre.
    Thus, we always convert the `input_` to `spyre`, which is a NoOp in
    case of `GraniteMLP`.
    """

    def forward(self, input_):
        return super().forward(convert(input_, device=self.weight.device))


def _qkv_op_func(input_: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Custom op: QKV linear forward (returns the bare output tensor).

    `_forward_spyre_impl` already returns just the output tensor (bias
    re-packing happens above the op boundary in `forward`).
    """
    return convert(input_, device="cpu")


def _qkv_op_fake(input_: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = get_layer(layer_name)
    out_shape = input_.shape[:-1] + (layer.output_size_per_partition,)
    return torch.empty(out_shape, dtype=input_.dtype, device="cpu")


@lru_cache(maxsize=1)
def register():
    """Register the spyre QKV linear custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_qkv_linear",
        op_func=_qkv_op_func,
        fake_impl=_qkv_op_fake,
    )
    logger.debug_once("Registered custom op: SpyreQKVParallelLinear")
