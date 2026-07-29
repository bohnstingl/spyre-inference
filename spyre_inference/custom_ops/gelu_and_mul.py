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

"""Spyre-specific GeluAndMul implementation (GeGLU).

Gemma models use `gelu_pytorch_tanh` gated MLPs -> vLLM's `GeluAndMul`. The stock
`forward_native` slices the fused `[..., 2*d]` tensor on-device, which corrupts
Spyre memory (same hazard `SpyreSiluAndMul` works around). This mirrors that
override with GELU instead of SiLU.
"""

import torch

from vllm.model_executor.layers.activation import GeluAndMul


@GeluAndMul.register_oot(name="GeluAndMul")
class SpyreGeluAndMul(GeluAndMul):
    """Out-of-tree (OOT) GeluAndMul implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        """Initialize SpyreGeluAndMul layer."""
        super().__init__(*args, **kwargs)

        # With fullgraph compile enabled, the _forward will be compiled anyways
        if not torch.compiler.is_dynamo_compiling():
            self._forward = torch.compile(self.forward_native, dynamic=False)

    def forward_oot(self, x) -> torch.Tensor:
        """GeGLU: gelu(gate) * up, output shape [..., d]."""

        if (x.shape[1] % 64) > 0:
            raise RuntimeError(
                f"SpyreGeluAndMul only works with inner dimension being divisible \
                    by 64, but got {x.shape[1]}"
            )

        return self._forward(x)
