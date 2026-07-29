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

"""Spyre-specific SiluAndMul implementation"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul


logger = init_logger(__name__)


@SiluAndMul.register_oot(name="SiluAndMul")
class SpyreSiluAndMul(SiluAndMul):
    """Out-of-tree (OOT) SiluAndMul implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        """Initialize SpyreSiluAndMul layer."""
        super().__init__(*args, **kwargs)

        # With fullgraph compile enabled, the _forward will be compiled anyways
        if not torch.compiler.is_dynamo_compiling():
            self._forward = torch.compile(self.forward_native, dynamic=False)

    def forward_oot(self, x) -> torch.Tensor:
        """SwiGLU: silu(gate) * up, output shape [..., d]."""

        if (x.shape[1] % 64) > 0:
            raise RuntimeError(
                f"SpyreSiluAndMul only works with inner dimension being divisible \
                    by 64, but got {x.shape[1]}"
            )

        return self._forward(x)
