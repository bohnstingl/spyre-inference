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

"""Compile upstream ``SiluAndMulWithClamp`` when it is outside a block graph.

MiniMax-M3's shared expert runs it inside the opaque ``moe_forward_shared`` op for
batches that are not a whole number of sticks. Run eagerly there, its ``x[..., :d]``
slice comes back in a device layout the next eager op cannot read.

References:
    - Upstream SiluAndMulWithClamp: vllm/model_executor/layers/activation.py
"""

import torch
from vllm.model_executor.layers.activation import SiluAndMulWithClamp

from .lazy_compile import CompileOutermost, maybe_compile


@SiluAndMulWithClamp.register_oot(name="SiluAndMulWithClamp")
class SpyreSiluAndMulWithClamp(CompileOutermost, SiluAndMulWithClamp):
    """Out-of-tree (OOT) SiluAndMulWithClamp implementation for IBM's Spyre."""

    @maybe_compile(force=True)
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """Run the unchanged vLLM native implementation."""
        return super().forward_native(x)
