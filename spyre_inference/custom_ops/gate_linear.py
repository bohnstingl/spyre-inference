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

"""Spyre OOT ``GateLinear``: MoE router logits stay in the model dtype.

Spyre cannot restickify fp32 (``spyre::ReStickifyOpHBM`` is unsupported for
IEEE_FP32); the routing reduction over them is promoted in ``moe`` instead.
"""

import torch
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


@GateLinear.register_oot(name="GateLinear")
class SpyreGateLinear(GateLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Cleared here rather than through the argument, to also catch positional callers.
        self.out_dtype = None
        # force_fp32_compute (Nemotron-H) stores an fp32 weight off Hopper/Blackwell, and
        # Spyre has no fp32 matmul; assigning .data keeps the parameter's weight_loader.
        dtype = torch.get_default_dtype()
        for param in (self.weight, getattr(self, "bias", None)):
            if isinstance(param, torch.nn.Parameter) and param.dtype != dtype:
                param.data = param.data.to(dtype)
