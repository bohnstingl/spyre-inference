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

"""Spyre OOT replacement for ParallelLMHead.

The matmul itself is delegated to upstream's gemm dispatcher; the only
Spyre-specific concern is wrapping it in an opaque custom op so the
`.to(...)` device transfers (the spyre `torch.Tensor.to` monkey-patch is
opaque to Dynamo) never enter a torch.compile graph.

Constraints:
    - tp_size > 1 not supported.
    - quant_config != None not supported.
"""

import torch
from functools import lru_cache

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from .utils import convert, register_layer, get_layer

logger = init_logger(__name__)


class SpyreUnquantizedLMHeadMethod(UnquantizedEmbeddingMethod):
    """Reroutes LogitsProcessor's lm_head.quant_method.apply call through
    SpyreParallelLMHead.forward_oot, so the matmul lands inside the opaque
    custom op (instead of running directly via the upstream dispatcher,
    which would leak `.to(...)` calls to Dynamo).
    """

    def apply(self, layer, x, bias=None):
        return layer.forward_oot(x, bias)

@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """OOT ParallelLMHead that executes the lm_head matmul on Spyre."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        quant_config = kwargs.get("quant_config")
        if quant_config is not None:
            raise NotImplementedError(
                "SpyreParallelLMHead does not support quantization "
                f"(quant_config={quant_config}). Only quant_config=None is supported."
            )

        if self.tp_size > 1:
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support Tensor Parallelism "
                f"(tp_size={self.tp_size}). Only tp_size=1 is supported."
            )

        self.quant_method = SpyreUnquantizedLMHeadMethod()
        self._layer_name = register_layer(self, "spyre_lm_head")

    def forward_oot(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """OOT forward pass — routes through the opaque Spyre custom op."""
        return UnquantizedEmbeddingMethod.apply(self.quant_method, self, x, bias)

@lru_cache(maxsize=1)
def register():
    """Register the spyre_lm_head custom op with vLLM."""
    pass
