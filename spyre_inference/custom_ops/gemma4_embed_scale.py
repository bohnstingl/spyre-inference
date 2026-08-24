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

"""Spyre compatibility patch for Gemma-4's embedding-scale multiply.

Gemma-4 scales embeddings by ``normalizer = sqrt(hidden_size)``, held as a buffer on
``Gemma4Model`` but aliased as a plain attribute on ``Gemma4SelfDecoderLayers``.
``model.to("spyre")`` moves the parent buffer but leaves the alias on the original 0-d
CPU tensor; torch-spyre cannot tile that operand (propagate_layouts: "does not have
FixedTiledLayout"). Scaling by a Python float instead lowers as ``aten.mul.Scalar``
with no 0-d input buffer, and is numerically identical.

The float must be precomputed in ``__init__``, not in ``forward``: ``forward`` is
``@support_torch_compile``, so a first-call ``float(self.normalizer)`` would run during
the Dynamo trace and lift the 0-d CPU tensor back into the graph.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _patch_gemma4_embed_scale() -> None:
    try:
        from vllm.model_executor.models.gemma4 import Gemma4SelfDecoderLayers
    except ImportError:
        # Gemma-4 not present in this vLLM build; nothing to patch.
        return

    if getattr(Gemma4SelfDecoderLayers.embed_input_ids, "_spyre_scalar_patch", False):
        return

    orig_init = Gemma4SelfDecoderLayers.__init__

    def patched_init(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        self._spyre_normalizer_scale = float(self.normalizer)

    def patched_embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self._spyre_normalizer_scale

    patched_embed_input_ids._spyre_scalar_patch = True
    # setattr with a non-constant name: dodges ty invalid-assignment and ruff B010.
    for name, fn in (("__init__", patched_init), ("embed_input_ids", patched_embed_input_ids)):
        setattr(Gemma4SelfDecoderLayers, name, fn)
    logger.info(
        "Patched Gemma4SelfDecoderLayers to scale embeddings by a precomputed "
        "Python float (avoids a stale 0-d CPU normalizer tensor on Spyre, in both "
        "eager and torch.compile modes)."
    )


_patch_gemma4_embed_scale()
