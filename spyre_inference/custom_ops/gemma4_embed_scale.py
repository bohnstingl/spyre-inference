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

Gemma-4 scales the token embeddings by ``normalizer = sqrt(hidden_size)`` in
``Gemma4SelfDecoderLayers.embed_input_ids``:

    return self.embed_tokens(input_ids) * self.normalizer

``normalizer`` is registered as a buffer on ``Gemma4Model`` but is *also* stored
on ``Gemma4SelfDecoderLayers`` as a plain (unregistered) tensor attribute — a
shared reference. When the Spyre model runner moves the model with
``model.to("spyre")``, ``nn.Module._apply`` replaces the parent's registered
buffer with a fresh device tensor but leaves the child's plain-attribute alias
pointing at the original **CPU** 0-d tensor. The forward multiply then feeds
torch-spyre a 0-d CPU operand, which becomes an untileable graph ``InputBuffer``
and trips ``propagate_layouts.py``:

    RuntimeError: TensorBox(StorageBox(InputBuffer(name=..., layout=FixedLayout(
        'cpu', torch.float16, size=[], stride=[])))) does not have FixedTiledLayout

A 0-d tensor operand that is already on the Spyre device lowers fine; only the
stale CPU operand fails. We sidestep the broken alias entirely by scaling with a
Python float, which lowers as ``aten.mul.Scalar`` (the scalar is folded into the
pointwise op, so there is no 0-d input buffer). This is numerically identical to
the original fp16 tensor multiply — ``float(normalizer)`` widens the same
fp16-rounded value that the tensor path used.

Gemma 1/2/3 register ``normalizer`` on the *same* class that performs the
multiply, so ``.to()`` moves it consistently and they do not need this patch.

References:
    - Upstream model: vllm/model_executor/models/gemma4.py (Gemma4SelfDecoderLayers)
    - Runner device move: spyre_inference/v1/worker/spyre_model_runner.py (load_model)
    - torch-spyre guard: torch_spyre/_inductor/propagate_layouts.py
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Cache the scalar so an on-device normalizer isn't re-synced per step.
        scale = self.__dict__.get("_spyre_normalizer_scale")
        if scale is None:
            scale = float(self.normalizer)
            self.__dict__["_spyre_normalizer_scale"] = scale
        return self.embed_tokens(input_ids) * scale

    embed_input_ids._spyre_scalar_patch = True
    Gemma4SelfDecoderLayers.embed_input_ids = embed_input_ids
    logger.info(
        "Patched Gemma4SelfDecoderLayers.embed_input_ids to scale embeddings by a "
        "Python float (avoids a stale 0-d CPU normalizer tensor on Spyre)."
    )


_patch_gemma4_embed_scale()
