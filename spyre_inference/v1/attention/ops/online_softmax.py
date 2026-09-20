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

"""The online-softmax accumulation the per-sequence page kernels share.

Each kernel builds its own scores -- that is where the layouts legitimately differ --
and then folds them in here, so the rescale-and-accumulate exists once. A divergence
between copies of it produces slightly wrong logits, which no shape assertion catches
and only an eval would notice.
"""

import torch

OnlineSoftmaxCarry = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def online_softmax_step(
    carry: OnlineSoftmaxCarry | None,
    scores: torch.Tensor,
    v_page: torch.Tensor,
) -> OnlineSoftmaxCarry:
    """Fold one block's scores into a running (max, denominator, numerator) carry.

    ``scores`` must already be final. The caller applies the scale, then the soft cap,
    then any ALiBi bias, then the additive mask, in that order: the cap precedes the
    bias so the positional term is not squashed by the tanh, and both precede the mask
    because ``tanh(finfo.min / cap) * cap`` is ``-cap`` rather than ``-inf`` and would
    un-mask a padded lane.

    Args:
        carry: The previous block's carry, or None for the first block of a Python-loop
            walk. A tiled walk passes a -inf/0/0 init instead, whose rescale is
            ``exp(-inf - max) = 0``, so both paths agree on the first trip; see
            `walk_tiles` for why that constant cannot be materialized inside the body.
        scores: [..., block_size] logits for this block, masked.
        v_page: This block's values, right-multiplying the probabilities.

    Returns:
        The new carry: running max, running denominator, running numerator.
    """
    scores_max = torch.amax(scores, dim=-1, keepdim=True)
    if carry is None:
        probs = torch.exp(scores - scores_max)
        return scores_max, probs.sum(dim=-1, keepdim=True), torch.matmul(probs, v_page)

    tile_max, tile_sum, tile_output = carry
    new_max = torch.maximum(tile_max, scores_max)
    rescale = torch.exp(tile_max - new_max)
    probs = torch.exp(scores - new_max)
    return (
        new_max,
        tile_sum * rescale + probs.sum(dim=-1, keepdim=True),
        tile_output * rescale + torch.matmul(probs, v_page),
    )
