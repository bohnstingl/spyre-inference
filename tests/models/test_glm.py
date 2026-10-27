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

"""GLM adapters: host-side checks that need no card."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from vllm.model_executor.models.glm4_moe import Glm4MoE

from spyre_inference.models import glm


def _moe_layer(scale: float):
    mlp = mock.MagicMock(spec=Glm4MoE)
    mlp.experts = SimpleNamespace(routed_scaling_factor=scale, routed_experts=object())
    return SimpleNamespace(mlp=mlp)


@pytest.mark.parametrize("fold", [True, False])
def test_routed_scale_folds_into_the_down_stack_only_when_asked(monkeypatch, fold):
    """GLM-4.7-Flash's decoder never compensates the fp16 shared-expert division, so its
    scale moves into the down weights; GLM-5.x's decoder does, so MoERunner keeps it."""
    recipes = []
    monkeypatch.setattr(glm, "configure_spyre_moe_layer", lambda _, r: recipes.append(r))
    layer = _moe_layer(1.8)

    assert glm.configure_glm_moe_layers([layer, SimpleNamespace()], fold_routed_scale=fold) == 1
    (recipe,) = recipes
    assert recipe.routing == "sigmoid_bias"
    if fold:
        assert layer.mlp.experts.routed_scaling_factor == 1.0
        down = torch.randn(4, 8, 2)
        torch.testing.assert_close(recipe.prepare_down_weight(down), down * 1.8)
    else:
        assert layer.mlp.experts.routed_scaling_factor == 1.8
        assert recipe.prepare_down_weight is None
