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

"""The Nemotron-H recipe: non-gated relu2 experts behind a sigmoid_bias router.

The reference is the transformers ``NemotronHExperts`` / ``NemotronHTopkRouter`` math on
vLLM's parameter layout, ``w13`` [E, M, H] holding the up projection alone. Shapes keep
Nemotron-3-Super's latent width (1024, the experts' hidden) and its per-rank expert width at
TP1 and TP8; the expert count is cut.
"""

from types import SimpleNamespace

import pytest
import torch

EXPERTS, HIDDEN, TOP_K = 16, 1024, 4


def _reference(x, logits, bias, w13, w2, top_k, scale):
    """transformers NemotronH routed experts in float32, routed scale applied."""
    scores = torch.sigmoid(logits.float())
    _, indices = torch.topk(scores + bias.float(), top_k, dim=-1)
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(-1, keepdim=True) * scale
    out = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token].float()
        for slot in range(top_k):
            e = int(indices[token, slot])
            hidden = w2[e].float() @ torch.square(torch.relu(w13[e].float() @ row))
            out[token] += hidden * float(weights[token, slot])
    return out


def _checkpoint_like(inter, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.float16)

    return {"w13": randn(EXPERTS, inter, HIDDEN, scale=0.05), "w2": randn(EXPERTS, HIDDEN, inter, scale=0.05)}


def _nemotron_layer(**overrides):
    """A ``RoutedExperts`` as vLLM builds it for Nemotron-H's MoE."""
    from spyre_inference.moe import SpyreUnquantizedFusedMoEMethod

    moe = {
        "num_logical_experts": EXPERTS,
        "num_experts": EXPERTS,
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pcp_size": 1,
        "sp_size": 1,
        "is_lora_enabled": False,
        "has_bias": False,
    } | overrides
    return SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(enable_eplb=False), **moe),
        custom_routing_function=None,
        scoring_func="sigmoid",
        e_score_correction_bias=torch.zeros(EXPERTS),
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        activation=SimpleNamespace(value="relu2_no_mul"),
        global_num_experts=EXPERTS,
        local_num_experts=EXPERTS,
        renormalize=True,
        apply_router_weight_on_input=False,
        quant_config=None,
        quant_method=object.__new__(SpyreUnquantizedFusedMoEMethod),
    )


def _recipe(scale=1.0):
    from functools import partial

    from spyre_inference.models.nemotron_h import _scale_down_weight
    from spyre_inference.moe import SpyreMoERecipe

    fold = partial(_scale_down_weight, scale=scale) if scale != 1.0 else None
    return SpyreMoERecipe(activation="relu2", routing="sigmoid_bias", prepare_down_weight=fold)


def test_configure_accepts_the_non_gated_relu2_layer():
    from spyre_inference.moe import configure_spyre_moe_layer

    layer = _nemotron_layer()
    recipe = _recipe()
    configure_spyre_moe_layer(layer, recipe)
    assert layer.spyre_moe_recipe is recipe


def test_configure_rejects_a_biased_relu2_layer():
    """No Nemotron-H expert has a bias; a layer with one would load and drop it."""
    from spyre_inference.moe import configure_spyre_moe_layer

    with pytest.raises(NotImplementedError, match="relu2 experts take no bias"):
        configure_spyre_moe_layer(_nemotron_layer(has_bias=True), _recipe())


def test_relu2_recipe_refuses_a_gated_activation():
    """A gated layer (silu, w13 = [gate | up]) under the relu2 recipe would read gate as up."""
    from spyre_inference.moe import configure_spyre_moe_layer

    layer = _nemotron_layer()
    layer.activation = SimpleNamespace(value="silu")
    with pytest.raises(NotImplementedError, match="requires activation='relu2'"):
        configure_spyre_moe_layer(layer, _recipe())


def test_adapter_folds_the_routed_scale_into_the_down_stack(monkeypatch):
    """Nemotron-H's decoder never compensates MoERunner's fp16 shared-expert division."""
    from spyre_inference.models import nemotron_h

    recipes = []
    monkeypatch.setattr(nemotron_h, "configure_spyre_moe_layer", lambda _, r: recipes.append(r))
    runner = SimpleNamespace(routed_scaling_factor=5.0, routed_experts=object())
    moe = object.__new__(nemotron_h.NemotronHMoE)
    moe.__dict__["experts"] = runner
    decoder = SimpleNamespace(mixer=moe)

    assert nemotron_h.configure_nemotron_h_moe_layers([decoder, SimpleNamespace()]) == 1
    (recipe,) = recipes
    assert runner.routed_scaling_factor == 1.0
    assert (recipe.activation, recipe.routing) == ("relu2", "sigmoid_bias")
    down = torch.randn(4, 8, 2)
    torch.testing.assert_close(recipe.prepare_down_weight(down), down * 5.0)


def test_non_gated_override_reaches_only_the_moe_config_module():
    """FusedMoEConfig sees an XPU-like platform while the model builds, nothing else does."""
    from vllm.model_executor.layers.fused_moe import config as fused_moe_config
    from vllm.platforms import current_platform

    from spyre_inference.models.nemotron_h import _non_gated_experts_allowed

    with _non_gated_experts_allowed():
        assert fused_moe_config.current_platform.is_xpu()
        assert not current_platform.is_xpu()
        assert fused_moe_config.current_platform.device_type == current_platform.device_type
    assert fused_moe_config.current_platform is current_platform


class _RoutedExperts(torch.nn.Module):
    """Stands in for vLLM's, which needs a whole FusedMoEConfig to build."""

    def __init__(self, params, bias):
        super().__init__()
        for key, value in params.items():
            param = torch.nn.Parameter(value.clone(), requires_grad=False)
            self.register_parameter(f"{key}_weight", param)
        self.top_k = TOP_K
        self.e_score_correction_bias = bias


def _prepared(inter, monkeypatch, scale=1.0):
    from spyre_inference.moe import _prepare_layer

    monkeypatch.delenv("SPYRE_MOE_CHUNKS", raising=False)
    params = _checkpoint_like(inter)
    # Real Nemotron-3 biases sit on a large shared offset with a small spread.
    bias = torch.randn(EXPERTS, generator=torch.Generator().manual_seed(1)) * 0.02 + 6.0
    layer = _RoutedExperts(params, bias)
    layer.spyre_moe_recipe = _recipe(scale)
    _prepare_layer(layer)
    return layer, params, bias


# Nemotron-3-Super's experts: 2688 per rank at TP1, 336 at TP8 (padded to 384).
_INTERS = [pytest.param(2688, id="tp1"), pytest.param(336, id="tp8")]


@pytest.mark.parametrize("inter", _INTERS)
def test_relayout_keeps_the_up_projection_and_builds_no_gate(inter, monkeypatch):
    layer, p, _ = _prepared(inter, monkeypatch)

    assert layer.spyre_moe_gate is None and layer.spyre_moe_gate_alias is None
    up = layer.spyre_moe_up.cpu()
    torch.testing.assert_close(up[..., :inter], p["w13"].transpose(1, 2), atol=1e-4, rtol=1e-2)
    assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")


def _inputs(tokens, bias, seed):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(tokens, HIDDEN, generator=gen) * 0.5).to(torch.float16)
    logits = torch.randn(tokens, EXPERTS, generator=gen)
    # Keep the biased top-k boundary clear of fp16 near-ties, which the card may break either way.
    choice = torch.sigmoid(logits) + bias
    order = torch.sort(choice, dim=-1, descending=True).indices
    lift = torch.zeros_like(logits).scatter_(-1, order[:, :TOP_K], 1.0)
    return x, (logits + lift).to(torch.float16)


@pytest.mark.parametrize("inter", _INTERS)
def test_gathered_matches_the_nemotron_reference(inter, monkeypatch):
    """One decode token through the gathered region the dispatcher picks for it."""
    from contextlib import nullcontext

    from spyre_inference import moe as moe_module

    layer, p, bias = _prepared(inter, monkeypatch, scale=5.0)
    layer.spyre_moe_regions = {}
    monkeypatch.setattr(moe_module, "_compiler_scopes", lambda: (nullcontext(), nullcontext()))
    x, logits = _inputs(1, bias, seed=inter)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    actual = method.apply_monolithic(layer, x.to("spyre"), logits.to("spyre"))
    assert set(layer.spyre_moe_regions) == {"gathered"}
    expected = _reference(x, logits, bias, p["w13"], p["w2"], TOP_K, scale=5.0)
    # Compared at unit scale: fp16's error grows with the folded x5, as the output does.
    torch.testing.assert_close(actual.cpu().float() / 5.0, expected / 5.0, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("inter", _INTERS)
@pytest.mark.parametrize("tokens", [16, 64])
def test_persistent_matches_the_nemotron_reference(inter, tokens, monkeypatch):
    """A prefill-sized batch through the all-expert regions, as ``apply_monolithic`` runs them."""
    from spyre_inference import moe as moe_module

    layer, p, bias = _prepared(inter, monkeypatch)
    layer.spyre_moe_regions = {}
    x, logits = _inputs(tokens, bias, seed=tokens)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    actual = method.apply_monolithic(layer, x.to("spyre"), logits.to("spyre"))
    assert "experts" in layer.spyre_moe_regions
    expected = _reference(x, logits, bias, p["w13"], p["w2"], TOP_K, scale=1.0)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("inter", [pytest.param(336, id="tp8")])
def test_traced_dispatch_matches_the_nemotron_reference(inter, monkeypatch):
    """Traced into one graph with in-graph producers, as a compiled block runs one stick of
    tokens: the in-graph persistent form without a gate pool."""
    import warnings

    from torch_spyre._C import get_elem_in_stick
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference import moe as moe_module

    layer, p, bias = _prepared(inter, monkeypatch)
    tokens = get_elem_in_stick(torch.float16)
    x, logits = _inputs(tokens, bias, seed=1)

    def block(x, logits):
        return moe_module.SpyreUnquantizedFusedMoEMethod.apply_monolithic(
            None, layer, x * 1.0, logits * 1.0
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        compiled = torch.compile(block, backend="inductor", fullgraph=True, dynamic=False)
        actual = compiled(x.to("spyre"), logits.to("spyre"))
    fallbacks = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not fallbacks, f"the traced dispatch fell back to CPU: {fallbacks}"
    expected = _reference(x, logits, bias, p["w13"], p["w2"], TOP_K, scale=1.0)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)


def test_large_prefill_routes_every_row_on_all_512_experts(monkeypatch):
    """Past 256 rows torch-spyre's top-22-of-512 keep_by_index drops selections (some rows
    keep 2), and their renormalized weights turned NaN; routing in row blocks keeps 22."""
    from spyre_inference import moe as moe_module

    experts, top_k, tokens = 512, 22, 602
    layer = SimpleNamespace(
        top_k=top_k,
        spyre_moe_regions={},
        spyre_moe_selection_bias=(torch.randn(experts) * 0.02).half().to("spyre"),
    )
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(tokens, experts, generator=gen).half()
    probs = moe_module._chunked_sigmoid_bias_probs(layer, logits.to("spyre")).cpu().float()
    assert torch.isfinite(probs).all()
    assert ((probs > 0).sum(-1) == top_k).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(tokens), atol=2e-2, rtol=0)


def test_runner_keeps_large_sigmoid_bias_batches_out_of_the_block_graph():
    """In-graph routing cannot split rows, so those batches take the opaque op."""
    from spyre_inference import moe as moe_module

    runner = object.__new__(moe_module.SpyreMoERunner)
    runner.moe_config = SimpleNamespace(in_dtype=torch.float16)
    runner._shared_experts = None
    runner.__dict__["routed_experts"] = SimpleNamespace(spyre_moe_recipe=_recipe())
    runner._select_forward()
    assert runner._entry_for(256) is moe_module._moe_forward
    assert runner._entry_for(320) is torch.ops.vllm.moe_forward
