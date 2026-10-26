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

"""Expert biases on the generic Spyre MoE backend.

A biased layer carries ``w13_bias`` [E, 2M] (gate half, then up) and ``w2_bias`` [E, H]. The
relayout places them next to the weight pools as [E, 1, F] stacks, and every expert form adds
them: gate/up after their matmuls, down after the down matmul, the down bias weighted by
routing like the rest of the expert output. The reference is a plain silu MoE with biases, so
a missing, misplaced or unpadded bias shows up as a value error. The hidden width is 45 sticks,
so the gathered form splits the down bias into chunks; the intermediate width is stick-aligned
at TP1 and lands mid-stick at TP4, so the biases are zero-padded there.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

EXPERTS, HIDDEN, TOP_K = 16, 2880, 4


def _reference(x, logits, w13, w13_bias, w2, w2_bias, top_k):
    """Biased silu MoE in float32 on vLLM's parameter layout: ``w13`` [E, 2M, H] as
    ``[gate | up]``, ``w13_bias`` [E, 2M], ``w2`` [E, H, M], ``w2_bias`` [E, H]."""
    weights, indices = torch.topk(torch.softmax(logits.float(), dim=-1), top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    inter = w2.shape[-1]
    out = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token].float()
        for slot in range(top_k):
            e = int(indices[token, slot])
            gate_up = w13[e].float() @ row + w13_bias[e].float()
            activated = F.silu(gate_up[:inter]) * gate_up[inter:]
            hidden = w2[e].float() @ activated + w2_bias[e].float()
            out[token] += hidden * float(weights[token, slot])
    return out


def _checkpoint_like(inter, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.float16)

    return {
        "w13": randn(EXPERTS, 2 * inter, HIDDEN, scale=0.02),
        "w13_bias": randn(EXPERTS, 2 * inter, scale=0.5),
        "w2": randn(EXPERTS, HIDDEN, inter, scale=0.02),
        "w2_bias": randn(EXPERTS, HIDDEN, scale=0.5),
    }


# Full-softmax routing, as a biased model adapter requests it (with its own routing function).
_RECIPE = ("silu", "full_softmax")


def _routing(hidden_states, gating_output, topk, renormalize):
    weights, ids = torch.topk(torch.softmax(gating_output.float(), dim=-1), topk, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return weights.to(gating_output.dtype), ids.to(torch.int32)


def _biased_layer():
    """An upstream ``RoutedExperts`` with expert biases, as the backend would claim it."""
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
        "has_bias": True,
    }
    return SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(enable_eplb=False), **moe),
        custom_routing_function=_routing,
        activation=SimpleNamespace(value="silu"),
        global_num_experts=EXPERTS,
        local_num_experts=EXPERTS,
        renormalize=True,
        apply_router_weight_on_input=False,
        quant_config=None,
        quant_method=object.__new__(SpyreUnquantizedFusedMoEMethod),
    )


def test_configure_accepts_a_biased_layer():
    from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

    layer = _biased_layer()
    configure_spyre_moe_layer(layer, SpyreMoERecipe(*_RECIPE))
    assert layer.spyre_moe_regions == {}


def test_post_load_builds_the_quant_config_after_the_biases_are_freed(monkeypatch):
    """Upstream's quant config reads ``w13_bias`` / ``w2_bias``, which the relayout frees: a
    biased layer must still get one at load, or the first traced call builds it."""
    from types import MethodType

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    from spyre_inference import moe as moe_module

    monkeypatch.setattr(moe_module, "_prepare_layer", lambda layer: None)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace(has_bias=True)
    method.moe_quant_config = None
    layer = SimpleNamespace(spyre_moe_recipe=object(), quant_method=method)
    layer._ensure_moe_quant_config_init = MethodType(
        RoutedExperts._ensure_moe_quant_config_init, layer
    )

    method.process_weights_after_loading(layer)
    assert method.moe_quant_config is not None


class _RoutedExperts(torch.nn.Module):
    """Stands in for vLLM's, which needs a whole FusedMoEConfig to build."""

    def __init__(self, params, has_bias=True):
        super().__init__()
        for key, value in params.items():
            if has_bias or not key.endswith("_bias"):
                name = key if key.endswith("_bias") else f"{key}_weight"
                param = torch.nn.Parameter(value.clone(), requires_grad=False)
                self.register_parameter(name, param)
        self.top_k = TOP_K


def _prepared(inter, monkeypatch, has_bias=True):
    from spyre_inference.moe import SpyreMoERecipe, _prepare_layer

    monkeypatch.delenv("SPYRE_MOE_CHUNKS", raising=False)
    params = _checkpoint_like(inter)
    layer = _RoutedExperts(params, has_bias=has_bias)
    layer.spyre_moe_recipe = SpyreMoERecipe(*_RECIPE)
    _prepare_layer(layer)
    return layer, params


# Stick-aligned at TP1; mid-stick at TP4 (720 -> 768 padded) and TP8 (360 -> 384 padded).
_INTERS = [pytest.param(2880, id="tp1"), pytest.param(720, id="tp4"), pytest.param(360, id="tp8")]


@pytest.mark.parametrize("inter", _INTERS)
def test_relayout_places_the_biases_next_to_the_pools(inter, monkeypatch):
    """Gate/up biases split like ``w13``, zero in the padded lanes; the down bias whole and
    in the gathered form's chunks; the source parameters freed."""
    from torch_spyre._C import get_elem_in_stick

    layer, p = _prepared(inter, monkeypatch)
    stick = get_elem_in_stick(torch.float16)
    width = inter + -inter % stick
    assert not hasattr(layer, "w13_bias") and not hasattr(layer, "w2_bias")

    close = {"atol": 1e-4, "rtol": 1e-2}
    b = layer.spyre_moe_biases
    gate_b, up_b = b.gate.cpu(), b.up.cpu()
    assert gate_b.shape == up_b.shape == (EXPERTS, 1, width)
    torch.testing.assert_close(gate_b[:, 0, :inter], p["w13_bias"][:, :inter], **close)
    torch.testing.assert_close(up_b[:, 0, :inter], p["w13_bias"][:, inter:], **close)
    torch.testing.assert_close(b.down.cpu()[:, 0], p["w2_bias"], **close)
    chunks = layer.spyre_moe_chunks
    torch.testing.assert_close(
        b.down_chunks.cpu(), p["w2_bias"].reshape(EXPERTS * chunks, 1, HIDDEN // chunks), **close
    )
    for padded in (gate_b[..., inter:], up_b[..., inter:]):
        assert not padded.count_nonzero()


def test_relayout_without_biases_keeps_none(monkeypatch):
    layer, _ = _prepared(720, monkeypatch, has_bias=False)
    assert layer.spyre_moe_biases is None


def _inputs(tokens, seed):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(tokens, HIDDEN, generator=gen) * 0.5).to(torch.float16)
    logits = torch.randn(tokens, EXPERTS, generator=gen)
    # Keep the top-k boundary clear of fp16 near-ties, which the card may break either way.
    order = torch.sort(logits, dim=-1, descending=True).indices
    lift = torch.zeros_like(logits).scatter_(-1, order[:, :TOP_K], 0.25)
    return x, (logits + lift).to(torch.float16)


@pytest.mark.parametrize("inter", _INTERS)
def test_gathered_matches_the_biased_reference(inter, monkeypatch):
    """One decode token through the gathered region the dispatcher picks for it."""
    from contextlib import nullcontext

    from spyre_inference import moe as moe_module

    layer, p = _prepared(inter, monkeypatch)
    layer.spyre_moe_regions = {}
    monkeypatch.setattr(moe_module, "_compiler_scopes", lambda: (nullcontext(), nullcontext()))
    x, logits = _inputs(1, seed=inter)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    actual = method.apply_monolithic(layer, x.to("spyre"), logits.to("spyre"))
    assert set(layer.spyre_moe_regions) == {"gathered"}
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("inter", _INTERS)
@pytest.mark.parametrize("tokens", [16, 64])
def test_persistent_matches_the_biased_reference(inter, tokens, monkeypatch):
    """A prefill-sized batch through the all-expert regions, as ``apply_monolithic`` runs them."""
    from spyre_inference import moe as moe_module

    layer, p = _prepared(inter, monkeypatch)
    layer.spyre_moe_regions = {}
    x, logits = _inputs(tokens, seed=tokens)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    actual = method.apply_monolithic(layer, x.to("spyre"), logits.to("spyre"))
    assert "experts" in layer.spyre_moe_regions
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("inter", [pytest.param(720, id="tp4")])
def test_traced_dispatch_matches_the_biased_reference(inter, monkeypatch):
    """Traced into one graph with in-graph producers, as a compiled block runs one stick of
    tokens: the in-graph persistent form, biases included."""
    import warnings

    from torch_spyre._C import get_elem_in_stick
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference import moe as moe_module

    layer, p = _prepared(inter, monkeypatch)
    tokens = get_elem_in_stick(torch.float16)
    x, logits = _inputs(tokens, seed=1)

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
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)
