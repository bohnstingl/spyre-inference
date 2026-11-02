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

"""The Muse Spark adapter's blocks against the checkpoint's reference math, on CPU.

transformers ships Muse Spark only from 5.18, past this repo's pin, so the references are
the HF modeling code's formulas written out (``modeling_muse_spark.py``, 5.22): the
input-scaled norm, the residual gate, the router and the experts with their post-expert norm.
"""

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.models import muse_spark as ms


def _hf_input_scaled_norm(x, w, eps):
    out = (x.float() * (1.0 + w).float()).type_as(x).float()
    return (out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)).type_as(x)


def _hf_router(hidden, weight, bias, top_k, multiplier, scale):
    scores = torch.sigmoid((hidden.float() @ weight.float().t()) * multiplier)
    ids = torch.topk(scores + bias, top_k, dim=-1, sorted=False)[1]
    w = scores.gather(1, ids)
    return ids, w / (w.sum(-1, keepdim=True) + 1e-15) * scale


def _hf_experts(x, gate_up, down, norm_weight, eps, ids, weights):
    """HF's loop: ``gate_up`` is ``[E, 2I, H]``, ``down`` ``[E, H, I]``."""
    out = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for k, e in enumerate(ids[token].tolist()):
            gate, up = F.linear(x[token], gate_up[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, down[e])
            out[token] += _hf_input_scaled_norm(y, norm_weight, eps) * weights[token, k]
    return out


def test_input_scaled_norm_matches_the_checkpoints_math(default_vllm_config):
    """``rms(x * (1 + w))``: the weight before the norm, not after it."""
    norm = ms._InputScaledRMSNorm(256, eps=1e-8)
    torch.nn.init.normal_(norm.weight, std=0.5)
    x = torch.randn(5, 256) * 30
    expected = _hf_input_scaled_norm(x, norm.weight.detach(), 1e-8)
    scaled = x * (1.0 + norm.weight.detach())
    torch.testing.assert_close(norm.norm.forward_native(scaled), expected, atol=1e-5, rtol=1e-5)


def test_residual_gate_is_norm_preserving():
    """alpha^2 + beta^2 = 1 (alpha is floored at sqrt(eps)), so unit-variance inputs stay so."""
    gate = ms._ResidualGate(512, temperature=0.3)
    torch.nn.init.normal_(gate.gate, std=0.3)
    gate.materialize()
    torch.testing.assert_close(gate.alpha**2 + gate.beta**2, torch.ones(512), atol=1e-5, rtol=0)
    stream, branch = torch.randn(2, 512), torch.randn(2, 512)
    w = gate.gate.detach() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1 + beta), min=1e-3))
    torch.testing.assert_close(gate(stream, branch), alpha * stream + beta * branch)


def _router_config(num_experts=64, top_k=8):
    class Config:
        output_multiplier = 0.177
        routed_scaling_factor = 1.0

    return ms._HostRouter(num_experts, top_k, 0, num_experts, Config())


def test_host_router_selects_with_the_bias_but_weights_without_it():
    router = _router_config()
    weight, bias = torch.randn(64, 128), torch.randn(64) * 0.3
    router.load_weight(weight)
    router.load_bias(bias)
    hidden = torch.randn(9, 128)
    ids, weights = _hf_router(hidden, weight, bias, 8, 0.177, 1.0)
    expected = torch.zeros(9, 64).scatter(1, ids, weights)
    torch.testing.assert_close(router.dense(hidden), expected)


def test_host_router_hands_each_rank_its_own_experts():
    """Rank r's table is the dense one's columns [r*E/TP, (r+1)*E/TP), one stick per entry."""
    router = _router_config()
    router.load_weight(torch.randn(64, 128))
    hidden = torch.randn(3, 128).half()
    dense = router.dense(hidden.float())
    router.first, router.count = 16, 16
    table = router.route(hidden)
    assert table.shape == (16, 3, ms._ROUTE_STICK)
    torch.testing.assert_close(table[..., 0].float(), dense[:, 16:32].t(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_local_experts_sum_to_the_checkpoints_moe(ranks):
    """Each rank's experts, normalized per route, partial-sum to HF's MoE output."""
    experts, hidden, inter, tokens, top_k = 8, 64, 32, 5, 3
    gate_up = torch.randn(experts, 2 * inter, hidden) * 0.2
    down = torch.randn(experts, hidden, inter) * 0.2
    norm_weight = torch.randn(hidden) * 0.5
    x = torch.randn(tokens, hidden)
    ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    weights = torch.rand(tokens, top_k)
    expected = _hf_experts(x, gate_up, down, norm_weight, 1e-8, ids, weights)

    w13, w2 = gate_up.transpose(1, 2), down.transpose(1, 2)
    dense = torch.zeros(tokens, experts).scatter(1, ids, weights)
    routes = dense.t().unsqueeze(-1).expand(-1, -1, ms._ROUTE_STICK)
    per_rank = experts // ranks
    total = sum(
        ms.local_experts(
            x,
            w13[r * per_rank : (r + 1) * per_rank],
            w2[r * per_rank : (r + 1) * per_rank],
            norm_weight,
            1e-8,
            routes[r * per_rank : (r + 1) * per_rank],
        )
        for r in range(ranks)
    )
    torch.testing.assert_close(total, expected, atol=1e-4, rtol=1e-4)


def test_checkpoint_names_map_onto_the_adapter():
    rest = "mlp.pre_expert_proj.weight"
    assert (
        ms._remap_layer_weight(f"model.layers.3.{rest}", rest) == "model.layers.3.mlp.pooler.weight"
    )
    rest = "mlp.experts.post_expert_norm.weight"
    assert (
        ms._remap_layer_weight(f"model.layers.3.{rest}", rest)
        == "model.layers.3.mlp.post_expert_norm_weight"
    )
    rest = "self_attn.o_proj.weight"
    assert ms._remap_layer_weight(f"model.layers.3.{rest}", rest) == f"model.layers.3.{rest}"
