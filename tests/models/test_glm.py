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
    mlp.gate = object()
    mlp.experts = SimpleNamespace(routed_scaling_factor=scale, routed_experts=object())
    return SimpleNamespace(mlp=mlp)


@pytest.mark.parametrize("fold", [True, False])
def test_routed_scale_folds_into_the_down_stack_only_when_asked(monkeypatch, fold):
    """GLM-4.7-Flash's decoder never compensates the fp16 shared-expert division, so its
    scale moves into the down weights; GLM-5.x's decoder does, so MoERunner keeps it."""
    recipes = []
    monkeypatch.setattr(glm, "configure_spyre_moe_layer", lambda _, r: recipes.append(r))
    monkeypatch.setattr(glm, "install_host_router", lambda e, g: None)
    layer = _moe_layer(1.8)

    assert glm.configure_glm_moe_layers([layer, SimpleNamespace()], fold_routed_scale=fold) == 1
    (recipe,) = recipes
    if fold:
        assert layer.mlp.experts.routed_scaling_factor == 1.0
        down = torch.randn(4, 8, 2)
        torch.testing.assert_close(recipe.prepare_down_weight(down), down * 1.8)
    else:
        assert layer.mlp.experts.routed_scaling_factor == 1.8
        assert recipe.prepare_down_weight is None


def test_host_router_matches_vllms_fp32_router():
    """The TEMPORARY host router reproduces vLLM's fp32 noaux_tc router exactly, as the dense
    routing table the Spyre MoE's dense_routes recipe takes."""
    from vllm.model_executor.layers.fused_moe.router.cpu_router import select_experts

    from spyre_inference.models.glm_cpu_router import HostRouter

    torch.manual_seed(0)
    tokens, hidden, num_experts, top_k = 6, 128, 64, 4
    weight = (torch.randn(num_experts, hidden) * 0.1).half()
    bias = torch.randn(num_experts) * 0.1 + 9.0
    gate = SimpleNamespace(weight=weight, input_size=hidden, output_size=num_experts)
    experts = SimpleNamespace(
        top_k=top_k,
        renormalize=True,
        num_expert_group=None,
        topk_group=None,
        e_score_correction_bias=bias,
    )
    router = HostRouter(experts, gate)
    router.capture()

    x = (torch.randn(tokens, hidden) * 0.5).half()
    dense = router.route(x)

    logits = x.float() @ weight.float().t()
    ref_weights, ref_ids = select_experts(
        x.float(),
        logits,
        top_k=top_k,
        use_grouped_topk=True,
        renormalize=True,
        topk_group=1,
        num_expert_group=1,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )
    expected = torch.zeros(tokens, num_experts).scatter(1, ref_ids.long(), ref_weights)
    torch.testing.assert_close(dense, expected)
    assert (dense > 0).sum(-1).eq(top_k).all()
    torch.testing.assert_close(dense.sum(-1), torch.ones(tokens))


@pytest.mark.parametrize("host_router", [True, False])
def test_host_router_switch_selects_the_routing_recipe(monkeypatch, host_router):
    """SPYRE_GLM_HOST_ROUTER picks the TEMPORARY host router (dense_routes) or the
    on-device sigmoid_bias router."""
    from spyre_inference import envs

    recipes, installed = [], []
    monkeypatch.setattr(glm, "configure_spyre_moe_layer", lambda _, r: recipes.append(r))
    monkeypatch.setattr(glm, "install_host_router", lambda e, g: installed.append(g))
    monkeypatch.setenv("SPYRE_GLM_HOST_ROUTER", str(int(host_router)))
    envs.clear_env_cache()
    try:
        glm.configure_glm_moe_layers([_moe_layer(2.5)], fold_routed_scale=False)
    finally:
        envs.clear_env_cache()
    assert recipes[0].routing == ("dense_routes" if host_router else "sigmoid_bias")
    assert len(installed) == int(host_router)


class _Norm:
    def __init__(self, eps: float) -> None:
        self.variance_epsilon = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)


def _model(num_layers: int, eps: float):
    layers = [
        SimpleNamespace(
            layer_idx=i, input_layernorm=_Norm(eps), post_attention_layernorm=_Norm(eps)
        )
        for i in range(num_layers)
    ]
    return SimpleNamespace(layers=layers, norm=_Norm(eps))


def test_fp16_residual_scale_keeps_every_stream_norm_exact():
    """DeepseekV2DecoderLayer divides the residual stream by the routed scale in fp16; with
    the epsilon compensated, each norm reading it returns what it would on the true stream.

    GLM-5.2's early stream is small enough (RMS about 0.007) that the uncompensated epsilon
    moves the logits by over a nat.
    """
    eps, scale = 1e-5, 2.5
    reference, model = _model(4, eps), _model(4, eps)
    glm.compensate_fp16_residual_scale(model, scale)

    stream = torch.randn(3, 6144, dtype=torch.float64) * 0.007
    for ref_layer, layer in zip(reference.layers, model.layers):
        torch.testing.assert_close(
            layer.post_attention_layernorm(stream / scale),
            ref_layer.post_attention_layernorm(stream),
        )
        if layer.layer_idx == 0:
            # Reads the embedding before the decoder scales it.
            assert layer.input_layernorm.variance_epsilon == eps
        else:
            torch.testing.assert_close(
                layer.input_layernorm(stream / scale), ref_layer.input_layernorm(stream)
            )
    torch.testing.assert_close(model.norm(stream / scale), reference.norm(stream))


def test_fp16_residual_scale_is_a_no_op_without_a_routed_scale():
    model = _model(2, 1e-5)
    glm.compensate_fp16_residual_scale(model, 1.0)
    assert all(layer.post_attention_layernorm.variance_epsilon == 1e-5 for layer in model.layers)


def test_mla_off_attention_matches_upstream_forward(tp_group, fp32_model_dtype):
    """The MLA-off prologue built with torch.cat gives upstream DeepseekV2Attention's
    q, k and v, which it writes into empty_like buffers Spyre cannot store."""
    from vllm.config import CacheConfig, get_current_vllm_config
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Attention
    from vllm.utils.torch_utils import set_default_torch_dtype

    config = SimpleNamespace(rms_norm_eps=1e-6, rope_parameters={"rope_type": "default"})
    get_current_vllm_config().compilation_config.static_forward_context.pop(
        "model.layers.0.self_attn.attn", None
    )
    torch.manual_seed(0)
    with set_default_torch_dtype(torch.float32):
        attn = DeepseekV2Attention(
            get_current_vllm_config(),
            config,
            256,
            4,
            64,
            64,
            128,
            None,
            128,
            max_position_embeddings=256,
            cache_config=CacheConfig(block_size=128),
            prefix="model.layers.0.self_attn",
        )
        for module in attn.modules():
            if hasattr(module, "quant_method") and hasattr(module, "weight"):
                module.weight.data.normal_(std=0.05)
                module.quant_method.process_weights_after_loading(module)

    class Recorder(torch.nn.Module):
        def forward(self, q, k, v):
            seen.append((q, k, v))
            return v

    seen = []
    attn.attn = Recorder()
    positions, hidden = torch.arange(6), torch.randn(6, 256)
    want = DeepseekV2Attention.forward(attn, positions, hidden, None)
    got = glm._spyre_mha_forward(attn, positions, hidden, None)
    torch.testing.assert_close(got, want)
    for ours, theirs in zip(seen[1], seen[0]):
        torch.testing.assert_close(ours.reshape(theirs.shape), theirs)
