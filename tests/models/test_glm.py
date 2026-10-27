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

import torch

from spyre_inference.models.glm import compensate_fp16_residual_scale


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
    compensate_fp16_residual_scale(model, scale)

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
    compensate_fp16_residual_scale(model, 1.0)
    assert all(layer.post_attention_layernorm.variance_epsilon == 1e-5 for layer in model.layers)


def test_host_router_matches_vllms_fp32_router():
    """The TEMPORARY host router reproduces vLLM's fp32 noaux_tc router exactly, as the dense
    routing table the Spyre MoE's dense_routes recipe takes, from a gate stored as Wᵀ."""
    from vllm.model_executor.layers.fused_moe.router.cpu_router import _grouped_topk

    from spyre_inference.models.glm_cpu_router import HostRouter

    torch.manual_seed(0)
    tokens, hidden, num_experts, top_k = 6, 128, 64, 4
    weight = torch.randn(num_experts, hidden) * 0.1
    bias = torch.randn(num_experts) * 0.1 + 9.0
    gate = SimpleNamespace(weight=weight.t().contiguous().half(), input_size=hidden)
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

    logits = x.float() @ weight.half().float().t()
    ref_weights, ref_ids = _grouped_topk(logits, top_k, True, 1, 1, "sigmoid", 1.0, bias)
    expected = torch.zeros(tokens, num_experts).scatter(1, ref_ids.long(), ref_weights)
    torch.testing.assert_close(dense, expected)
    assert (dense > 0).sum(-1).eq(top_k).all()
    torch.testing.assert_close(dense.sum(-1), torch.ones(tokens))
