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

"""The MiniMax-M3 adapter's host-side pieces, against transformers' MiniMaxM3VL math.

Routing reuses GLM's host router (``test_glm.py`` checks it against vLLM's fp32 router);
here it is checked against transformers' ``MiniMaxM3VLTopKRouter`` including the routed
scale, which the adapter folds into the down weights. The attention prologue is checked
against transformers' per-head Gemma norm and partial RoPE.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers.models.minimax_m3_vl")


def _hf_text_config(**overrides):
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )

    cfg = dict(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_local_experts=16,
        num_experts_per_tok=4,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 5e6, "partial_rotary_factor": 0.5},
        partial_rotary_factor=0.5,
        num_hidden_layers=1,
    )
    return MiniMaxM3VLTextConfig(**(cfg | overrides))


def test_host_router_times_folded_scale_matches_transformers_router():
    """Host routing table x routed scale == transformers' sigmoid + bias top-k, renormalized,
    then x routed_scaling_factor (MiniMaxM3VLSparseMoeBlock)."""
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLTopKRouter

    from spyre_inference.models.glm_cpu_router import HostRouter

    torch.manual_seed(0)
    cfg = _hf_text_config()
    hf = MiniMaxM3VLTopKRouter(cfg).float()
    torch.nn.init.normal_(hf.weight, std=0.1)
    # Real M3 biases sit on a large shared offset with a small spread.
    hf.e_score_correction_bias.copy_(torch.randn(cfg.num_local_experts) * 0.05 + 9.5)

    gate = SimpleNamespace(weight=hf.weight.detach().t().contiguous(), input_size=cfg.hidden_size)
    experts = SimpleNamespace(
        top_k=cfg.num_experts_per_tok,
        renormalize=True,
        num_expert_group=None,
        topk_group=None,
        e_score_correction_bias=hf.e_score_correction_bias,
    )
    router = HostRouter(experts, gate)
    router.capture()

    x = torch.randn(8, cfg.hidden_size)
    _, weights, ids = hf(x)
    expected = torch.zeros(8, cfg.num_local_experts).scatter(1, ids, weights)
    scale = 2.0
    torch.testing.assert_close(router.route(x) * scale, expected * scale)


def test_routed_scale_is_folded_into_the_down_weights():
    """The fold scales the source down stack once; MoERunner's own scale stays 1."""
    from spyre_inference.models.minimax_m3 import _scale_down_weight

    w2 = torch.randn(4, 8, 16)
    torch.testing.assert_close(_scale_down_weight(w2, 2.0), w2 * 2.0)


@pytest.mark.parametrize("num_tokens", [1, 7])
def test_prologue_matches_transformers_qk_norm_and_partial_rope(default_vllm_config, num_tokens):
    """The torch prologue that replaces upstream's fused CUDA op: per-head Gemma norm on q and
    k, then NeoX RoPE on the first 64 of 128 dims (forward_native on CPU)."""
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as hfm
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope

    from spyre_inference.models.minimax_m3 import _SpyreMiniMaxM3Attention

    torch.manual_seed(0)
    cfg = _hf_text_config()
    heads, kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q_size, kv_size = heads * head_dim, kv_heads * head_dim

    attn = object.__new__(_SpyreMiniMaxM3Attention)
    torch.nn.Module.__init__(attn)
    attn.num_heads, attn.num_kv_heads, attn.head_dim = heads, kv_heads, head_dim
    attn.q_size, attn.kv_size = q_size, kv_size
    attn.q_norm = GemmaRMSNorm(head_dim, eps=cfg.rms_norm_eps)
    attn.k_norm = GemmaRMSNorm(head_dim, eps=cfg.rms_norm_eps)
    for norm in (attn.q_norm, attn.k_norm):
        torch.nn.init.normal_(norm.weight, std=0.1)
    rope = get_rope(
        head_dim,
        max_position=4096,
        rope_parameters={"rope_theta": 5e6, "partial_rotary_factor": 0.5},
        dtype=torch.float32,
    )
    attn.rotary_emb = lambda pos, q, k: rope.forward_native(pos, q, k)

    positions = torch.randint(0, 4096, (num_tokens,))
    # A sparse layer's projection carries [index_q | index_k] after v; the prologue drops it.
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size + 2 * head_dim)
    q, k, v = attn._prologue(positions, qkv)

    ref_q_norm = hfm.MiniMaxM3VLRMSNorm(head_dim, eps=cfg.rms_norm_eps)
    ref_k_norm = hfm.MiniMaxM3VLRMSNorm(head_dim, eps=cfg.rms_norm_eps)
    ref_q_norm.weight.data.copy_(attn.q_norm.weight)
    ref_k_norm.weight.data.copy_(attn.k_norm.weight)
    hq, hk, hv, _ = qkv.split([q_size, kv_size, kv_size, 2 * head_dim], -1)
    hq = ref_q_norm(hq.view(1, num_tokens, heads, head_dim)).transpose(1, 2)
    hk = ref_k_norm(hk.view(1, num_tokens, kv_heads, head_dim)).transpose(1, 2)
    cos, sin = hfm.MiniMaxM3VLRotaryEmbedding(cfg)(hq, positions.view(1, -1))
    hq, hk = hfm.apply_rotary_pos_emb(hq, hk, cos, sin)

    # The two RoPE caches round theta**(-2i/d) differently (transformers in fp32).
    tol = dict(atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(q.view(num_tokens, heads, head_dim), hq[0].transpose(0, 1), **tol)
    torch.testing.assert_close(k.view(num_tokens, kv_heads, head_dim), hk[0].transpose(0, 1), **tol)
    torch.testing.assert_close(v, hv)
