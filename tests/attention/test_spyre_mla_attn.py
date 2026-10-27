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

"""Spyre MLA on the host: the absorbed MQA form against upstream's expanded attention.

The latent attention call is replaced by a dense causal reference, so these check the
wrapper's projections, absorption and output, not the paged kernels.
"""

import pytest
import torch
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import split_kv_b_proj
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper

from spyre_inference.v1.attention.mla import SpyreMultiHeadLatentAttention

pytestmark = pytest.mark.attention

HEADS, NOPE, ROPE, V, KV_LORA, Q_LORA, HIDDEN = 4, 64, 64, 128, 128, 192, 256


def _modules(**overrides) -> MLAModules:
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.rotary_embedding import get_rope

    def linear(cls, i, o, name):
        return cls(i, o, bias=False, disable_tp=True, prefix=name)

    fields = dict(
        kv_a_layernorm=RMSNorm(KV_LORA),
        kv_b_proj=linear(ColumnParallelLinear, KV_LORA, HEADS * (NOPE + V), "kv_b_proj"),
        rotary_emb=get_rope(ROPE, max_position=256, is_neox_style=False),
        o_proj=linear(RowParallelLinear, HEADS * V, HIDDEN, "o_proj"),
        fused_qkv_a_proj=MergedColumnParallelLinear(
            HIDDEN, [Q_LORA, KV_LORA + ROPE], bias=False, disable_tp=True, prefix="qkv_a"
        ),
        kv_a_proj_with_mqa=None,
        q_a_layernorm=RMSNorm(Q_LORA),
        q_b_proj=linear(ColumnParallelLinear, Q_LORA, HEADS * (NOPE + ROPE), "q_b_proj"),
        q_proj=None,
        indexer=None,
        is_sparse=False,
        topk_indices_buffer=None,
    )
    return MLAModules(**(fields | overrides))


def _build(modules: MLAModules, prefix: str) -> SpyreMultiHeadLatentAttention:
    get_current_vllm_config().compilation_config.static_forward_context.pop(f"{prefix}.attn", None)
    layer = MultiHeadLatentAttentionWrapper(
        HIDDEN,
        HEADS,
        (NOPE + ROPE) ** -0.5,
        NOPE,
        ROPE,
        V,
        Q_LORA,
        KV_LORA,
        modules,
        CacheConfig(block_size=128),
        prefix=prefix,
    )
    assert isinstance(layer, SpyreMultiHeadLatentAttention)
    return layer


def _expanded_reference(layer, modules, weights, positions, hidden):
    """Upstream's MLA-off attention: kv_b_proj expanded per head, causal softmax."""
    q_c, kv_lora = (hidden @ weights["qkv_a"].t()).split([Q_LORA, KV_LORA + ROPE], dim=-1)
    q = (modules.q_a_layernorm.forward_native(q_c) @ weights["q_b_proj"].t()).view(
        -1, HEADS, NOPE + ROPE
    )
    kv_c, k_pe = kv_lora.split([KV_LORA, ROPE], dim=-1)
    kv = (modules.kv_a_layernorm.forward_native(kv_c) @ weights["kv_b_proj"].t()).view(
        -1, HEADS, NOPE + V
    )
    k_nope, v = kv.split([NOPE, V], dim=-1)
    q_nope, q_pe = q.split([NOPE, ROPE], dim=-1)
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    rope = modules.rotary_emb
    q_pe, k_pe = RotaryEmbedding.forward_native(rope, positions, q_pe, k_pe.unsqueeze(1))
    k = torch.cat([k_nope, k_pe.expand(-1, HEADS, -1)], dim=-1)
    q = torch.cat([q_nope, q_pe], dim=-1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * (NOPE + ROPE) ** -0.5
    tokens = hidden.shape[0]
    causal = torch.ones(tokens, tokens, dtype=torch.bool).tril()
    probs = scores.masked_fill(~causal, float("-inf")).softmax(-1)
    out = torch.einsum("hqk,khd->qhd", probs, v).reshape(tokens, HEADS * V)
    return out @ weights["o_proj"].t()


def _dense_latent_attention(scale):
    def attend(query, key, value):
        # One latent KV head: query [T, H, D], key and value [T, 1, D].
        scores = torch.einsum("qhd,kd->hqk", query, key[:, 0]) * scale
        tokens = query.shape[0]
        causal = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        probs = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        return torch.einsum("hqk,kd->qhd", probs, value[:, 0]).reshape(tokens, -1)

    return attend


def test_absorbed_mla_matches_expanded_attention(tp_group, monkeypatch, fp32_model_dtype):
    """Absorbing kv_b_proj into the query and output projections, and attending over the
    one latent head, gives upstream's expanded multi-head attention."""
    torch.manual_seed(0)
    modules = _modules()
    layer = _build(modules, "model.layers.0.self_attn")
    weights = {}
    for name in ("fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj"):
        proj = getattr(layer, name)
        proj.weight.data.normal_(std=0.05)
        weights["qkv_a" if name == "fused_qkv_a_proj" else name] = proj.weight.data.clone()
    for norm in (modules.q_a_layernorm, modules.kv_a_layernorm):
        norm.weight.data.uniform_(0.5, 1.5)
    for proj in (layer.fused_qkv_a_proj, layer.q_b_proj, layer.kv_b_proj, layer.o_proj):
        proj.quant_method.process_weights_after_loading(proj)
    layer.mla_attn.process_weights_after_loading(torch.float32)

    attn = layer.mla_attn
    monkeypatch.setattr(attn, "forward", _dense_latent_attention(attn.impl.scale))
    positions = torch.arange(12)
    hidden = torch.randn(12, HIDDEN)
    want = _expanded_reference(layer, modules, weights, positions, hidden)
    got = layer(positions, hidden)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_absorbed_weights_match_upstream_split(tp_group):
    """W_UK_T and W_UV come out of the stored Wᵀ in upstream's split_kv_b_proj layout."""
    torch.manual_seed(0)
    layer = _build(_modules(), "model.layers.1.self_attn")
    kv_b_proj = layer.kv_b_proj
    kv_b_proj.weight.data.normal_()
    w_uk, w_uv = split_kv_b_proj(kv_b_proj, torch.float32, KV_LORA, HEADS, NOPE, V)
    kv_b_proj.quant_method.process_weights_after_loading(kv_b_proj)
    layer.mla_attn.process_weights_after_loading(torch.float32)
    torch.testing.assert_close(layer.mla_attn.W_UK_T, w_uk.permute(1, 2, 0), rtol=0, atol=0)
    torch.testing.assert_close(layer.mla_attn.W_UV, w_uv.transpose(0, 1), rtol=0, atol=0)


def test_quantized_kv_b_proj_is_rejected(tp_group):
    """An fp8 kv_b_proj must not be absorbed from its raw codes without the scales."""
    modules = _modules()
    modules.kv_b_proj.quant_method = object()
    with pytest.raises(NotImplementedError, match="unquantized kv_b_proj"):
        _build(modules, "model.layers.2.self_attn")


def test_output_gate_is_rejected(tp_group):
    with pytest.raises(NotImplementedError, match="output gate"):
        _build(_modules(g_proj=torch.nn.Identity()), "model.layers.3.self_attn")
