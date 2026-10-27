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

"""Multi-head latent attention (DeepSeek-V2 MLA) on the Spyre paged attention backend.

``SpyreMultiHeadLatentAttention`` replaces upstream's ``MultiHeadLatentAttentionWrapper``
(its ``PluggableLayer`` OOT hook). It keeps upstream's projections, RMSNorms, RoPE and
``kv_b_proj`` and computes MLA in upstream's weight-absorbed MQA form (see
``mla_attention.py``) for every token, prefill included:

    ql_nope = q_nope @ W_UK_T                  per head, [T, H, kv_lora_rank]
    o_lat   = attn([ql_nope | q_pe], K = V = [kv_c | k_pe])   one latent KV head
    o       = o_lat[..., :kv_lora_rank] @ W_UV

The attention itself is a plain ``Attention`` with one KV head, so it runs on the paged
Spyre backend, its KV store, buckets and warmup recorder unchanged. The page holds the
576-wide latent once as K; V is the same latent, stored again because the kernels read V
from its own pages (the latent is 1/16 of an expanded GLM-5 K+V even so). The upstream
``MLAAttention`` layer and its CUDA/CPU backends are not built.

DSA (``index_topk`` configs) reuses upstream's ``Indexer`` projections and applies the
selected top-k as an additive mask on the same attention; see
``spyre_inference.v1.attention.dsa``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import split_kv_b_proj
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

if TYPE_CHECKING:
    from vllm.config import CacheConfig
    from vllm.model_executor.layers.mla import MLAModules
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)

# One fp16 stick: the latent head (kv_lora_rank + qk_rope_head_dim) is padded to it.
_STICK = 64


class SpyreLatentAttention(Attention):
    """The latent MQA attention, which absorbs ``kv_b_proj`` once the weights are loaded.

    ``W_UK_T`` [H, P, L] and ``W_UV`` [H, L, V] are built from ``kv_b_proj`` here rather
    than in the wrapper so the runner's ``process_weights_after_loading`` pass (which
    visits attention layers after every linear layer) sees the transposed weight.
    """

    def __init__(self, *args, mla_dims: tuple[int, int, int, int], kv_b_proj, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_lora_rank, self.qk_nope_head_dim, self.v_head_dim, self.mla_num_heads = mla_dims
        # A plain attribute, not a submodule: kv_b_proj is registered on the wrapper.
        self.__dict__["_kv_b_proj"] = kv_b_proj
        self.W_UK_T: torch.Tensor | None = None
        self.W_UV: torch.Tensor | None = None

    def _apply(self, fn, recurse=True):
        # The runner pins Attention's own buffers (KV scales) to the host; the absorbed
        # projections are matmul operands and move with the model.
        for name in ("W_UK_T", "W_UV"):
            param = getattr(self, name)
            if param is not None:
                setattr(self, name, torch.nn.Parameter(fn(param.data), requires_grad=False))
        return self

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        kv_b_proj = self.__dict__["_kv_b_proj"]
        # SpyreTransposedWeightMethod stores kv_b_proj.weight as Wᵀ [in, out];
        # split_kv_b_proj expects upstream's [out, in] weight.
        weight = kv_b_proj.weight.data
        if weight.shape[0] == self.kv_lora_rank:
            weight = weight.t()
        holder = type("_W", (), {"weight": weight, "quant_method": None})()
        w_uk, w_uv = split_kv_b_proj(
            holder,  # ty: ignore[invalid-argument-type]
            act_dtype,
            self.kv_lora_rank,
            self.mla_num_heads,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )
        self.W_UK_T = torch.nn.Parameter(w_uk.permute(1, 2, 0).contiguous(), requires_grad=False)
        self.W_UV = torch.nn.Parameter(w_uv.transpose(0, 1).contiguous(), requires_grad=False)


@MultiHeadLatentAttentionWrapper.register_oot(name="MultiHeadLatentAttentionWrapper")
class SpyreMultiHeadLatentAttention(MultiHeadLatentAttentionWrapper):
    """MLA in its absorbed (MQA) form on a one-KV-head Spyre ``Attention``."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
        **_unused,
    ) -> None:
        # Skip MultiHeadLatentAttentionWrapper.__init__: it builds upstream's MLAAttention,
        # whose backend selection raises for sparse configs and has no Spyre kernel.
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.skip_topk = skip_topk
        self.g_proj = mla_modules.g_proj
        if self.g_proj is not None:
            raise NotImplementedError("Spyre MLA does not support the attention output gate.")

        latent = kv_lora_rank + qk_rope_head_dim
        attn_kwargs: dict = {}
        self.dsa = None
        if self.is_sparse:
            from vllm.config import get_current_vllm_config

            from spyre_inference.v1.attention.dsa import (
                DsaStaging,
                SpyreDsaAttentionBackend,
                SpyreDsaConfig,
                SpyreDsaIndexer,
            )

            vllm_config = get_current_vllm_config()
            dsa = SpyreDsaConfig.from_hf(vllm_config.model_config.hf_text_config, latent)
            # Every sparse layer is sized for the indexer key, so all of them share one
            # KV cache group, and with it the step's metadata the shared layers read.
            latent += dsa.head_dim
            staging = None
            if self.indexer is not None:
                assert self.indexer_rope_emb is not None
                staging = DsaStaging.shared(
                    vllm_config.scheduler_config.max_num_batched_tokens + 1,
                    dsa.n_heads,
                    dsa.head_dim,
                    vllm_config.model_config.dtype,
                )
                self.dsa = SpyreDsaIndexer(self.indexer, self.indexer_rope_emb, dsa, staging)
            attn_kwargs = {
                "attn_backend": SpyreDsaAttentionBackend,
                "dsa": dsa,
                "dsa_staging": staging,
            }
        self.latent_head_size = -(-latent // _STICK) * _STICK
        if self.fused_qkv_a_proj is not None:
            # A MergedColumnParallelLinear subclass the OOT linear registry does not name.
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod

            from spyre_inference.custom_ops.linear import SpyrePaddedRowsLinearMethod

            if isinstance(self.fused_qkv_a_proj.quant_method, UnquantizedLinearMethod):
                self.fused_qkv_a_proj.quant_method = SpyrePaddedRowsLinearMethod()
        self.mla_attn = SpyreLatentAttention(
            num_heads,
            self.latent_head_size,
            scale,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            mla_dims=(kv_lora_rank, qk_nope_head_dim, v_head_dim, num_heads),
            kv_b_proj=self.kv_b_proj,
            **attn_kwargs,
        )
        self.prefix = prefix

    def _pad_latent(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.latent_head_size - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad)) if pad else x

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None and self.q_b_proj is not None
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None and self.q_proj is not None
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        if llama_4_scaling is not None:
            q_nope = q_nope * llama_4_scaling
            q_pe = q_pe * llama_4_scaling

        latent_parts = [kv_c, k_pe.reshape(-1, self.qk_rope_head_dim)]
        attn_kwargs = {}
        if self.dsa is not None:
            assert q_c is not None
            index_k, attn_kwargs["extra_inputs"] = self.dsa(hidden_states, q_c, positions)
            latent_parts.append(index_k)

        attn = self.mla_attn
        assert attn.W_UK_T is not None and attn.W_UV is not None
        # [H, T, P] @ [H, P, L] -> [T, H, L]
        ql_nope = torch.matmul(q_nope.transpose(0, 1), attn.W_UK_T).transpose(0, 1)
        # Zero beyond the latent, so the indexer key adds nothing to the scores.
        mqa_q = self._pad_latent(torch.cat([ql_nope, q_pe], dim=-1))
        latent = self._pad_latent(torch.cat(latent_parts, dim=-1)).unsqueeze(1)
        o_lat = attn(mqa_q, latent, latent, **attn_kwargs)
        o_lat = o_lat.view(-1, self.num_heads, self.latent_head_size)[..., : self.kv_lora_rank]
        # [H, T, L] @ [H, L, V] -> [T, H * V]
        out = torch.matmul(o_lat.transpose(0, 1), attn.W_UV).transpose(0, 1)
        return self.o_proj(out.reshape(-1, self.num_heads * self.v_head_dim))[0]
