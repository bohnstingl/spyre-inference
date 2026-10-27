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

"""DSA top-k mask over a head-major latent cache, against the fp32 lightning indexer."""

from types import SimpleNamespace

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops.utils import convert, row_outermost_layout
from spyre_inference.v1.attention.ops.dsa_head_major import dsa_mask_head_major_kernel
from spyre_inference.v1.attention.ops.layout import INT32_ELEMS_PER_STICK, head_major_kv_layout

pytestmark = pytest.mark.attention

# GLM-5.x: 576-wide latent, 32 indexer heads of 128, top-2048.
LATENT, INDEX_HEADS, INDEX_DIM = 576, 32, 128
HEAD = LATENT + INDEX_DIM
BLOCK = 128
FLOOR = torch.finfo(torch.float16).min


@pytest.fixture()
def device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


def _indexer_case(kv_len: int, query_len: int, num_blocks: int):
    """A sequence's latent pages (indexer key in columns 576:704), index inputs and mask."""
    torch.manual_seed(0)
    keys = torch.randn(num_blocks * BLOCK, INDEX_DIM) * 0.5
    rows = torch.cat([torch.randn(num_blocks * BLOCK, LATENT) * 0.5, keys], dim=-1)
    # Pages out of order, and page 0 left unused, as vLLM's null block is.
    page_ids = torch.randperm(num_blocks) + 1
    cache = torch.zeros(num_blocks + 1, 1, BLOCK, HEAD)
    for i, page in enumerate(page_ids):
        cache[page, 0] = rows[i * BLOCK : (i + 1) * BLOCK]
    q = torch.randn(query_len + 1, INDEX_HEADS, INDEX_DIM) * INDEX_DIM**-0.5
    w = torch.zeros(query_len + 1, 64)
    w[:, :INDEX_HEADS] = torch.randn(query_len + 1, INDEX_HEADS) * INDEX_HEADS**-0.5

    q_pos = kv_len - query_len + torch.arange(query_len)
    valid = torch.arange(num_blocks * BLOCK)[None, :] <= q_pos[:, None]
    base = torch.where(valid, 0.0, FLOOR).view(query_len, num_blocks, BLOCK).transpose(0, 1)
    table = torch.zeros(num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    table[:, 0] = page_ids
    return cache, q, w, keys, valid, base.contiguous(), table


@pytest.mark.parametrize(
    ("kv_len", "query_len", "index_topk"),
    [
        (4096, 64, 2048),  # prefill past GLM-5's index_topk
        (6000, 1, 2048),  # decode, ragged last page
        (1500, 64, 2048),  # context below index_topk keeps every valid token
        (1024, 64, 64),
    ],
)
def test_dsa_mask_keeps_the_fp32_topk(device, kv_len, query_len, index_topk):
    """The mask keeps the fp32 indexer's top-k per query row, plus at most a few fp16
    near-ties at the k-th score, and never a masked (future or padded) position."""
    num_blocks = -(-kv_len // BLOCK)
    cache, q, w, keys, valid, base, table = _indexer_case(kv_len, query_len, num_blocks)

    scores = torch.einsum(
        "qh,qhk->qk", w[:query_len, :INDEX_HEADS], torch.relu(q[:query_len] @ keys.t())
    )
    scores = scores.masked_fill(~valid, float("-inf"))
    k = min(index_topk, kv_len)
    want = torch.zeros_like(valid).scatter(1, scores.topk(k, -1).indices, True) & valid

    layout = head_major_kv_layout(num_blocks + 1, BLOCK, HEAD, torch.float16)
    pages = cache.half().to(device, device_layout=layout).view(num_blocks + 1, BLOCK, HEAD)
    q_dev = convert(q.half(), device, device_layout=row_outermost_layout(q.shape, torch.float16))
    w_dev = convert(w.half(), device, device_layout=row_outermost_layout(w.shape, torch.float16))
    mask = torch.compile(dsa_mask_head_major_kernel, dynamic=False, fullgraph=True)(
        q_dev,
        w_dev,
        convert(torch.arange(query_len, dtype=torch.int32), device),
        pages,
        convert(table, device),
        convert(base.half(), device),
        num_blocks,
        query_len,
        LATENT,
        INDEX_HEADS,
        index_topk,
        16,
    )
    got = (mask.cpu().float() > -1).transpose(0, 1).reshape(query_len, -1)

    assert not (got & ~valid).any(), "a masked position was kept"
    # The scores are fp16 on the device, so tokens tied with the k-th fp32 score within
    # fp16 resolution may swap; anything further from the boundary must be decided right.
    kth = scores.topk(k, -1).values[:, -1:]
    finite = scores.masked_fill(~valid, 0.0)
    tol = 2e-3 * finite.abs().amax(-1, keepdim=True)
    assert not (got & (scores < kth - tol)).any(), "kept a token well below the k-th score"
    assert not (~got & valid & (scores > kth + tol)).any(), "dropped a token well above it"
    assert (want & ~got).sum(-1).max() <= max(2, k // 500)


# Host-side checks of the indexer projections and the mask dispatch; no card needed.

_TINY = dict(index_topk=256, index_n_heads=32, index_head_dim=128, qk_rope_head_dim=64)


def _tiny_indexer():
    from vllm.config import CacheConfig, get_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.deepseek_v2 import Indexer

    from spyre_inference.v1.attention.dsa import DsaStaging, SpyreDsaConfig, SpyreDsaIndexer

    hf = SimpleNamespace(**_TINY)
    vllm_config = get_current_vllm_config()
    prefix = "model.layers.0.self_attn.indexer"
    vllm_config.compilation_config.static_forward_context.pop(f"{prefix}.k_cache", None)
    indexer = Indexer(vllm_config, hf, 256, 128, None, CacheConfig(block_size=BLOCK), None, prefix)
    rope = get_rope(64, max_position=1024, is_neox_style=False)
    cfg = SpyreDsaConfig.from_hf(hf, key_offset=LATENT)
    staging = DsaStaging(rows=17, n_heads=32, head_dim=128, dtype=torch.float32)
    return indexer, rope, SpyreDsaIndexer(indexer, rope, cfg, staging)


def test_indexer_projections_match_upstream_scores(tp_group, fp32_model_dtype):
    """The key, query and head weights SpyreDsaIndexer stages give upstream's lightning
    indexer score, sum_h w_h * relu(q_h . k) * softmax_scale * n_heads**-0.5, in fp32."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.utils.torch_utils import set_default_torch_dtype

    torch.manual_seed(0)
    with set_default_torch_dtype(torch.float32):
        indexer, rope, spyre = _tiny_indexer()
        for layer in (indexer.wq_b, indexer.wk_weights_proj):
            layer.weight.data.normal_(std=0.05)
        indexer.k_norm.weight.data.uniform_(0.5, 1.5)
        indexer.k_norm.bias.data.normal_(std=0.1)
        fused = indexer.wk_weights_proj.weight.data.clone()
        for layer in (indexer.wq_b, indexer.wk_weights_proj):
            layer.quant_method.process_weights_after_loading(layer)
        spyre.split_projections()

        tokens = 12
        hidden, q_c = torch.randn(tokens, 256), torch.randn(tokens, 128)
        positions = torch.arange(tokens)
        k, (q_buf, w_buf) = spyre(hidden, q_c, positions)

    # Upstream Indexer.forward's generic branch, before the fp8 quantization.
    n, d, r = 32, 128, 64
    q = indexer.wq_b(q_c)[0].view(-1, n, d)
    kw = hidden @ fused.t()
    ref_k = torch.nn.functional.layer_norm(
        kw[:, :d], (d,), indexer.k_norm.weight, indexer.k_norm.bias, 1e-6
    )
    q_pe, k_pe = RotaryEmbedding.forward_native(
        rope, positions, q[..., :r], ref_k[:, :r].unsqueeze(1)
    )
    ref_q = torch.cat([q_pe.reshape(-1, n, r), q[..., r:]], dim=-1)
    ref_k = torch.cat([k_pe.reshape(-1, r), ref_k[:, r:]], dim=-1)
    ref_w = kw[:, d:] * d**-0.5 * n**-0.5
    want = torch.einsum("qh,qhk->qk", ref_w, torch.relu(ref_q @ ref_k.t()))

    torch.testing.assert_close(k, ref_k, atol=1e-5, rtol=1e-5)
    got = torch.einsum("qh,qhk->qk", w_buf[:tokens, :n], torch.relu(q_buf[:tokens] @ k.t()))
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert not w_buf[:, n:].any(), "the padded head-weight lanes must weigh nothing"


@pytest.fixture()
def dsa_impl(default_vllm_config, monkeypatch):
    """A DSA impl whose page-attention kernels are stubbed to return the mask they get."""
    from spyre_inference.v1.attention import dsa
    from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
        SpyreHeadMajorAttentionImpl,
    )

    monkeypatch.setattr(dsa, "_dsa_mask_compiled", lambda *a: torch.full_like(a[5], 7.0))
    monkeypatch.setattr(SpyreHeadMajorAttentionImpl, "_run_page_attn", lambda self, *a: a[5])
    monkeypatch.setattr(SpyreHeadMajorAttentionImpl, "_folded_pages", lambda self, k, v: (k, v))

    def make(staging):
        return dsa.SpyreDsaAttentionImpl(
            num_heads=4,
            head_size=HEAD,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            dsa=dsa.SpyreDsaConfig.from_hf(SimpleNamespace(**_TINY), key_offset=LATENT),
            dsa_staging=staging,
        )

    return make


def _page_attn(impl, metadata, index_inputs, num_blocks):
    """One sequence's walk; the stubbed kernel returns the mask stack it was handed."""
    impl._call = (metadata, index_inputs)
    pages, base = torch.zeros(1), torch.zeros(num_blocks, 1)
    return impl._run_page_attn(
        pages, metadata.row_table, pages, pages, None, base, num_blocks, 1, None, None
    )


def test_dsa_mask_dispatch(dsa_impl):
    """Walks within index_topk stay dense without scoring; longer ones take the full
    layer's mask, which the next shared layer reuses; a shared layer with no mask to
    reuse raises outside warmup recording."""
    from spyre_inference.v1.attention.dsa import DsaStaging

    full = dsa_impl(DsaStaging(17, 32, 128, torch.float16))
    shared = dsa_impl(None)
    inputs = (torch.zeros(1), torch.zeros(1))
    metadata = SimpleNamespace(dsa_mask_stacks={}, row_table=torch.zeros(1))
    short = _TINY["index_topk"] // full.block_size

    assert not _page_attn(full, metadata, inputs, short).any()
    assert not metadata.dsa_mask_stacks, "a dense walk must not score"
    assert (_page_attn(full, metadata, inputs, short + 1) == 7).all()
    assert (_page_attn(shared, metadata, None, short + 1) == 7).all()

    fresh = SimpleNamespace(dsa_mask_stacks={}, row_table=torch.zeros(1))
    with pytest.raises(RuntimeError, match="sharing the top-k"):
        _page_attn(shared, fresh, None, short + 1)
    shared._recording = True
    assert not _page_attn(shared, fresh, None, short + 1).any()
