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

"""Per-block compile granularity: block discovery, in-place wrapping, artifact reuse.

Two model shapes are covered: the in-tree vLLM one, where the block owns its
``Attention`` (``_Model``), and the Transformers modeling backend's, where the
``Attention`` layers live in a plain dict until ``attach_attention_instances`` hangs them
off the HF modules that use them (``_HFModel``).

Only the artifact-reuse tests trace, and tracing resolves the accelerator stream,
which opens the single contested Spyre card -- hence their ``compile`` marker. The
rest is device-free and stays in the smoke job.
"""

from __future__ import annotations

import types
from typing import cast

import pytest
import torch
import torch.nn as nn
from vllm.config import CompilationMode
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.encoder_only_attention import EncoderOnlyAttention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.models.utils import PPMissingLayer

from spyre_inference.transformers_backend import attach_attention_instances
from spyre_inference.v1.worker.spyre_model_runner import (
    TorchSpyreModelRunner,
    _repeated_block_lists,
)


@pytest.fixture
def isolated_dynamo_state():
    """Bound this test's Dynamo cache churn; save and restore the global counters."""
    import copy

    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    saved = copy.deepcopy(counters)
    dynamo.reset()
    try:
        yield
    finally:
        dynamo.reset()
        counters.clear()
        counters.update(saved)


def _fake_layer(cls: type) -> nn.Module:
    """An ``AttentionLayerBase`` of *cls*, skipping an ``__init__`` that would need a full
    model config, a KV-cache group and a backend. Discovery only reads the class."""
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    return layer


def _fake_attention() -> Attention:
    """Skips ``Attention.__init__``, which needs a full model config."""
    return cast(Attention, _fake_layer(Attention))


def _stub_attention_forward(self, query, key, value):
    """Shape-faithful stand-in for paged attention: ``[L, H*D]`` in, ``[L, H*D]`` out."""
    return torch.softmax(query @ key.transpose(0, 1) * self.impl.scale, dim=-1) @ value


def _fake_attention_instance(scale: float) -> Attention:
    """A ``_fake_attention`` that can actually be called, the way the two Transformers
    backend interfaces call it: positionally, with flattened ``[L, H*D]`` tensors."""
    attn = _fake_attention()
    attn.impl = types.SimpleNamespace(scale=scale)  # ty: ignore[invalid-assignment]
    attn.forward = types.MethodType(_stub_attention_forward, attn)  # ty: ignore
    return attn


class _Block(nn.Module):
    """Threads a residual like a real decoder layer: ``residual is None`` on layer 0
    is a trace-time branch, so layer 0 compiles to its own graph."""

    def __init__(self, hidden: int, attn_cls: type = Attention):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn = _fake_layer(attn_cls)
        self.o_proj = nn.Linear(hidden, hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, 2 * hidden)
        self.down = nn.Linear(2 * hidden, hidden)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            residual = x + residual
            h = self.input_layernorm(residual)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        x = self.o_proj(torch.softmax(q * k, dim=-1) * v)
        h = self.post_attention_layernorm(x)
        return self.down(torch.relu(self.up(h))), residual


class _MambaBlock(nn.Module):
    """Attention-free layer, as in a hybrid Mamba+attention stack.

    *mixer_cls* gives it a real Mamba mixer, which is an ``AttentionLayerBase`` without
    being an ``Attention``; without one the block owns no per-layer state at all, the way
    a plain MLP-only layer does.
    """

    def __init__(self, hidden: int, mixer_cls: type | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, hidden)
        if mixer_cls is not None:
            self.mixer = _fake_layer(mixer_cls)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
        else:
            residual = x + residual
        return self.proj(self.norm(residual)), residual


class _Backbone(nn.Module):
    def __init__(
        self,
        hidden: int,
        num_layers: int,
        num_missing: int = 0,
        hybrid: bool = False,
        attn_cls: type = Attention,
        mixer_cls: type | None = None,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden)
        blocks: list[nn.Module] = []
        for i in range(num_layers):
            # mixer_cls without hybrid is a pure Mamba stack (Codestral Mamba); with it,
            # the mixer-bearing half of a real hybrid one.
            if mixer_cls is not None and not (hybrid and i % 2 == 0):
                blocks.append(_MambaBlock(hidden, mixer_cls))
            elif hybrid and i % 2:
                blocks.append(_MambaBlock(hidden))
            else:
                blocks.append(_Block(hidden, attn_cls))
        self.layers = nn.ModuleList(blocks + [PPMissingLayer() for _ in range(num_missing)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            x, residual = layer(x, residual)
        return self.norm(x + residual)


class _Model(nn.Module):
    def __init__(
        self,
        hidden: int = 32,
        num_layers: int = 4,
        num_missing: int = 0,
        hybrid: bool = False,
        attn_cls: type = Attention,
        mixer_cls: type | None = None,
    ):
        super().__init__()
        self.model = _Backbone(hidden, num_layers, num_missing, hybrid, attn_cls, mixer_cls)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


# --- The Transformers modeling backend's shape ------------------------------------------
#
# Everything above is the in-tree vLLM shape: the block owns its Attention. Upstream's
# Transformers backend instead keeps the Attention layers in a plain dict on the model and
# indexes it by ``layer_idx`` from inside the HF attention forward, so nothing under a
# decoder layer is an Attention and nothing about the lookup is depth-independent.


class _HFAttention(nn.Module):
    """The HF attention module, dispatching the way every ``modeling_*.py`` does: look the
    interface up by ``config._attn_implementation`` and call it with ``self`` as *module*."""

    def __init__(self, hidden: int, num_heads: int, layer_idx: int, config):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.o_proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, attention_instances: dict) -> torch.Tensor:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        shape = (*x.shape[:-1], self.num_heads, self.head_dim)
        # [B, L, hidden] -> [B, H, L, D], the layout the interfaces expect.
        q, k, v = (t.view(shape).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, None)
        out, _ = interface(
            self, q, k, v, None, scaling=self.scaling, attention_instances=attention_instances
        )
        return self.o_proj(out.reshape(*x.shape[:-1], -1))


class _HFBlock(nn.Module):
    """No ``residual is None`` branch, so a shared artifact means exactly one graph."""

    def __init__(self, hidden: int, num_heads: int, layer_idx: int, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = _HFAttention(hidden, num_heads, layer_idx, config)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, 2 * hidden)
        self.down = nn.Linear(2 * hidden, hidden)

    def forward(self, x: torch.Tensor, attention_instances: dict) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_instances)
        h = self.post_attention_layernorm(x)
        return x + self.down(torch.relu(self.up(h)))


class _HFBackbone(nn.Module):
    def __init__(self, hidden: int, num_heads: int, num_layers: int, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden)
        self.layers = nn.ModuleList(
            [_HFBlock(hidden, num_heads, i, config) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor, attention_instances: dict) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_instances)
        return self.norm(x)


class _HFModel(nn.Module):
    """``attention_instances`` is a plain dict on purpose: ``nn.Module.__setattr__`` only
    registers Modules and Parameters, so these never appear in ``named_modules()``."""

    def __init__(
        self, hidden: int = 32, num_heads: int = 4, num_layers: int = 4, impl: str = "vllm"
    ):
        super().__init__()
        self.config = types.SimpleNamespace(_attn_implementation=impl)
        self.model = _HFBackbone(hidden, num_heads, num_layers, self.config)
        self.attention_instances = {
            i: _fake_attention_instance((hidden // num_heads) ** -0.5) for i in range(num_layers)
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Batch 1: vLLM flattens to one sequence dimension, and both interfaces
        # ``reshape(num_tokens, -1)`` on that assumption.
        return self.model(input_ids, self.attention_instances)


def _runner(model: nn.Module, enforce_eager: bool = False) -> TorchSpyreModelRunner:
    """A runner with only the attributes _compile_for_spyre reads."""
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model = model
    runner.compilation_config = types.SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE)
    runner.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=enforce_eager)
    )
    return runner


def test_finds_the_transformer_block_list() -> None:
    model = _Model(num_layers=4)
    found = _repeated_block_lists(model)
    assert len(found) == 1
    assert found[0] is model.model.layers


def test_ignores_module_lists_without_attention() -> None:
    class NoAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    assert _repeated_block_lists(NoAttention()) == []


def test_pp_missing_layers_do_not_break_discovery() -> None:
    model = _Model(num_layers=2, num_missing=2)
    assert _repeated_block_lists(model) == [model.model.layers]


def test_ignores_a_module_list_of_bare_attention_layers() -> None:
    """Zamba2's shared ``dpa_list`` holds bare Attention layers, not blocks."""

    class SharedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.dpa_list = nn.ModuleList([_fake_attention() for _ in range(2)])

    assert _repeated_block_lists(SharedAttention()) == []


def test_finds_heterogeneous_hybrid_stacks() -> None:
    """Hybrid Mamba+attention stacks (Granite 4.0, Jamba, Nemotron-H) mix classes."""
    model = _Model(num_layers=4, hybrid=True)
    assert len({type(b) for b in model.model.layers}) == 2
    assert _repeated_block_lists(model) == [model.model.layers]


# --- The widened predicate ---------------------------------------------------------------
#
# Discovery matches on ``AttentionLayerBase``, what vLLM itself keys KV-cache group
# discovery off, rather than on ``Attention``. These are the classes that brings in, and
# the model families that stop falling back to a whole-model graph because of it.

_ATTENTION_LAYER_CLASSES = [
    pytest.param(Attention, id="attention"),
    pytest.param(EncoderOnlyAttention, id="encoder_only"),
    pytest.param(MLAAttention, id="mla"),
    pytest.param(MambaMixer2, id="mamba_mixer"),
]

_WIDENED_ONLY = [p for p in _ATTENTION_LAYER_CLASSES if not issubclass(p.values[0], Attention)]


@pytest.mark.parametrize("layer_cls", _WIDENED_ONLY)
def test_the_widened_classes_are_the_ones_attention_alone_would_miss(layer_cls) -> None:
    """MLA (DeepSeek, Kimi) and the Mamba mixers hold per-layer state without being
    ``Attention`` subclasses, so an ``Attention`` predicate never saw their stacks.
    ``EncoderOnlyAttention`` is not in this list: it subclasses ``Attention`` and was
    always found."""
    assert issubclass(layer_cls, AttentionLayerBase)
    assert not issubclass(layer_cls, Attention)


@pytest.mark.parametrize("layer_cls", _ATTENTION_LAYER_CLASSES)
def test_a_block_stack_is_found_for_every_attention_layer_base(layer_cls) -> None:
    model = _Model(num_layers=4, attn_cls=layer_cls)

    assert _repeated_block_lists(model) == [model.model.layers]
    assert _runner(model)._compile_blocks() == 4


@pytest.mark.parametrize("layer_cls", _ATTENTION_LAYER_CLASSES)
def test_a_module_list_of_bare_layers_is_skipped_for_every_class(layer_cls) -> None:
    """The Zamba2 ``dpa_list`` guard has to widen with the predicate: a list of bare
    layers is not a block stack whichever ``AttentionLayerBase`` it holds."""

    class SharedLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.dpa_list = nn.ModuleList([_fake_layer(layer_cls) for _ in range(2)])

    assert _repeated_block_lists(SharedLayers()) == []


def test_finds_a_pure_mamba_stack() -> None:
    """No attention anywhere (Codestral Mamba, Mamba-2): every layer's state lives in a
    mixer, and only the widened predicate sees it."""
    model = _Model(num_layers=4, mixer_cls=MambaMixer2)

    assert not any(isinstance(m, Attention) for m in model.modules())
    assert all(isinstance(b.mixer, AttentionLayerBase) for b in model.model.layers)
    assert _repeated_block_lists(model) == [model.model.layers]
    assert _runner(model)._compile_blocks() == 4


def test_finds_a_hybrid_stack_of_real_layer_classes() -> None:
    """Granite 4.0 / Jamba: attention blocks and mixer blocks in one ``ModuleList``.
    Discovery already handled the mix; what is new is that the mixer halves count."""
    model = _Model(num_layers=4, hybrid=True, mixer_cls=MambaMixer2)
    kinds = {type(b).__name__ for b in model.model.layers}

    assert kinds == {"_Block", "_MambaBlock"}
    assert _repeated_block_lists(model) == [model.model.layers]
    assert _runner(model)._compile_blocks() == 4


def test_compile_blocks_wraps_every_block_in_place() -> None:
    model = _Model(num_layers=4)
    originals = list(model.model.layers)

    assert _runner(model)._compile_blocks() == 4

    for i, original in enumerate(originals):
        assert model.model.layers[i] is original
        assert isinstance(original, _Block)
        assert original._compiled_call_impl is not None


def test_compile_blocks_preserves_parameter_names() -> None:
    """An ``_orig_mod.`` segment in a parameter path breaks weight save/reload."""
    model = _Model(num_layers=4)
    before = [name for name, _ in model.named_parameters()]

    _runner(model)._compile_blocks()

    after = [name for name, _ in model.named_parameters()]
    assert after == before
    assert not any("_orig_mod" in name for name in after)


def test_pp_missing_layers_are_not_compiled() -> None:
    model = _Model(num_layers=2, num_missing=2)
    assert _runner(model)._compile_blocks() == 2
    assert all(isinstance(layer, PPMissingLayer) for layer in model.model.layers[2:])


def test_rejects_an_unknown_granularity_even_when_eager(monkeypatch) -> None:
    """Validation runs before the eager short-circuit, so typos are never silent."""
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "blocks")
    with pytest.raises(ValueError, match="SPYRE_COMPILE_GRANULARITY"):
        _runner(_Model(num_layers=2), enforce_eager=True)._compile_for_spyre()


def test_empty_granularity_falls_back_to_block(monkeypatch) -> None:
    """`export SPYRE_COMPILE_GRANULARITY=$UNSET` must mean unset, not invalid."""
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "")
    model = _Model(num_layers=2)
    _runner(model)._compile_for_spyre()
    assert all(block._compiled_call_impl is not None for block in model.model.layers)


def test_model_granularity_compiles_the_whole_model(monkeypatch) -> None:
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "model")
    compiled: list[nn.Module] = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: compiled.append(m) or m)

    model = _Model(num_layers=2)
    runner = _runner(model)
    runner._compile_for_spyre()

    assert compiled == [model]
    assert all(block._compiled_call_impl is None for block in model.model.layers)


def test_falls_back_to_whole_model_when_no_blocks_are_found(monkeypatch) -> None:
    """The path every MLA and vision-tower model takes."""
    compiled: list[nn.Module] = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: compiled.append(m) or m)

    class NoBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    model = NoBlocks()
    _runner(model)._compile_for_spyre()
    assert compiled == [model]


def test_eager_compiles_nothing(monkeypatch) -> None:
    compiled: list[nn.Module] = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: compiled.append(m) or m)

    model = _Model(num_layers=2)
    _runner(model, enforce_eager=True)._compile_for_spyre()

    assert compiled == []
    assert all(block._compiled_call_impl is None for block in model.model.layers)


def test_transformers_backend_attention_is_invisible_before_attach() -> None:
    """The bug: the layers are there, the Attention is in a dict, so nothing matches."""
    model = _HFModel(num_layers=4)
    assert isinstance(model.model.layers, nn.ModuleList)
    assert not any(isinstance(m, Attention) for m in model.modules())
    assert _repeated_block_lists(model) == []


def test_attach_makes_the_transformers_backend_block_list_discoverable() -> None:
    model = _HFModel(num_layers=4)

    assert attach_attention_instances(model) == 4

    for i, block in enumerate(model.model.layers):
        assert block.self_attn.attn is model.attention_instances[i]
    assert _repeated_block_lists(model) == [model.model.layers]
    assert _runner(model)._compile_blocks() == 4


def test_attach_switches_the_config_to_the_spyre_interface() -> None:
    """The attribute is only half of it: HF still has to dispatch to the forward that
    reads it, or the dict lookup (and its per-layer guard) stays in the graph."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from spyre_inference.transformers_backend import (
        SPYRE_ATTN_IMPL,
        _spyre_vllm_attention_forward,
    )

    model = _HFModel(num_layers=2)
    attach_attention_instances(model)

    assert model.config._attn_implementation == SPYRE_ATTN_IMPL
    assert ALL_ATTENTION_FUNCTIONS[SPYRE_ATTN_IMPL] is _spyre_vllm_attention_forward


def test_attach_is_idempotent() -> None:
    model = _HFModel(num_layers=2)
    assert attach_attention_instances(model) == 2
    instances = [block.self_attn.attn for block in model.model.layers]

    assert attach_attention_instances(model) == 2

    assert [block.self_attn.attn for block in model.model.layers] == instances
    assert _repeated_block_lists(model) == [model.model.layers]


def test_attach_skips_a_config_not_dispatching_through_vllm() -> None:
    """A vision tower keeps HF's own attention on its own sub-config; MLA uses
    ``vllm_mla``, whose interface takes ``(query, kv_c_normed, k_pe)``. Neither may be
    rerouted through the full-attention forward."""
    for impl in ("sdpa", "vllm_mla"):
        model = _HFModel(num_layers=2, impl=impl)
        assert attach_attention_instances(model) == 0
        assert not any(hasattr(b.self_attn, "attn") for b in model.model.layers)
        assert model.config._attn_implementation == impl


def test_attach_ignores_a_model_without_attention_instances() -> None:
    """Every in-tree model, where the Attention already is a submodule."""
    model = _Model(num_layers=2)
    assert attach_attention_instances(model) == 0
    assert _repeated_block_lists(model) == [model.model.layers]


def test_attach_leaves_an_attention_named_wrapper_alone() -> None:
    """Zamba2 names its decoder layer ``Zamba2AttentionDecoderLayer`` and gives it a
    ``layer_idx`` and a ``config``; matching on ``in`` rather than ``endswith`` would hang
    a second, redundant Attention off it."""
    model = _HFModel(num_layers=2)
    wrapper = model.model.layers[0]
    wrapper.__class__ = type("_HFAttentionDecoderLayer", (_HFBlock,), {})
    wrapper.layer_idx = 0
    wrapper.config = model.config

    assert attach_attention_instances(model) == 2

    assert "attn" not in wrapper._modules
    assert wrapper.self_attn.attn is model.attention_instances[0]


def test_attach_copies_the_hf_softmax_scale_onto_the_vllm_layer() -> None:
    """Upstream re-copies it on every forward, which a shared artifact never re-runs:
    only the first layer's Python frame executes, so the copy has to happen here."""
    model = _HFModel(num_layers=3)
    for i, block in enumerate(model.model.layers):
        block.self_attn.scaling = 0.125 + i
        model.attention_instances[i].impl.scale = 999.0

    attach_attention_instances(model)

    assert [inst.impl.scale for inst in model.attention_instances.values()] == [
        0.125,
        1.125,
        2.125,
    ]


def test_the_spyre_interface_reproduces_upstreams_output() -> None:
    """``_spyre_vllm_attention_forward`` duplicates upstream's reshape/pad body to change
    only the lookup, so this is what catches the two drifting apart on a vLLM bump."""
    from vllm.model_executor.models.transformers import vllm_attention_forward

    from spyre_inference.transformers_backend import _spyre_vllm_attention_forward

    hidden, num_heads, seq = 32, 4, 5
    head_dim = hidden // num_heads
    module = _HFAttention(hidden, num_heads, 0, types.SimpleNamespace(_attn_implementation="vllm"))
    instances = {0: _fake_attention_instance(head_dim**-0.5)}
    # Arithmetic, not random: this test is in the device-free smoke job, and seeding the
    # global RNG opens the Spyre card.
    qkv = torch.linspace(-1, 1, 3 * num_heads * seq * head_dim)
    q, k, v = (t.view(1, num_heads, seq, head_dim) for t in qkv.chunk(3))

    def call(fn, **kwargs):
        out, weights = fn(module, q, k, v, None, scaling=module.scaling, **kwargs)
        assert weights is None
        return out

    expected = call(vllm_attention_forward, attention_instances=instances)

    # No ``.attn`` yet, so the Spyre interface has to hand this back to upstream unchanged.
    assert torch.equal(call(_spyre_vllm_attention_forward, attention_instances=instances), expected)

    module.attn = instances[0]
    # ``attention_instances=None``: once attached, the dict is not consulted at all.
    assert torch.equal(call(_spyre_vllm_attention_forward, attention_instances=None), expected)


@pytest.mark.compile
def test_identical_blocks_share_compiled_artifacts_regardless_of_depth(
    isolated_dynamo_state,
) -> None:
    """Backend compile count must not grow with layer count -- and is 2, not 1,
    because layer 0 specializes on ``residual is None``."""
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    from torch._inductor.utils import fresh_cache

    def compile_counts(num_layers: int) -> tuple[int, int]:
        dynamo.reset()
        counters.clear()
        with fresh_cache():
            model = _Model(hidden=32, num_layers=num_layers)
            _runner(model)._compile_blocks()
            with torch.inference_mode():
                model(torch.zeros(2, 4, dtype=torch.long))
            return (
                counters["stats"]["unique_graphs"],
                counters["inductor"]["fxgraph_cache_miss"],
            )

    shallow_graphs, shallow_backend = compile_counts(2)
    deep_graphs, deep_backend = compile_counts(8)

    assert deep_backend == shallow_backend
    assert deep_graphs == shallow_graphs
    assert deep_backend < 8


@pytest.mark.compile
@pytest.mark.parametrize(
    "kwargs, expected_graphs",
    [
        # Homogeneous: one class, two graphs -- layer 0 specializes on ``residual is None``.
        pytest.param({"attn_cls": MLAAttention}, 2, id="mla"),
        pytest.param({"mixer_cls": MambaMixer2}, 2, id="pure_mamba"),
        # Alternating from layer 0, so ``_Block`` sees both residual branches and
        # ``_MambaBlock``, never at layer 0, sees only one: 2 + 1.
        pytest.param({"hybrid": True, "mixer_cls": MambaMixer2}, 3, id="hybrid"),
    ],
)
def test_the_widened_stacks_also_share_artifacts_regardless_of_depth(
    isolated_dynamo_state, kwargs, expected_graphs
) -> None:
    """Discovery is only half of it: the stacks the widened predicate brings in have to
    actually reuse one artifact per layer class, not one per layer.

    Both depths are even multiples of the hybrid stack's period, so the two arms differ
    only in depth and not in which residual branches each class sees.
    """
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    from torch._inductor.utils import fresh_cache

    def compile_counts(num_layers: int) -> tuple[int, int]:
        dynamo.reset()
        counters.clear()
        with fresh_cache():
            model = _Model(hidden=32, num_layers=num_layers, **kwargs)
            assert _runner(model)._compile_blocks() == num_layers
            with torch.inference_mode():
                model(torch.zeros(2, 4, dtype=torch.long))
            return (
                counters["stats"]["unique_graphs"],
                counters["inductor"]["fxgraph_cache_miss"],
            )

    shallow_graphs, shallow_backend = compile_counts(4)
    deep_graphs, deep_backend = compile_counts(8)

    assert deep_backend == shallow_backend
    assert deep_graphs == shallow_graphs
    assert deep_graphs == expected_graphs


@pytest.mark.compile
def test_attaching_is_what_makes_transformers_backend_blocks_share_one_artifact(
    isolated_dynamo_state,
) -> None:
    """The point of ``attach_attention_instances``: discovery alone is not enough.

    Upstream's interface resolves the layer as ``attention_instances[module.layer_idx]``
    and writes ``impl.scale`` on the object it finds. Both are per-layer facts inside the
    traced frame, so every block traces to a graph of its own however the block list was
    found. Attaching hoists both out and the count stops tracking depth.
    """
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    from torch._inductor.utils import fresh_cache

    def graph_count(num_layers: int, attach: bool) -> int:
        dynamo.reset()
        counters.clear()
        with fresh_cache():
            model = _HFModel(hidden=32, num_heads=4, num_layers=num_layers)
            if attach:
                assert attach_attention_instances(model) == num_layers
                assert _runner(model)._compile_blocks() == num_layers
            else:
                # Discovery rejects this model, so compile the blocks by hand: the only
                # difference between the two arms is then how attention is resolved.
                for block in model.model.layers:
                    block.compile(backend="inductor", fullgraph=True, dynamic=False)
            with torch.inference_mode():
                model(torch.zeros(1, 4, dtype=torch.long))
            return counters["stats"]["unique_graphs"]

    # One graph for the whole stack: ``_HFBlock`` has no trace-time branch, so unlike
    # ``_Block`` above there is not even a layer-0 specialization.
    assert graph_count(2, attach=True) == 1
    assert graph_count(8, attach=True) == 1
    # Upstream's lookup: one Dynamo trace per layer. The Inductor cache still hits (the
    # index never reaches the FX graph, only the guards), so the counter to watch is
    # ``unique_graphs``, not ``fxgraph_cache_miss``.
    assert graph_count(2, attach=False) == 2
    assert graph_count(8, attach=False) == 8
