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

"""Spyre backend for vLLM's unquantized routed experts.

vLLM's unquantized MoE oracle selects ``UnquantizedMoeBackend.OOT`` without building a
kernel, leaving ``UnquantizedFusedMoEMethod`` for the platform to implement. Model
adapters opt a layer in with a ``SpyreMoERecipe``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward,
    _moe_forward_shared,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from spyre_inference import envs

if TYPE_CHECKING:
    from torch_spyre._C import SpyreTensorLayout
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts as _RoutedExperts,
    )

    class RoutedExperts(_RoutedExperts):
        spyre_moe_recipe: SpyreMoERecipe
        spyre_moe_regions: dict[str, Any]
        spyre_moe_stick: int
        spyre_moe_chunks: int
        spyre_moe_route_dtype: torch.dtype
        spyre_moe_gate: torch.Tensor | None
        spyre_moe_up: torch.Tensor
        spyre_moe_down: torch.Tensor
        spyre_moe_gate_alias: torch.Tensor | None
        spyre_moe_up_alias: torch.Tensor
        spyre_moe_down_alias: torch.Tensor
        spyre_moe_biases: _ExpertBiases | None
        spyre_moe_chunk_scale: torch.Tensor
        spyre_moe_chunk_offsets: torch.Tensor
        spyre_moe_route_identity: torch.Tensor
        spyre_moe_selection_bias: torch.Tensor | None


logger = init_logger(__name__)

_MOE_COMPILER_CONFIG = {"frontend_pool_allocation": True}
_PERSISTENT_COMPILER_CONFIG = {"allow_all_ops_in_lx_planning": True}


def _derive_moe_chunks(
    hidden: int,
    top_k: int,
    stick: int,
    max_cores: int,
    override: int | None = None,
) -> int:
    """Choose the fewest stick-legal chunks that spread the gathered entries over the most cores.

    The gathered kernel splits work only across its ``top_k * chunks`` entries. A core holds
    ``top_k * hidden * inter / cores_used`` weight elements whatever the chunk count, so the
    split that fills the most cores also has the smallest per-core working set.
    """
    if hidden <= 0 or top_k <= 0 or stick <= 0:
        raise ValueError("MoE chunk selection requires positive dimensions, top_k, and stick size")
    if max_cores <= 0:
        raise ValueError(f"MoE chunk selection requires at least one core, got {max_cores}")
    if hidden % stick:
        raise ValueError(
            f"Spyre MoE hidden size {hidden} must be a multiple of {stick}-element sticks."
        )
    if override is not None:
        if override <= 0:
            raise ValueError(f"SPYRE_MOE_CHUNKS must be positive, got {override}")
        if hidden % (override * stick):
            raise ValueError(
                f"SPYRE_MOE_CHUNKS={override} requires hidden size {hidden} to split into "
                f"{override} whole {stick}-element-stick chunks."
            )
        return override

    hidden_sticks = hidden // stick
    best_cores, best_chunks = 0, 1
    for chunks in range(1, hidden_sticks + 1):
        if hidden_sticks % chunks:
            continue
        entries = top_k * chunks
        cores_used = max(
            split for split in range(1, min(entries, max_cores) + 1) if entries % split == 0
        )
        if cores_used > best_cores:
            best_cores, best_chunks = cores_used, chunks
    return best_chunks


@dataclass(frozen=True)
class SpyreMoERecipe:
    """The upstream MoE semantics implemented by Spyre's current kernels.

    The routing function follows vLLM's canonical ``(topk_weights, topk_ids)``
    contract. The persistent form expands that result to Spyre's dense route
    layout without selecting top-k a second time.
    """

    # "relu2" is non-gated (Nemotron-H): ``w13`` holds only the up projection, relu(up)**2.
    activation: Literal["gelu_tanh", "silu", "swigluoai", "relu2"]
    # "sigmoid_bias" is DeepSeek-V3's noaux_tc: sigmoid scores, the correction bias
    # added for selection only, the unbiased scores of the selected experts renormalized.
    # "dense_routes": router_logits already is the renormalized [tokens, experts] routing
    # table, zero off the top-k. Used by the TEMPORARY host router for GLM
    # (models/glm_cpu_router.py), which computes noaux_tc in fp32 on the CPU.
    routing: Literal["full_softmax", "topk_softmax", "sigmoid_bias", "dense_routes"]
    prepare_down_weight: Callable[[torch.Tensor], torch.Tensor] | None = None
    # Reorders ``w13``'s output rows (dim 1) into ``[gate | up]`` halves. gpt-oss interleaves
    # them, ``[gate0, up0, gate1, ...]``; the gate/up split below takes the first half as gate.
    # Also applied to ``w13_bias``, whose output axis is its last.
    prepare_gate_up_weight: Callable[[torch.Tensor], torch.Tensor] | None = None


def deinterleave_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """``[E, 2M, ...]`` with gate on even rows, up on odd, as ``[gate | up]`` halves."""
    return torch.cat([weight[:, 0::2], weight[:, 1::2]], dim=1)


@dataclass(frozen=True)
class _ExpertBiases:
    """Per-expert bias stacks, ``[E, 1, F]`` so each tiles and gathers like its weight pool.

    ``down_chunks`` is ``down`` split along the hidden axis into the gathered form's
    ``[E*chunks, 1, hidden/chunks]`` entries; the persistent forms read ``down`` whole.
    """

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    down_chunks: torch.Tensor


def _validate_recipe(layer: _RoutedExperts, recipe: SpyreMoERecipe) -> None:
    if recipe.routing == "full_softmax" and layer.custom_routing_function is None:
        raise NotImplementedError(
            "Full-softmax routing requires a model-specific Spyre MoE recipe."
        )
    # The recipe replaces upstream's router, so a routing field it does not implement
    # would load and route wrongly rather than fail.
    scoring = getattr(layer, "scoring_func", "softmax")
    bias = getattr(layer, "e_score_correction_bias", None)
    if recipe.routing == "dense_routes":
        # Whatever computes the routing table implements these fields.
        pass
    elif recipe.routing == "sigmoid_bias":
        if scoring != "sigmoid" or bias is None:
            raise NotImplementedError(
                "sigmoid_bias routing needs scoring_func='sigmoid' and an "
                f"e_score_correction_bias, got scoring_func={scoring!r}, bias={bias is not None}."
            )
        groups = (getattr(layer, "num_expert_group", None), getattr(layer, "topk_group", None))
        if any(g not in (None, 1) for g in groups):
            raise NotImplementedError(
                f"Spyre MoE does not support grouped top-k (num_expert_group, topk_group)={groups}."
            )
    elif scoring != "softmax" or bias is not None:
        raise NotImplementedError(
            f"Spyre {recipe.routing} routing does not implement scoring_func={scoring!r} "
            f"with e_score_correction_bias={bias is not None}."
        )
    if getattr(layer, "routed_scaling_factor", 1.0) != 1.0:
        raise NotImplementedError(
            "Spyre MoE recipes leave routed_scaling_factor to MoERunner "
            "(apply_routed_scale_to_output=True) or to prepare_down_weight."
        )


@cache
def _compiler_scopes() -> tuple[Any, Any]:
    from torch_spyre._inductor import config as spyre_config

    return (
        spyre_config.patch(_MOE_COMPILER_CONFIG),
        spyre_config.patch(_PERSISTENT_COMPILER_CONFIG),
    )


def configure_spyre_moe_layer(layer: _RoutedExperts, recipe: SpyreMoERecipe) -> None:
    """Opt one upstream ``RoutedExperts`` layer into Spyre, before weight loading.

    The post-load hook replaces vLLM's source layout with a second, transposed
    device layout. It frees source parameters because holding both layouts would
    double MoE weight memory, so unsupported configurations must be rejected here.
    """
    moe = layer.moe_config
    _validate_recipe(layer, recipe)
    if not isinstance(layer.quant_method, UnquantizedFusedMoEMethod):
        raise NotImplementedError("Spyre MoE backend does not support quantized experts.")
    if moe.is_lora_enabled:
        raise NotImplementedError("Spyre MoE backend does not support LoRA experts.")
    parallel = moe.moe_parallel_config
    # TP only narrows each expert's ``M``, and MoERunner all-reduces the partial sums.
    # The other axes would split the expert stacks the regions hold whole.
    unsupported = {
        "ep_size": moe.ep_size,
        "dp_size": moe.dp_size,
        "pcp_size": moe.pcp_size,
        "sp_size": moe.sp_size,
    }
    active = {name: value for name, value in unsupported.items() if value != 1}
    if active or parallel.enable_eplb:
        detail = ", ".join(f"{name}={value}" for name, value in active.items())
        if parallel.enable_eplb:
            detail = f"{detail}, enable_eplb=True" if detail else "enable_eplb=True"
        raise NotImplementedError(f"Spyre MoE backend requires local experts ({detail}).")
    if (
        layer.global_num_experts != layer.local_num_experts
        or moe.num_logical_experts != moe.num_experts
    ):
        raise NotImplementedError("Spyre MoE backend does not support remapped experts.")
    if not layer.renormalize:
        raise NotImplementedError("Spyre MoE backend requires normalized top-k routing weights.")
    if layer.apply_router_weight_on_input:
        raise NotImplementedError("Spyre MoE backend does not support input-weighted routing.")
    # swigluoai_uninterleave is the same math with gate and up already in halves (MiniMax-M3);
    # gpt-oss's interleaved swigluoai reaches that layout through prepare_gate_up_weight.
    accepted = {recipe.activation}
    if recipe.activation == "swigluoai":
        accepted.add("swigluoai_uninterleave")
    if recipe.activation == "relu2":
        accepted = {"relu2_no_mul"}
        if moe.has_bias or recipe.prepare_gate_up_weight is not None:
            raise NotImplementedError(
                "Spyre relu2 experts take no bias and have no gate/up halves to reorder."
            )
    if layer.activation.value not in accepted:
        raise NotImplementedError(
            f"Spyre MoE recipe requires activation={recipe.activation!r}, "
            f"got {layer.activation.value!r}."
        )
    if recipe.activation == "swigluoai":
        # The constants are vLLM's swigluoai_and_mul, which the layer's own limit/alpha do not
        # reach on any backend; a model that sets other values would be silently wrong.
        limit, alpha = moe.swiglu_limit, moe.swiglu_alpha
        beta = getattr(moe, "swiglu_beta", None)
        if (
            (limit is not None and limit != _SWIGLUOAI_LIMIT)
            or (alpha is not None and alpha != _SWIGLUOAI_ALPHA)
            or (beta is not None and beta != 1.0)
        ):
            raise NotImplementedError(
                f"Spyre swigluoai uses limit={_SWIGLUOAI_LIMIT}, alpha={_SWIGLUOAI_ALPHA} and "
                f"beta=1; the layer asks for limit={limit}, alpha={alpha}, beta={beta}."
            )
    layer = cast("RoutedExperts", layer)
    layer.spyre_moe_recipe = recipe
    layer.spyre_moe_regions = {}


def _topk(values: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = values.shape[0]
    # A single-row top-k does not lower; pad to two rows and drop the copy.
    padded = values.expand(2, -1).contiguous() if tokens == 1 else values
    weights, indices = torch.topk(padded, top_k, dim=-1)
    return weights[:tokens], indices[:tokens]


def _route_reduce_dtype(experts: int, dtype: torch.dtype) -> torch.dtype:
    # An fp32 rescale needs whole sticks in the source dtype ("cannot rescale device layout").
    from torch_spyre._C import get_elem_in_stick

    stick = get_elem_in_stick(dtype)
    if experts % stick == 0:
        return torch.float32
    logger.warning_once(
        "Spyre: reducing the routing softmax over %d experts in %s; an fp32 reduction "
        "requires the expert count to be a multiple of %d.",
        experts,
        dtype,
        stick,
    )
    return dtype


def _probs(router_logits: torch.Tensor, reduce_dtype: torch.dtype) -> torch.Tensor:
    return torch.softmax(router_logits.to(reduce_dtype), dim=-1).to(router_logits.dtype)


def _sigmoid_bias_kept(
    router_logits: torch.Tensor,
    selection_bias: torch.Tensor,
    top_k: int,
    fill: float,
) -> torch.Tensor:
    """Sigmoid scores at the experts the biased scores select, ``fill`` elsewhere.

    Selection runs in the logits' dtype: an fp32 sigmoid plus bias does not lower
    ("Multi-arg pointwise with mixed EA"), nor does an fp32 top-k. A gather by the top-k
    indices fails too (they come back in the input dtype), hence ``keep_by_index``.
    Rows stay padded to two, as ``_topk`` pads them.
    """
    tokens = router_logits.shape[0]
    scores = torch.sigmoid(router_logits)
    if tokens == 1:
        scores = scores.expand(2, -1).contiguous()
    _, selected = torch.topk(scores + selection_bias, top_k, dim=-1)
    return torch.ops.spyre.keep_by_index(
        scores,  # ty: ignore[invalid-argument-type]
        selected,  # ty: ignore[invalid-argument-type]
        -1,  # ty: ignore[invalid-argument-type]
        fill,  # ty: ignore[invalid-argument-type]
    )


def _sigmoid_bias_probs(
    router_logits: torch.Tensor, selection_bias: torch.Tensor, top_k: int
) -> torch.Tensor:
    """Renormalized sigmoid_bias weights in dense expert order."""
    dense = _sigmoid_bias_kept(router_logits, selection_bias, top_k, 0.0)
    dense = dense[: router_logits.shape[0]]
    return dense / dense.sum(-1, keepdim=True)


def _routing_weights(
    router_logits: torch.Tensor,
    top_k: int,
    routing: str,
    reduce_dtype: torch.dtype,
    selection_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if routing == "dense_routes":
        # The table holds exactly top_k positive weights per row, already renormalized.
        weights, indices = _topk(router_logits, top_k)
    elif routing == "sigmoid_bias":
        assert selection_bias is not None
        # Sigmoid scores are positive, so a -1 fill keeps the unselected experts last
        # and the second top-k returns exactly the selected ones with their weights.
        kept = _sigmoid_bias_kept(router_logits, selection_bias, top_k, -1.0)
        selected_values, indices = torch.topk(kept, top_k, dim=-1)
        tokens = router_logits.shape[0]
        selected_values, indices = selected_values[:tokens], indices[:tokens]
        weights = selected_values / selected_values.sum(-1, keepdim=True)
    elif routing == "full_softmax":
        selected_values, indices = _topk(_probs(router_logits, reduce_dtype), top_k)
        weights = selected_values / selected_values.sum(-1, keepdim=True)
    else:
        # ``reduce_dtype`` gates the expert dim; this reduces over ``top_k`` instead.
        selected_logits, indices = _topk(router_logits, top_k)
        weights = torch.softmax(selected_logits, dim=-1)
    return weights, indices


_SWIGLUOAI_LIMIT, _SWIGLUOAI_ALPHA = 7.0, 1.702


def _activation(x: torch.Tensor | None, up: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "relu2":
        return torch.square(torch.relu(up))
    assert x is not None
    if activation == "gelu_tanh":
        return F.gelu(x, approximate="tanh") * up
    if activation == "swigluoai":
        # gpt-oss: (clamp(up, +-limit) + 1) * gate * sigmoid(alpha * gate), gate clamped above.
        gate = torch.clamp(x, max=_SWIGLUOAI_LIMIT)
        up = torch.clamp(up, -_SWIGLUOAI_LIMIT, _SWIGLUOAI_LIMIT)
        return (up + 1) * gate * torch.sigmoid(gate * _SWIGLUOAI_ALPHA)
    return F.silu(x) * up


def _moe_gathered(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    gate: torch.Tensor | None,
    up: torch.Tensor,
    down: torch.Tensor,
    top_k: int,
    chunks: int,
    stick: int,
    reduce_dtype: torch.dtype,
    routing: str,
    activation: str,
    *,
    chunk_scale: torch.Tensor,
    chunk_offsets: torch.Tensor,
    biases: _ExpertBiases | None = None,
    selection_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens, hidden = x.shape
    if hidden % chunks:
        raise ValueError(
            f"gathered MoE hidden size {hidden} must divide evenly into {chunks} chunks"
        )
    if up.ndim != 3 or up.shape[0] % chunks:
        raise ValueError(
            "gathered up pool must have chunked [E*chunks, hidden/chunks, inter] shape"
        )
    slice_rows = hidden // chunks
    experts = up.shape[0] // chunks
    inter = up.shape[-1]
    expected_gate_shape = (experts * chunks, slice_rows, inter)
    if up.shape != expected_gate_shape or (gate is not None and gate.shape != expected_gate_shape):
        raise ValueError(
            f"gathered gate/up pools must have shape {expected_gate_shape}, "
            f"got gate={None if gate is None else tuple(gate.shape)}, up={tuple(up.shape)}"
        )
    expected_down_shape = (experts * chunks, inter, slice_rows)
    if down.shape != expected_down_shape:
        raise ValueError(
            f"gathered down pool must have chunked shape {expected_down_shape}, "
            f"got {tuple(down.shape)}"
        )
    weights, raw_indices = _routing_weights(
        router_logits, top_k, routing, reduce_dtype, selection_bias
    )
    rows = tokens * top_k
    # Entry e*chunks+c selects chunk c of expert e, in (row, chunk) order. Broadcasting against
    # the [1, chunks, 1] device offsets, not a cat of per-chunk index tensors, keeps the region
    # one kernel without per-call constants.
    expert_fp32 = (
        raw_indices.reshape(-1)[:, None].expand(rows, stick).contiguous().to(torch.float32)
    )
    entries = (expert_fp32[:, None, :] * chunk_scale + chunk_offsets).reshape(rows * chunks, stick)
    # Every lane of the one-stick-wide entries holds the entry, so take it as the max over the
    # stick: the float32 holds each pair of sticks in an interleaved order, and a reduction
    # reads it whole where a slice would split a pair.
    slice_index = entries.amax(-1).to(torch.int32)
    inputs = (
        x.reshape(tokens, 1, chunks, slice_rows)
        .expand(tokens, top_k, chunks, slice_rows)
        .contiguous()
        .reshape(rows * chunks, 1, slice_rows)
    )
    gate_out = None
    if gate is not None:
        gate_out = torch.bmm(inputs, gate[slice_index].reshape(rows * chunks, slice_rows, inter))
    up_out = torch.bmm(inputs, up[slice_index].reshape(rows * chunks, slice_rows, inter))
    if chunks > 1:
        if gate_out is not None:
            gate_out = gate_out.reshape(rows, chunks, 1, inter).sum(dim=1)
        up_out = up_out.reshape(rows, chunks, 1, inter).sum(dim=1)
    if biases is not None:
        # Gathered per expert, read off the same fp32 lanes as the entry index; a gate/up bias
        # is the expert's whole row, the down bias its chunk of the hidden axis.
        expert_index = expert_fp32.amax(-1).to(torch.int32)
        assert gate_out is not None
        gate_out = gate_out + biases.gate[expert_index]
        up_out = up_out + biases.up[expert_index]
    activated = _activation(gate_out, up_out, activation)
    act_rep = activated[:, None].expand(rows, chunks, 1, inter).reshape(rows * chunks, 1, inter)
    # Output chunk c of every row is the c-th hidden slice, so the parts reshape in place.
    parts = torch.bmm(act_rep, down[slice_index].reshape(rows * chunks, inter, slice_rows))
    if biases is not None:
        parts = parts + biases.down_chunks[slice_index]
    expert_out = parts.reshape(tokens, top_k, hidden)
    return (expert_out * weights[..., None]).sum(dim=1)


def _moe_persistent_routing(
    values: torch.Tensor,
    route_identity: torch.Tensor,
    top_k: int,
    stick: int,
) -> torch.Tensor:
    _, selected = _topk(values, top_k)
    weights = torch.ops.spyre.keep_by_index(
        values,  # ty: ignore[invalid-argument-type]
        selected,  # ty: ignore[invalid-argument-type]
        -1,  # ty: ignore[invalid-argument-type]
        0.0,  # ty: ignore[invalid-argument-type]
    )
    weights = weights / weights.sum(-1, keepdim=True)
    packed = torch.relu(weights.unsqueeze(-1).expand(-1, -1, stick))
    return (packed @ route_identity)[..., :1]


def _moe_persistent_selected_routing(
    dense_topk_probs: torch.Tensor, route_identity: torch.Tensor, stick: int
) -> torch.Tensor:
    """Convert dense weights already selected by vLLM-style top-k routing."""
    packed = torch.relu(dense_topk_probs.unsqueeze(-1).expand(-1, -1, stick))
    return (packed @ route_identity)[..., :1]


def _name_persistent_dims(
    x: torch.Tensor, gate: torch.Tensor | None, up: torch.Tensor, down: torch.Tensor
) -> None:
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )

    # Whole-expert pools [E, hidden, inter] / [E, inter, hidden].
    experts, hidden, inter = up.shape
    for name, extent in (
        ("E", experts),
        ("T", x.shape[0]),
        ("H", hidden),
        ("M", inter),
        ("ONE", 1),
    ):
        declare_tensor_dim(name, extent)
    name_tensor_dims(x, ["T", "H"])
    if gate is not None:
        name_tensor_dims(gate, ["E", "H", "M"])
    name_tensor_dims(up, ["E", "H", "M"])
    name_tensor_dims(down, ["E", "M", "H"])


def _moe_persistent(
    x: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor | None,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: str,
    biases: _ExpertBiases | None = None,
) -> torch.Tensor:
    # Process routed experts over token tiles using the persistent weight pools.
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr import for_each_tile

    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()

    operands, dims = _expert_loop_operands(x, route, gate, up, down, biases)

    def expert_body(acc, tiles):
        x, route_tile, *weights = tiles
        expert_out = _expert_tile(x, weights, gate is not None, activation)
        return acc + (expert_out * route_tile).squeeze(0), None

    result, _ = for_each_tile(
        expert_body,
        operands,
        dims=dims,
        tile_size=1,
        init=torch.zeros_like(x),
    )
    return result


def _expert_loop_operands(x, route, gate, up, down, biases):
    """The persistent loops' operands: the bias stacks tile on the expert axis with the pools.

    A non-gated recipe has no gate pool, so it is left out rather than passed as ``None``.
    """
    pools = (up, down) if gate is None else (gate, up, down)
    operands: tuple[torch.Tensor, ...] = (x, route, *pools)
    dims: tuple[int | None, ...] = (None, 0) + (0,) * len(pools)
    if biases is not None:
        operands += (biases.gate, biases.up, biases.down)
        dims += (0, 0, 0)
    return operands, dims


def _expert_tile(x, weights, gated, activation):
    """One expert's output for every token.

    ``weights`` is the tiled ``[gate,] up, down`` pools, then the gate/up/down biases if any.
    """
    if gated:
        gate_tile, up_tile, down_tile, *bias = weights
        gate_out = torch.matmul(x, gate_tile)
    else:
        up_tile, down_tile, *bias = weights
        gate_out = None
    up_out = torch.matmul(x, up_tile)
    if bias:
        assert gate_out is not None
        gate_out = gate_out + bias[0]
        up_out = up_out + bias[1]
    out = torch.matmul(_activation(gate_out, up_out, activation), down_tile)
    return out + bias[2] if bias else out


def _moe_persistent_in_graph(
    x: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor | None,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: str,
    biases: _ExpertBiases | None = None,
) -> torch.Tensor:
    """``_moe_persistent`` inside a block graph, where nothing names x's token axis."""
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr import for_each_tile

    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()

    operands, dims = _expert_loop_operands(x, route, gate, up, down, biases)

    def expert_body(acc, tiles):
        x, route_tile, *weights = tiles
        expert_out = _expert_tile(x, weights, gate is not None, activation)
        return acc + expert_out * route_tile, None

    # Loop-body ops only receive this scope's hints, so T is named here; every body op stays
    # at [1, T, C] (C is M or H) so one name list fits them all.
    init = torch.zeros_like(x).unsqueeze(0)
    with spyre_hint(named_dims=["ONE", "T", "C"]):
        result, _ = for_each_tile(
            expert_body,
            operands,
            dims=dims,
            tile_size=1,
            init=init,
        )
    return result.squeeze(0)


def _layer_biases(layer: RoutedExperts) -> _ExpertBiases | None:
    # Set by _prepare_layer; absent on a layer built without it, which carries no biases.
    return getattr(layer, "spyre_moe_biases", None)


def _gathered(layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    recipe = layer.spyre_moe_recipe
    chunks = layer.spyre_moe_chunks
    gate, up, down = (
        layer.spyre_moe_gate_alias,
        layer.spyre_moe_up_alias,
        layer.spyre_moe_down_alias,
    )
    return _moe_gathered(
        x,
        router_logits,
        gate,
        up,
        down,
        layer.top_k,
        chunks,
        layer.spyre_moe_stick,
        layer.spyre_moe_route_dtype,
        recipe.routing,
        recipe.activation,
        chunk_scale=layer.spyre_moe_chunk_scale,
        chunk_offsets=layer.spyre_moe_chunk_offsets,
        biases=_layer_biases(layer),
        selection_bias=getattr(layer, "spyre_moe_selection_bias", None),
    )


def _gathered_tokens(
    layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    # The gathered kernel only lowers at one token. ``dynamic=False`` specializes this loop to
    # the packed bucket, so slicing, expert calls, and assembly stay in one compiled region.
    rows = [
        _gathered(layer, x[token : token + 1], router_logits[token : token + 1])
        for token in range(x.shape[0])
    ]
    return torch.cat(rows)


def _rows_are_stick_addressable(x: torch.Tensor, router_logits: torch.Tensor, stick: int) -> bool:
    # Cloning row ``t`` bakes its flat storage offset, ``storage_offset() + t * stride(0)``, into
    # the kernel coordinate, and the backend can only bake whole sticks: hence both terms.
    return all(
        t.storage_offset() % stick == 0 and t.stride(0) % stick == 0 for t in (x, router_logits)
    )


def _topk_probs(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Materialize canonical vLLM top-k weights in dense expert order."""
    topk_weights, topk_ids = _routing_weights(
        router_logits, top_k, "topk_softmax", router_logits.dtype
    )
    return torch.zeros_like(router_logits).scatter(-1, topk_ids, topk_weights)


def _layer_sigmoid_bias_probs(layer: RoutedExperts, router_logits: torch.Tensor) -> torch.Tensor:
    assert layer.spyre_moe_selection_bias is not None
    return _sigmoid_bias_probs(router_logits, layer.spyre_moe_selection_bias, layer.top_k)


# Above this many rows, torch-spyre's top-k + keep_by_index over 512 experts keeps the wrong
# count once k >= 17 (Nemotron-3-Super's 22 of 512): 474 of 602 rows, some keeping 2, so
# their renormalization divides by zero. Row blocks of at most this size are each exact.
_SIGMOID_BIAS_MAX_ROWS = 256


def _chunked_sigmoid_bias_probs(
    layer: RoutedExperts, router_logits: torch.Tensor
) -> torch.Tensor:
    """``sigmoid_bias_probs`` region calls over row blocks of at most _SIGMOID_BIAS_MAX_ROWS."""
    region = _region(layer, "sigmoid_bias_probs", _layer_sigmoid_bias_probs)
    tokens = router_logits.shape[0]
    if tokens <= _SIGMOID_BIAS_MAX_ROWS:
        return region(layer, router_logits)
    # Equal blocks, so a bucket compiles one block shape (plus at most one remainder).
    blocks = -(-tokens // _SIGMOID_BIAS_MAX_ROWS)
    rows = -(-tokens // blocks)
    return torch.cat(
        [
            region(layer, router_logits[start : start + rows].clone())
            for start in range(0, tokens, rows)
        ]
    )


def _route(layer: RoutedExperts, probs: torch.Tensor) -> torch.Tensor:
    return _moe_persistent_routing(
        probs,
        layer.spyre_moe_route_identity,
        layer.top_k,
        layer.spyre_moe_stick,
    )


def _route_selected(layer: RoutedExperts, topk_probs: torch.Tensor) -> torch.Tensor:
    return _moe_persistent_selected_routing(
        topk_probs, layer.spyre_moe_route_identity, layer.spyre_moe_stick
    )


def _experts(layer: RoutedExperts, x: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    return _moe_persistent(
        x,
        route,
        layer.spyre_moe_gate,
        layer.spyre_moe_up,
        layer.spyre_moe_down,
        layer.spyre_moe_recipe.activation,
        _layer_biases(layer),
    )


def _region(layer: RoutedExperts, name: str, fn: Any) -> Any:
    region = layer.spyre_moe_regions.get(name)
    if region is None:
        region = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
        layer.spyre_moe_regions[name] = region
        # Deferred import: spyre_inference.v1.worker imports this module's package.
        from spyre_inference.v1.worker import compile_guard

        compile_guard.watch(fn, f"MoE region {name!r}")
    return region


def _reset_named_dims() -> None:
    from torch_spyre._inductor.wsr.propagate_named_dims import reset

    reset()


def _expert_kernel_layout(weight: torch.Tensor) -> SpyreTensorLayout:
    """``[E, C, F]`` as ``[E, F // stick, C, stick]``, the nn.Linear order (torch-spyre #1339)."""
    from torch_spyre._C import SpyreTensorLayout

    return SpyreTensorLayout(list(weight.shape), list(weight.stride()), weight.dtype, [1, 0, 2])


def _to_spyre_expert_weight(
    weight: torch.Tensor, pad: tuple[int, ...], *, kernel_order: bool = False
) -> torch.Tensor:
    """Move an ``[E, C, F]`` expert stack, widened by the ``F.pad`` spec ``pad``, to the device."""
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    if any(pad):
        weight = F.pad(weight, pad)
    if not kernel_order:
        moved = dma_moe_expert_weight_to_spyre(weight)
        assert moved is not None
        return moved
    if not torch.spyre.is_initialized():
        torch.empty(0, dtype=weight.dtype, device="spyre")
    weight = weight.contiguous()
    layout = _expert_kernel_layout(weight)
    assert weight.shape[-1] % layout.elems_per_stick() == 0, "the free dim must span whole sticks"
    # Lands on the current Spyre device, as the ``dma_*`` helpers do.
    return weight.to(device_layout=layout)  # ty: ignore[no-matching-overload]


def _chunk_entry_constants(chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Device scale and offsets of the gathered entry index ``e*chunks + c``.

    Inputs rather than in-graph scalars, so the region needs no per-call constants; the
    size-1 stick dim lets them broadcast against the widened fp32 expert index.
    """
    scale = torch.full((1, 1, 1), float(chunks), dtype=torch.float32).to("spyre")
    offsets = torch.arange(chunks, dtype=torch.float32).reshape(1, chunks, 1).to("spyre")
    return scale, offsets


def _check_pool_layout(pool: torch.Tensor, expected: SpyreTensorLayout) -> None:
    from torch_spyre._C import get_spyre_tensor_layout

    actual = get_spyre_tensor_layout(pool)
    if (actual.device_size, actual.stride_map) != (expected.device_size, expected.stride_map):
        raise RuntimeError(
            f"MoE pool device layout {actual.device_size} / {actual.stride_map} differs from "
            f"the {expected.device_size} / {expected.stride_map} its chunk alias reinterprets."
        )


def _chunk_pool_alias(pool: torch.Tensor, chunks: int) -> torch.Tensor:
    """Expose a whole-expert [E, contract, free] pool as [E*chunks, contract/chunks, free].

    The view shares storage; entry e*chunks+c selects chunk c of expert e. The contract axis
    is outer to the free-dim sticks, so chunking it preserves flat physical order.
    """
    from torch_spyre import _C
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    experts, contract, free = pool.shape
    if contract % chunks:
        raise ValueError(f"gate/up contract dim {contract} must divide into {chunks} chunks")
    eps = SpyreTensorLayout([experts, contract, free], pool.dtype).elems_per_stick()
    device_dtype = get_device_dtype(pool.dtype)
    _check_pool_layout(
        pool,
        SpyreTensorLayout(
            [experts, contract, free // eps, eps], [contract * free, free, eps, 1], device_dtype
        ),
    )
    c4 = contract // chunks
    layout = SpyreTensorLayout(
        [experts * chunks, c4, free // eps, eps],
        [c4 * free, free, eps, 1],
        device_dtype,
    )
    return _C.reinterpret_tensor_with_layout(
        pool,
        [experts * chunks, c4, free],
        [c4 * free, free, 1],
        0,
        layout,
    )


def _down_chunk_pool_alias(pool: torch.Tensor, chunks: int) -> torch.Tensor:
    """Expose down weights as expert-major [E*chunks, M, H/chunks] entries.

    The view shares storage. Each e*chunks+c entry selects output chunk c for expert e.
    The H-stick axis is outer to M, so chunking it preserves flat physical order.
    """
    from torch_spyre import _C
    from torch_spyre._C import SpyreTensorLayout, get_elem_in_stick

    experts, inter, hidden = pool.shape
    stick = get_elem_in_stick(pool.dtype)
    if hidden % chunks or (hidden // chunks) % stick:
        raise ValueError(
            f"down hidden size {hidden} must divide into {chunks} stick-aligned chunks"
        )
    _check_pool_layout(pool, _expert_kernel_layout(pool))
    chunk_hidden = hidden // chunks
    size = [experts * chunks, inter, chunk_hidden]
    stride = [inter * chunk_hidden, chunk_hidden, 1]
    layout = SpyreTensorLayout(size, stride, pool.dtype, [1, 0, 2])
    return _C.reinterpret_tensor_with_layout(pool, size, stride, 0, layout)


def _prepare_biases(
    layer: RoutedExperts, inter: int, pad: int, chunks: int
) -> _ExpertBiases | None:
    """Relayout ``w13_bias`` / ``w2_bias`` next to the pools, freeing the sources."""
    # vLLM registers both exactly when ``moe_config.has_bias``.
    if not hasattr(layer, "w13_bias"):
        return None
    w13_bias = layer.get_parameter("w13_bias").data
    if layer.spyre_moe_recipe.prepare_gate_up_weight is not None:
        w13_bias = layer.spyre_moe_recipe.prepare_gate_up_weight(w13_bias)
    w2_bias = layer.get_parameter("w2_bias").data
    experts, hidden = w2_bias.shape
    if w13_bias.shape != (experts, 2 * inter):
        raise ValueError(
            f"unexpected MoE expert bias shapes: w13_bias={tuple(w13_bias.shape)} "
            f"w2_bias={tuple(w2_bias.shape)}"
        )
    biases = _ExpertBiases(
        gate=_to_spyre_expert_weight(w13_bias[:, :inter].unsqueeze(1), (0, pad)),
        up=_to_spyre_expert_weight(w13_bias[:, inter:].unsqueeze(1), (0, pad)),
        down=_to_spyre_expert_weight(w2_bias.unsqueeze(1), ()),
        down_chunks=_to_spyre_expert_weight(
            w2_bias.reshape(experts * chunks, 1, hidden // chunks), ()
        ),
    )
    del layer.w13_bias, layer.w2_bias
    return biases


def _prepare_layer(layer: RoutedExperts) -> None:
    from torch_spyre._C import get_elem_in_stick

    if hasattr(layer, "spyre_moe_gate"):
        raise RuntimeError(
            "Spyre MoE weights are immutable after loading; weight reload is unsupported."
        )
    w13 = layer.get_parameter("w13_weight").data
    w2_shape = tuple(layer.get_parameter("w2_weight").shape)
    recipe = layer.spyre_moe_recipe
    # A non-gated recipe's w13 is the up projection alone, [E, M, H].
    gated = recipe.activation != "relu2"
    experts, up_rows, hidden = w13.shape
    if gated and up_rows % 2:
        raise ValueError(f"Spyre MoE requires a gated w13 stack, got {tuple(w13.shape)}.")
    inter = up_rows // 2 if gated else up_rows
    if w2_shape != (experts, hidden, inter):
        raise ValueError(
            f"unexpected MoE expert weight shapes: w13={tuple(w13.shape)} w2={w2_shape}"
        )

    from torch_spyre._inductor import config as spyre_config

    # All shape and alignment checks run before the first device relayout.
    stick = get_elem_in_stick(w13.dtype)
    chunks = _derive_moe_chunks(
        hidden, layer.top_k, stick, spyre_config.sencores, envs.SPYRE_MOE_CHUNKS
    )
    # TP can leave intermediate shards mid-stick; zero-padding is inert because those
    # activation lanes multiply zero rows in the down stack. A gate/up bias pads with zeros
    # too, and its padded activation lanes still meet those zero rows.
    pad = -inter % stick
    if recipe.prepare_gate_up_weight is not None:
        w13 = recipe.prepare_gate_up_weight(w13)
    # Share gate/up allocations between persistent and gathered chunk views.
    layer.spyre_moe_gate = layer.spyre_moe_gate_alias = None
    if gated:
        layer.spyre_moe_gate = _to_spyre_expert_weight(
            w13[:, :inter, :].transpose(1, 2), (0, pad)
        )
        layer.spyre_moe_gate_alias = _chunk_pool_alias(layer.spyre_moe_gate, chunks)
    up_rows_start = inter if gated else 0
    layer.spyre_moe_up = _to_spyre_expert_weight(
        w13[:, up_rows_start:, :].transpose(1, 2), (0, pad)
    )
    layer.spyre_moe_chunks = chunks
    layer.spyre_moe_up_alias = _chunk_pool_alias(layer.spyre_moe_up, chunks)
    del layer.w13_weight, w13
    w2 = layer.get_parameter("w2_weight").data
    transform_down = recipe.prepare_down_weight
    if transform_down is not None:
        w2 = transform_down(w2)
    # Keep the complete stack for persistent compute and expose a view for gathered decode.
    layer.spyre_moe_down = _to_spyre_expert_weight(
        w2.transpose(1, 2), (0, 0, 0, pad), kernel_order=True
    )
    layer.spyre_moe_down_alias = _down_chunk_pool_alias(layer.spyre_moe_down, chunks)
    layer.spyre_moe_chunk_scale, layer.spyre_moe_chunk_offsets = _chunk_entry_constants(chunks)
    logger.info_once(
        "Spyre MoE: selected %d gathered weight chunks (top_k=%d, cores=%d).",
        chunks,
        layer.top_k,
        spyre_config.sencores,
    )
    del layer.w2_weight, w2
    layer.spyre_moe_biases = _prepare_biases(layer, inter, pad, chunks)

    dtype = layer.spyre_moe_up.dtype
    layer.spyre_moe_stick = stick
    layer.spyre_moe_route_dtype = (
        _route_reduce_dtype(experts, dtype)
        if layer.spyre_moe_recipe.routing == "full_softmax"
        else dtype
    )
    layer.spyre_moe_route_identity = torch.eye(stick, dtype=dtype).to("spyre")
    layer.spyre_moe_selection_bias = None
    if layer.spyre_moe_recipe.routing == "sigmoid_bias":
        bias = cast(torch.Tensor, layer.e_score_correction_bias).detach().float()
        # A shared offset does not change the selection, and these biases sit near a
        # large common value (about 9 for GLM-4.7-Flash): centred, their per-expert
        # spread survives the cast to the selection dtype.
        layer.spyre_moe_selection_bias = (bias - bias.mean()).to(dtype).to("spyre")
    logger.info_once(
        "Spyre: relaid out routed-expert stacks (%d experts, hidden=%d, intermediate=%d%s).",
        experts,
        hidden,
        inter,
        f" padded to {inter + pad}" if pad else "",
    )


@MoERunner.register_oot(name="MoERunner")
class SpyreMoERunner(MoERunner):
    """Takes vLLM's direct MoE entry, as upstream does on CPU and TPU, for stick-aligned batches.

    A compiled block then compiles the MoE dispatch along with the rest of its graph. In-graph
    routing only lowers when the token axis spans whole sticks, so other batch sizes keep the
    opaque ``torch.ops.vllm.moe_forward`` op and its compiled regions.
    """

    def _select_forward(self) -> Callable:
        from torch_spyre._C import get_elem_in_stick

        self._spyre_stick = get_elem_in_stick(self.moe_config.in_dtype)
        return self._spyre_forward_entry

    def _spyre_forward_entry(self, hidden_states: torch.Tensor, *args: Any) -> Any:
        return self._entry_for(hidden_states.shape[0])(hidden_states, *args)

    def _entry_for(self, tokens: int) -> Callable:
        shared = self.shared_experts is not None
        # In-graph routing cannot split rows, which sigmoid_bias needs past
        # _SIGMOID_BIAS_MAX_ROWS; the opaque op's regions route in row blocks.
        modules = self.__dict__.get("_modules") or {}
        routed = modules.get("routed_experts", self.__dict__.get("routed_experts"))
        recipe = getattr(routed, "spyre_moe_recipe", None)
        too_many_rows = (
            recipe is not None
            and recipe.routing == "sigmoid_bias"
            and tokens > _SIGMOID_BIAS_MAX_ROWS
        )
        if tokens % self._spyre_stick == 0 and not too_many_rows:
            return _moe_forward_shared if shared else _moe_forward
        return torch.ops.vllm.moe_forward_shared if shared else torch.ops.vllm.moe_forward


@CustomOp.register_oot(name="UnquantizedFusedMoEMethod")
class SpyreUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """OOT bridge from vLLM's unquantized MoE method to ``SpyreMoERecipe``."""

    def __init__(self, moe) -> None:
        super().__init__(moe)
        # The source expert parameters are replaced with device-specific stacks. Set
        # after super().__init__, which since 0.31 assigns this per instance.
        self.supports_pre_processed_weights = False

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # The biases live in spyre_moe_biases, read by the Spyre kernels; the quant config
        # only carries them to upstream's fused kernels, and the relayout freed the sources.
        return FusedMoEQuantConfig.make(
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def process_weights_after_loading(self, layer: _RoutedExperts) -> None:
        if getattr(layer, "spyre_moe_recipe", None) is None:
            raise NotImplementedError(
                "Spyre MoE backend requires an explicit model-specific recipe; "
                "this architecture has not opted in."
            )
        _prepare_layer(cast("RoutedExperts", layer))
        # MoERunner._forward_impl initializes this lazily. Traced, that check becomes a guard
        # which warmup's first call flips, so the first real request would re-trace the block.
        layer._ensure_moe_quant_config_init()

    def apply_monolithic(
        self,
        layer: _RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer = cast("RoutedExperts", layer)
        tokens = x.shape[0]
        if torch.compiler.is_compiling():
            # Traced into the block graph, which SpyreMoERunner only does for stick-aligned
            # batches: the persistent form runs inline, without compiled regions or their
            # scopes. An in-graph fp32 softmax cannot be lowered, so routing stays in the
            # logits' dtype.
            # Note: This branching here is solely done to support eager and compiled
            # execution of the experts.
            recipe = layer.spyre_moe_recipe
            if recipe.routing == "full_softmax":
                route = _route(layer, _probs(router_logits, router_logits.dtype))
            elif recipe.routing == "sigmoid_bias":
                route = _route_selected(layer, _layer_sigmoid_bias_probs(layer, router_logits))
            elif recipe.routing == "dense_routes":
                route = _route_selected(layer, router_logits)
            else:
                route = _route_selected(layer, _topk_probs(router_logits, layer.top_k))
            return _moe_persistent_in_graph(
                x,
                route,
                layer.spyre_moe_gate,
                layer.spyre_moe_up,
                layer.spyre_moe_down,
                recipe.activation,
                _layer_biases(layer),
            )
        moe_scope, persistent_scope = _compiler_scopes()
        # A single row is handed to the region whole, so no row slice needs an addressable offset.
        if tokens == 1 or (
            tokens <= envs.SPYRE_MOE_GATHERED_MAX_TOKENS
            and _rows_are_stick_addressable(x, router_logits, layer.spyre_moe_stick)
        ):
            with moe_scope:
                if tokens == 1:
                    return _region(layer, "gathered", _gathered)(layer, x, router_logits)
                return _region(layer, "gathered_batch", _gathered_tokens)(layer, x, router_logits)
        with moe_scope:
            recipe = layer.spyre_moe_recipe
            if recipe.routing == "full_softmax":
                probs = _region(layer, "probs", _probs)(router_logits, layer.spyre_moe_route_dtype)
                route = _region(layer, "route", _route)(layer, probs)
            elif recipe.routing == "sigmoid_bias":
                topk_probs = _chunked_sigmoid_bias_probs(layer, router_logits)
                route = _region(layer, "route_selected", _route_selected)(layer, topk_probs)
            elif recipe.routing == "dense_routes":
                route = _region(layer, "route_selected", _route_selected)(layer, router_logits)
            else:
                topk_probs = _region(layer, "topk_probs", _topk_probs)(router_logits, layer.top_k)
                route = _region(layer, "route_selected", _route_selected)(layer, topk_probs)
            _name_persistent_dims(x, layer.spyre_moe_gate, layer.spyre_moe_up, layer.spyre_moe_down)
            try:
                with persistent_scope:
                    return _region(layer, "experts", _experts)(layer, x, route)
            finally:
                _reset_named_dims()
