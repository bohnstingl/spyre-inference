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

"""Spyre RMSNorm: OOT dispatch, and on-device agreement with the fp32 reference.

Since the op forwards to vLLM's ``forward_native``, the oracle is the fp32-accumulating
upstream definition itself, so the tolerance is the device's own error rather than the
fp16-vs-fp32 gap the pre-fp32 op had to allow.
"""

import sys

import pytest
import torch

# vLLM's own float16 bound for these ops (ir/ops/layernorm.py `override_tolerance`).
# Tighter than the fp16-reference tolerance this test used before fp32 promotion
# (atol=1e-2, rtol=1e-2); the measured worst case over seeds/shapes/var widths needs
# atol 3.3e-3 at this rtol, so the margin is ~3x.
_ATOL, _RTOL = 1e-2, 2e-3


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    variance_size: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """CPU mirror of ``vllm.ir.ops.{rms_norm,fused_add_rms_norm}``.

    The reduction and the residual add accumulate in fp32 and only the result is cast
    back, which is exactly what the op now promises.
    """
    orig_dtype = x.dtype
    x = x.float()
    residual_out = None
    if residual is not None:
        x = x + residual.float()
        residual_out = x.to(orig_dtype)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    normed = (x.to(weight.dtype) * weight).to(orig_dtype)
    return normed if residual is None else (normed, residual_out)


@pytest.mark.rmsnorm
@pytest.mark.parametrize("batch_size", [1, 8])
# Hidden sizes must be a multiple of 64 (Spyre 128-byte stick / 2 bytes fp16).
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_rmsnorm_matches_reference(
    default_vllm_config, batch_size, hidden_size, use_residual
):
    """SpyreRMSNorm.forward_oot on device matches the upstream fp32 reference."""
    import vllm.ir
    from vllm.config import get_current_vllm_config

    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    # ``WorkerBase.__init__`` installs this from the config; without it the vllm_ir
    # torch custom op wraps the norm and the Spyre backend cannot lower the wrapped
    # reduction ("Multi-arg pointwise with mixed EA"), which no engine run hits.
    vllm.ir.set_default_torch_wrap(
        get_current_vllm_config().compilation_config.ir_enable_torch_wrap
    )

    eps = 1e-6
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype) if use_residual else None
    layer = SpyreRMSNorm(hidden_size, eps=eps).to(dtype)
    # The default weight is all ones, which hides a dropped or wrongly shaped multiply.
    layer.weight.data = torch.randn(hidden_size, dtype=dtype)

    expected = reference_rms_norm(x, layer.weight.data, eps, residual)

    layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre"), residual.to("spyre") if use_residual else None)

    if use_residual:
        expected_norm, expected_residual = expected
        actual_norm, actual_residual = actual
        torch.testing.assert_close(
            actual_norm.cpu().float(), expected_norm.float(), atol=_ATOL, rtol=_RTOL
        )
        torch.testing.assert_close(
            actual_residual.cpu().float(), expected_residual.float(), atol=_ATOL, rtol=_RTOL
        )
    else:
        torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=_ATOL, rtol=_RTOL)


@pytest.mark.rmsnorm
@pytest.mark.parametrize("hidden_size", [256, 512])
def test_spyre_rmsnorm_accumulates_the_variance_in_fp32(default_vllm_config, hidden_size):
    """The point of the switch to ``forward_native``: the reduction is not fp16.

    Scaled so ``x**2`` exceeds the fp16 max (65504) and the fp16 sum-of-squares
    saturates to inf, collapsing ``rsqrt`` to 0. The device must track the fp32 oracle
    and visibly disagree with the fp16 one, which the moderate-magnitude cases above
    cannot show because there both oracles agree to within tolerance.
    """
    import vllm.ir
    from vllm.config import get_current_vllm_config

    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    vllm.ir.set_default_torch_wrap(
        get_current_vllm_config().compilation_config.ir_enable_torch_wrap
    )

    eps = 1e-6
    torch.manual_seed(0)
    x = (torch.randn(4, hidden_size) * 64.0).half()
    layer = SpyreRMSNorm(hidden_size, eps=eps).to(torch.float16)
    layer.weight.data = torch.ones(hidden_size, dtype=torch.float16)
    weight = layer.weight.data.clone()

    fp16_variance = x.pow(2).mean(dim=-1, keepdim=True)
    assert fp16_variance.isinf().any(), "input no longer overflows fp16; pick a larger scale"
    fp16_oracle = x * torch.rsqrt(fp16_variance + eps) * weight
    fp32_oracle = reference_rms_norm(x, weight, eps)

    layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre")).cpu().float()

    torch.testing.assert_close(actual, fp32_oracle.float(), atol=_ATOL, rtol=_RTOL)
    assert (actual - fp16_oracle.float()).abs().max() > 1.0


@pytest.mark.rmsnorm
def test_rmsnorm_oot_dispatch():
    """Verify RMSNorm OOT registration: class swap."""
    from vllm.model_executor.layers.layernorm import RMSNorm

    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    layer = RMSNorm(128, eps=1e-6)

    # OOT class swap: RMSNorm.__new__ should produce SpyreRMSNorm
    assert isinstance(layer, SpyreRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot


@pytest.mark.rmsnorm
@pytest.mark.parametrize("var_hidden_size", [16, 32, 48, 96, 100])
def test_rmsnorm_rejects_unaligned_variance_size(default_vllm_config, var_hidden_size):
    """An off-stick reduced axis either crashes the compiler or is silently wrong.

    32 and 96 are the dangerous ones: they lower cleanly and then match no reduction
    width, so the guard has to reject every non-multiple of the 64-element stick, not
    just the ones that fail loudly.
    """
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    with pytest.raises(NotImplementedError, match="var_hidden_size"):
        SpyreRMSNorm(256, eps=1e-6, var_hidden_size=var_hidden_size)


@pytest.mark.rmsnorm
@pytest.mark.parametrize("var_hidden_size", [64, 128, 192])
def test_rmsnorm_stick_aligned_variance_size_runs_on_device(default_vllm_config, var_hidden_size):
    """A stick-aligned reduced axis is accepted and normalizes over that prefix only."""
    import vllm.ir
    from vllm.config import get_current_vllm_config

    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    vllm.ir.set_default_torch_wrap(
        get_current_vllm_config().compilation_config.ir_enable_torch_wrap
    )

    eps, hidden_size = 1e-6, 256
    torch.manual_seed(0)
    x = torch.randn(8, hidden_size, dtype=torch.float16)
    layer = SpyreRMSNorm(hidden_size, eps=eps, var_hidden_size=var_hidden_size).to(torch.float16)
    layer.weight.data = torch.randn(hidden_size, dtype=torch.float16)

    # Only the leading var_hidden_size columns enter the variance.
    expected = reference_rms_norm(x, layer.weight.data, eps, variance_size=var_hidden_size)

    layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre"))

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=_ATOL, rtol=_RTOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
