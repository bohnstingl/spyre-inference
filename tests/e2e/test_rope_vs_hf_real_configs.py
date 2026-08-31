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

"""The Spyre rotation must equal HF's own, on the *real* configs of the models we ship.

The small hand-built configs elsewhere in the suite miss what these two disagree on:
Gemma 4's text config is heterogeneous (``head_dim`` varies per layer, 256 on sliding
layers and 512 on full-attention ones) and its ``rope_parameters`` is keyed per layer
type. CPU-only and weightless -- config plus rotary module, no checkpoint, no device.
"""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch
from transformers import AutoConfig

from spyre_inference.transformers_backend import (
    _rope_frequencies,
    _rope_matmul_tokens_major,
    _SpyreRotaryEmbedding,
)

MODELS = ["google/gemma-4-31b", "ibm-granite/granite-3.3-8b-base"]
MAX_POSITION = 512


def _hf_rotary(model: str):
    """The HF text config, its rotary module and its ``apply_rotary_pos_emb``."""
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    # Gemma 4 raises on a bare `head_dim` read; the widths we need come from the
    # per-layer-type inv_freq, so the global value is only read for reporting.
    text_cfg.allow_global_per_layer_attribute_access = True

    modeling = importlib.import_module(
        type(text_cfg).__module__.replace("configuration_", "modeling_")
    )
    names = [n for n in dir(modeling) if n.endswith("RotaryEmbedding")]
    text_names = [n for n in names if n.endswith("TextRotaryEmbedding")]
    rot_cls = getattr(modeling, (text_names or names)[0])
    return text_cfg, rot_cls(config=text_cfg), modeling.apply_rotary_pos_emb


@pytest.mark.cpu
@pytest.mark.parametrize("model", MODELS)
def test_spyre_rotation_matches_hf(model, default_vllm_config):
    text_cfg, hf_rope, apply_fn = _hf_rotary(model)
    del text_cfg

    params = inspect.signature(apply_fn).parameters
    takes_qk_pair = "k" in params or "key" in params

    # fp32 throughout so a real error cannot hide under fp16 rounding.
    spyre_rope = _SpyreRotaryEmbedding(hf_rope, MAX_POSITION, None, torch.float32)

    batch, seq_len, heads = 1, 6, 2
    positions = torch.arange(seq_len).unsqueeze(0)
    generator = torch.Generator().manual_seed(0)

    frequencies = _rope_frequencies(hf_rope)
    assert frequencies, "no rotary frequencies found on the HF module"

    for layer_type, (inv_freq, _) in frequencies.items():
        head_dim = 2 * inv_freq.shape[0]
        x = torch.randn(
            batch, seq_len, heads, head_dim, generator=generator, dtype=torch.float32
        )
        extra = () if layer_type is None else (layer_type,)

        cos, sin = hf_rope(x, positions, *extra)
        if takes_qk_pair:
            x_heads_major = x.transpose(1, 2)
            expected = apply_fn(
                x_heads_major, x_heads_major, cos, sin, unsqueeze_dim=1
            )[0].transpose(1, 2)
        else:
            expected = apply_fn(x, cos, sin, unsqueeze_dim=2)

        rot, _ = spyre_rope(x, positions, *extra)
        actual = _rope_matmul_tokens_major(x, rot)

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, lt=layer_type, hd=head_dim: (
                f"{model} layer_type={lt!r} head_dim={hd}: {m}"
            ),
        )
