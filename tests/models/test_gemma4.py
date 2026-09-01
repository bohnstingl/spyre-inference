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

"""Unit tests for the Gemma-4 text-backbone selection and weight rebase."""

from __future__ import annotations

import pickle
from collections import Counter

from transformers import PretrainedConfig
from vllm.engine.arg_utils import EngineArgs

from spyre_inference.models.gemma4 import force_text_backbone
from spyre_inference.transformers_backend import _rebase_onto_text_backbone


def _multimodal_config() -> PretrainedConfig:
    """A nested config shaped like Gemma4Config: text stack under ``text_config``."""
    config = PretrainedConfig()
    config.text_config = PretrainedConfig()
    config.text_config.num_hidden_layers = 48
    config.vision_config = PretrainedConfig()
    return config


def _engine_args(model: str = "google/gemma-4-31b", **kwargs) -> EngineArgs:
    return EngineArgs(model=model, max_model_len=128, **kwargs)


def test_text_config_replaces_the_multimodal_one():
    """The whole point: hf_config becomes the text config, so the Transformers backend
    resolves TransformersForCausalLM over HF's text model instead of the multimodal one
    whose forward compares int32 input_ids against the image/video/audio token ids."""
    engine_args = _engine_args()
    force_text_backbone(engine_args)

    config = _multimodal_config()
    text_config = engine_args.hf_overrides(config)

    assert text_config is config.text_config
    assert text_config.architectures == ["Gemma4ForCausalLM"]
    assert text_config._spyre_text_backbone_prefix == "model.language_model."


def test_other_models_are_left_alone():
    engine_args = _engine_args(model="Qwen/Qwen3-0.6B")
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides == {}


def test_a_pinned_architecture_wins():
    """The user asking for a specific architecture means they own the choice."""
    engine_args = _engine_args(hf_overrides={"architectures": ["Gemma4ForConditionalGeneration"]})
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides == {"architectures": ["Gemma4ForConditionalGeneration"]}


def test_a_user_dict_lands_on_the_text_config():
    """Installing a callable takes vLLM's dict handling out of the loop, so the dict form
    of hf_overrides has to be applied here — flat keys and nested sub-configs alike."""
    engine_args = _engine_args(
        hf_overrides={"num_hidden_layers": 2, "vision_config": {"num_hidden_layers": 1}}
    )
    force_text_backbone(engine_args)

    config = _multimodal_config()
    config.text_config.vision_config = PretrainedConfig()
    text_config = engine_args.hf_overrides(config)

    assert text_config.num_hidden_layers == 2
    assert text_config.vision_config.num_hidden_layers == 1


def test_a_user_callable_is_chained():
    def user_override(config):
        config.text_config.num_hidden_layers = 4
        return config

    engine_args = _engine_args(hf_overrides=user_override)
    force_text_backbone(engine_args)

    text_config = engine_args.hf_overrides(_multimodal_config())

    assert text_config.num_hidden_layers == 4
    assert text_config.architectures == ["Gemma4ForCausalLM"]


def test_a_text_only_config_passes_through_untouched():
    """A text-only re-upload has nothing to unnest, and needs no weight rebase."""
    engine_args = _engine_args()
    force_text_backbone(engine_args)

    config = PretrainedConfig()
    config.architectures = ["Gemma4ForCausalLM"]

    assert engine_args.hf_overrides(config) is config
    assert not hasattr(config, "_spyre_text_backbone_prefix")


def test_the_override_survives_pickling():
    """VllmConfig is pickled to the engine-core process; a local closure would not be."""
    engine_args = _engine_args()
    force_text_backbone(engine_args)

    override = pickle.loads(pickle.dumps(engine_args.hf_overrides))

    assert override(_multimodal_config()).architectures == ["Gemma4ForCausalLM"]


def test_weights_are_rebased_off_the_text_prefix():
    """The checkpoint nests the text stack; the module tree (a text config) does not."""
    weights = [
        ("model.language_model.embed_tokens.weight", 0),
        ("model.language_model.layers.0.self_attn.q_proj.weight", 1),
        ("model.vision_tower.encoder.layers.0.mlp.fc1.weight", 2),
        ("model.embed_vision", 3),
        ("lm_head.weight", 4),
    ]
    dropped: Counter[str] = Counter()

    rebased = list(_rebase_onto_text_backbone(iter(weights), "model.language_model.", dropped))

    assert rebased == [
        ("model.embed_tokens.weight", 0),
        ("model.layers.0.self_attn.q_proj.weight", 1),
        ("lm_head.weight", 4),
    ]
    assert dropped == {"vision_tower": 1, "embed_vision": 1}
