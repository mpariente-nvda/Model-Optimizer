# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import json

import pytest
import torch
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.export import export_hf_checkpoint


class _NemotronOmniConfig(PretrainedConfig):
    model_type = "nemotron_omni_vision_recipe_test"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_encoder_decoder = False
        self.architectures = ["_NemotronOmniForConditionalGeneration"]


class _RadioBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.qkv = torch.nn.Linear(16, 48, bias=False)
        self.attn.proj = torch.nn.Linear(16, 16, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.fc1 = torch.nn.Linear(16, 32, bias=False)
        self.mlp.fc2 = torch.nn.Linear(32, 16, bias=False)

    def forward(self, x):
        self.attn.qkv(x)
        x = self.attn.proj(x)
        return self.mlp.fc2(torch.nn.functional.gelu(self.mlp.fc1(x)))


class _NemotronOmniForConditionalGeneration(PreTrainedModel):
    config_class = _NemotronOmniConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_model = torch.nn.Module()
        self.vision_model.radio_model = torch.nn.Module()
        self.vision_model.radio_model.model = torch.nn.Module()
        radio = self.vision_model.radio_model.model
        radio.blocks = torch.nn.ModuleList([_RadioBlock()])
        radio.patch_generator = torch.nn.Module()
        radio.patch_generator.embedder = torch.nn.Linear(16, 16, bias=False)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16, bias=False),
        )
        self.language_model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
        self.sound_encoder = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))

    def forward(self, input_ids, **kwargs):
        return self.language_model(input_ids)

    def calibrate_vision(self, x):
        x = self.vision_model.radio_model.model.blocks[0](x)
        return self.mlp1(x)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 vision encoder export requires compute capability 8.9 or newer",
)
def test_nemotron_vision_recipe_calibrates_and_exports(tmp_path):
    model = _NemotronOmniForConditionalGeneration(_NemotronOmniConfig()).to("cuda").eval()
    config = load_recipe("huggingface/nemotron_vl/ptq/fp8_vision-kv_none").quantize.model_dump()
    calibration_input = torch.randn(2, 4, 16, device="cuda")

    mtq.quantize(
        model,
        config,
        forward_loop=lambda _: model.calibrate_vision(calibration_input),
    )
    export_hf_checkpoint(model, export_dir=tmp_path)

    with (tmp_path / "hf_quant_config.json").open() as config_file:
        quantization = json.load(config_file)["quantization"]
    assert quantization["quant_algo"] == "FP8"
    assert quantization["kv_cache_quant_algo"] is None
    assert "language_model*" in quantization["exclude_modules"]
    assert "sound_encoder*" in quantization["exclude_modules"]
    assert "vision_model.radio_model.model.patch_generator*" in quantization["exclude_modules"]

    tensor_names = set()
    for checkpoint_path in tmp_path.glob("*.safetensors"):
        with safe_open(str(checkpoint_path), framework="pt") as checkpoint:
            tensor_names.update(checkpoint.keys())
    scale_names = {name for name in tensor_names if name.endswith(("input_scale", "weight_scale"))}
    assert any(
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight_scale" in name
        for name in scale_names
    )
    assert any("mlp1.1.weight_scale" in name for name in scale_names)
    assert any("mlp1.3.weight_scale" in name for name in scale_names)
    assert not any("language_model" in name for name in scale_names)
    assert not any("sound_encoder" in name for name in scale_names)
    assert not any("patch_generator" in name for name in scale_names)
