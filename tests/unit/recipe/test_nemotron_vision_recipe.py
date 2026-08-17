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

import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe


class _RadioAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 48, bias=False)
        self.proj = nn.Linear(16, 16, bias=False)


class _RadioBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _RadioAttention()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(16, 32, bias=False)
        self.mlp.fc2 = nn.Linear(32, 16, bias=False)


class _NemotronOmni(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.radio_model = nn.Module()
        self.vision_model.radio_model.model = nn.Module()
        radio = self.vision_model.radio_model.model
        radio.blocks = nn.ModuleList([_RadioBlock()])
        radio.patch_generator = nn.Module()
        radio.patch_generator.embedder = nn.Linear(16, 16, bias=False)
        radio.patch_generator.video_embedder = nn.Linear(16, 16, bias=False)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, 32, bias=False),
            nn.GELU(),
            nn.Linear(32, 16, bias=False),
        )
        self.model = nn.Module()
        self.model.vision_model = nn.Module()
        self.model.vision_model.radio_model = nn.Module()
        self.model.vision_model.radio_model.model = nn.Module()
        wrapped_radio = self.model.vision_model.radio_model.model
        wrapped_radio.blocks = nn.ModuleList([_RadioBlock()])
        wrapped_radio.patch_generator = nn.Module()
        wrapped_radio.patch_generator.embedder = nn.Linear(16, 16, bias=False)
        self.model.mlp1 = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, 32, bias=False),
            nn.GELU(),
            nn.Linear(32, 16, bias=False),
        )
        self.language_model = nn.Sequential(nn.Linear(16, 16, bias=False))
        self.sound_encoder = nn.Sequential(nn.Linear(16, 16, bias=False))
        self.sound_projection = nn.Sequential(nn.Linear(16, 16, bias=False))
        self.sound_mlp1 = nn.Sequential(nn.Linear(16, 16, bias=False))
        self.language_model.mlp1 = nn.Sequential(nn.Linear(16, 16, bias=False))
        self.patch_generator = nn.Sequential(nn.Linear(16, 16, bias=False))


def test_nemotron_vision_recipe_quantizes_only_radio_and_projector_linears():
    model = _NemotronOmni()
    config = load_recipe("huggingface/nemotron_vl/ptq/fp8_vision-kv_none").quantize.model_dump()

    mtq.quantize(model, config, forward_loop=None)
    modules = dict(model.named_modules())
    enabled = {name for name, module in modules.items() if getattr(module, "is_enabled", False)}

    expected_linears = {
        "vision_model.radio_model.model.blocks.0.attn.qkv",
        "vision_model.radio_model.model.blocks.0.attn.proj",
        "vision_model.radio_model.model.blocks.0.mlp.fc1",
        "vision_model.radio_model.model.blocks.0.mlp.fc2",
        "mlp1.1",
        "mlp1.3",
        "model.vision_model.radio_model.model.blocks.0.attn.qkv",
        "model.vision_model.radio_model.model.blocks.0.attn.proj",
        "model.vision_model.radio_model.model.blocks.0.mlp.fc1",
        "model.vision_model.radio_model.model.blocks.0.mlp.fc2",
        "model.mlp1.1",
        "model.mlp1.3",
    }
    expected_quantizers = {
        f"{linear}.{quantizer}"
        for linear in expected_linears
        for quantizer in ("weight_quantizer", "input_quantizer")
    }
    assert expected_quantizers <= enabled
    assert not any("patch_generator" in name for name in enabled)
    assert not any(name.startswith("language_model.") for name in enabled)
    assert not any(name.startswith("sound_") for name in enabled)
    assert not any(name.endswith("_bmm_quantizer") for name in enabled)

    vision_weight = modules["vision_model.radio_model.model.blocks.0.attn.qkv.weight_quantizer"]
    vision_input = modules["vision_model.radio_model.model.blocks.0.attn.qkv.input_quantizer"]
    assert vision_weight.num_bits == (4, 3)
    assert vision_input.num_bits == (4, 3)
