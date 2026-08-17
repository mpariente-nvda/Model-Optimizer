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

from types import SimpleNamespace

import pytest
import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils, nemotron_vl_calib


class _FakeQuantizer(torch.nn.Module):
    def __init__(self, enabled):
        super().__init__()
        self.is_enabled = enabled


class _LanguageModel(torch.nn.Module):
    def __init__(self, quantizer_enabled):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.input_quantizer = _FakeQuantizer(quantizer_enabled)
        self.config = SimpleNamespace(torch_dtype=torch.float32)
        self.forward_calls = []
        self.embedding_calls = 0

    def get_input_embeddings(self):
        self.embedding_calls += 1
        return self.embedding

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)


class _Radio(torch.nn.Module):
    def __init__(self, summary_idxs, num_cls_tokens=128):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.model = torch.nn.Module()
        self.model.patch_generator = SimpleNamespace(num_cls_tokens=num_cls_tokens)
        self.register_buffer("summary_idxs", summary_idxs)


class _NemotronOmni(torch.nn.Module):
    def __init__(self, quantizer_enabled, summary_idxs=None, args=None):
        super().__init__()
        if summary_idxs is None:
            summary_idxs = torch.tensor([99])
        self.language_model = _LanguageModel(quantizer_enabled)
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.vision_model = torch.nn.Module()
        self.vision_model.radio_model = _Radio(summary_idxs)
        self.vision_model.config = SimpleNamespace(
            torch_dtype=torch.float32,
            args=args
            or {
                "teachers": [
                    {"use_summary": True},
                    {"use_summary": True},
                    {"use_summary": False},
                ]
            },
        )
        self.img_context_token_id = 18
        self.extract_feature_calls = 0

    def extract_feature(self, pixel_values):
        self.extract_feature_calls += 1
        batch_size = pixel_values.shape[0] if torch.is_tensor(pixel_values) else len(pixel_values)
        return torch.ones(batch_size, 2, 4)

    def forward(self, **kwargs):
        raise AssertionError("The wrapper forward must not be used for Nemotron Omni calibration")


def _batch():
    return {
        "pixel_values": torch.ones(1, 3, 2, 2),
        "input_ids": torch.tensor([[18, 18]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "image_flags": torch.ones(1, 1, dtype=torch.long),
    }


def test_vision_only_calibration_skips_decoder_and_preserves_loaded_radio_indices():
    model = _NemotronOmni(quantizer_enabled=False)

    nemotron_vl_calib.safe_nemotron_vl_forward(model, _batch(), calibrate_language_model=False)

    assert model.extract_feature_calls == 1
    assert model.language_model.forward_calls == []
    assert model.language_model.embedding_calls == 0
    assert torch.equal(model.vision_model.radio_model.summary_idxs, torch.tensor([99]))


def test_missing_radio_indices_are_registered_from_object_config():
    args = SimpleNamespace(
        teachers=[SimpleNamespace(use_summary=True), "default-summary", {"use_summary": False}]
    )
    model = _NemotronOmni(False, summary_idxs=torch.empty(0, device="meta"), args=args)

    with pytest.warns(UserWarning, match="Materializing missing C-RADIO"):
        nemotron_vl_calib.safe_nemotron_vl_forward(model, _batch(), calibrate_language_model=False)

    radio = model.vision_model.radio_model
    assert torch.equal(radio.summary_idxs, torch.tensor([0, 1]))
    assert radio.summary_idxs.device.type == "cpu"
    assert "summary_idxs" not in radio.state_dict()


def test_out_of_range_materialized_radio_indices_are_rebuilt():
    model = _NemotronOmni(False, summary_idxs=torch.tensor([224968, 0, 32]))
    model.vision_model.radio_model.model.patch_generator.num_cls_tokens = 4

    with pytest.warns(UserWarning, match="materialized but invalid"):
        nemotron_vl_calib.safe_nemotron_vl_forward(model, _batch(), calibrate_language_model=False)

    assert torch.equal(model.vision_model.radio_model.summary_idxs, torch.tensor([0, 1]))


def test_nonbuffer_radio_indices_are_reregistered_as_nonpersistent_buffer():
    model = _NemotronOmni(False)
    radio = model.vision_model.radio_model
    del radio._buffers["summary_idxs"]
    radio.summary_idxs = None

    with pytest.warns(UserWarning, match="Materializing missing C-RADIO"):
        nemotron_vl_calib.safe_nemotron_vl_forward(model, _batch(), calibrate_language_model=False)

    assert torch.equal(radio.summary_idxs, torch.tensor([0, 1]))
    assert "summary_idxs" not in radio.state_dict()


def test_missing_radio_indices_warn_when_config_cannot_rebuild_them():
    model = _NemotronOmni(False, summary_idxs=torch.empty(0, device="meta"), args=SimpleNamespace())

    with pytest.warns(UserWarning, match="cannot be rebuilt"):
        nemotron_vl_calib.safe_nemotron_vl_forward(model, _batch(), calibrate_language_model=False)


def test_joint_calibration_runs_decoder_when_its_quantizers_are_enabled():
    model = _NemotronOmni(quantizer_enabled=True)

    nemotron_vl_calib.safe_nemotron_vl_forward(model, _batch(), calibrate_language_model=True)

    assert model.extract_feature_calls == 1
    assert model.language_model.embedding_calls == 1
    assert len(model.language_model.forward_calls) == 1
    assert model.language_model.forward_calls[0]["use_cache"] is False


def test_list_pixel_values_are_supported():
    model = _NemotronOmni(quantizer_enabled=False)
    batch = _batch()
    batch["pixel_values"] = [torch.ones(1, 3, 2, 2)]
    batch.pop("image_flags")

    nemotron_vl_calib.safe_nemotron_vl_forward(model, batch, calibrate_language_model=False)

    assert model.extract_feature_calls == 1


def test_calibration_loop_resolves_decoder_scope_after_quantizers_are_enabled():
    model = _NemotronOmni(quantizer_enabled=False)
    calibration_loop = example_utils.create_vlm_calibration_loop(model, [_batch()])
    model.language_model.input_quantizer.is_enabled = True

    calibration_loop(model)

    assert len(model.language_model.forward_calls) == 1


def test_calibration_loop_caches_scope_across_temporarily_disabled_quantizers():
    model = _NemotronOmni(quantizer_enabled=True)
    calibration_loop = example_utils.create_vlm_calibration_loop(model, [_batch()])

    calibration_loop(model)
    model.language_model.input_quantizer.is_enabled = False
    calibration_loop(model)

    assert len(model.language_model.forward_calls) == 2


def test_calibration_loop_rejects_recipe_that_enables_no_quantizers():
    model = _NemotronOmni(quantizer_enabled=False)
    calibration_loop = example_utils.create_vlm_calibration_loop(model, [_batch()])

    with pytest.raises(RuntimeError, match="found no enabled quantizers"):
        calibration_loop(model)


def test_scope_detects_enabled_later_sequential_quantizer_stage():
    model = _NemotronOmni(quantizer_enabled=False)
    model.language_model.input_quantizer = torch.nn.Sequential(
        _FakeQuantizer(False), _FakeQuantizer(True)
    )

    assert nemotron_vl_calib.resolve_nemotron_calibration_scope(model)


def test_scope_rejects_enabled_audio_quantizers():
    model = _NemotronOmni(quantizer_enabled=False)
    model.sound_encoder = torch.nn.Module()
    model.sound_encoder.input_quantizer = _FakeQuantizer(True)

    with pytest.raises(NotImplementedError, match="audio quantizers"):
        nemotron_vl_calib.resolve_nemotron_calibration_scope(model)
