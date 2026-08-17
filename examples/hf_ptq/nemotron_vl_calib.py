# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Nemotron VL calibration helpers.

Nemotron Nano VL v2 remote-code wrapper `forward()` is not ideal to call during PTQ calibration because it may:
- Call `torch.distributed.get_rank()` unconditionally
- Assume `past_key_values` exists in the language model output

Instead, we run a "safe multimodal forward" that exercises:
- Vision encoder feature extraction (C-RADIO)
- Vision-projector calibration
- For joint recipes only, insertion of vision embeddings and language-model calibration
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Mapping
from itertools import chain
from typing import Any

import torch

__all__ = [
    "materialize_radio_summary_idxs",
    "resolve_nemotron_calibration_scope",
    "safe_nemotron_vl_forward",
]


def _enabled_quantizer_names(module: torch.nn.Module) -> list[str]:
    """Return enabled leaf quantizers, including SequentialQuantizer stages."""
    return [
        name
        for name, submodule in module.named_modules()
        if "quantizer" in name
        and getattr(submodule, "is_enabled", False)
        and not any(hasattr(child, "is_enabled") for child in submodule.children())
    ]


def _has_path_component(name: str, keywords: tuple[str, ...]) -> bool:
    """Whether any dotted component of `name` contains one of `keywords`."""
    return any(keyword in part for part in name.split(".") for keyword in keywords)


def resolve_nemotron_calibration_scope(full_model: torch.nn.Module) -> bool:
    """Validate enabled Nemotron quantizers and return whether to run the decoder."""
    enabled = _enabled_quantizer_names(full_model)
    if not enabled:
        raise RuntimeError(
            "Nemotron multimodal calibration found no enabled quantizers. Check that "
            "the selected recipe matches this model's module paths."
        )

    # Match on dotted components rather than a fixed prefix list so that audio
    # branches nested under another module, or named differently, are still caught.
    enabled_audio = [name for name in enabled if _has_path_component(name, ("sound", "audio"))]
    if enabled_audio:
        raise NotImplementedError(
            "Image-text calibration cannot calibrate enabled Nemotron audio quantizers: "
            f"{enabled_audio[:3]}. Use an audio-capable calibration loop."
        )

    vision_prefixes = (
        "vision_model.",
        "model.vision_model.",
        "mlp1.",
        "model.mlp1.",
    )
    return any(not name.startswith(vision_prefixes) for name in enabled)


def _radio_device(radio: torch.nn.Module) -> torch.device | None:
    for tensor in chain(radio.parameters(), radio.buffers()):
        if tensor.device.type != "meta" and tensor is not getattr(radio, "summary_idxs", None):
            return tensor.device
    return None


def _register_radio_summary_idxs(
    radio: torch.nn.Module, summary_idxs: torch.Tensor, *, persistent: bool
) -> None:
    """Install `summary_idxs` as a buffer.

    `persistent=True` keeps an existing registration untouched (only the tensor is
    swapped), so a checkpoint-provided buffer we merely move across devices stays
    serialized by export. `persistent=False` forces a non-persistent buffer, so a
    value we reconstruct ourselves never leaks into the exported checkpoint.
    """
    if persistent and "summary_idxs" in radio._buffers:
        radio._buffers["summary_idxs"] = summary_idxs
        return

    if "summary_idxs" in radio._buffers:
        del radio._buffers["summary_idxs"]
        radio._non_persistent_buffers_set.discard("summary_idxs")
    elif hasattr(radio, "summary_idxs"):
        try:
            delattr(radio, "summary_idxs")
        except AttributeError:
            # A class-level attribute or property cannot be deleted per instance.
            # Register directly rather than let a defensive repair crash.
            radio._buffers["summary_idxs"] = summary_idxs
            if not persistent:
                radio._non_persistent_buffers_set.add("summary_idxs")
            return

    radio.register_buffer("summary_idxs", summary_idxs, persistent=persistent)


def materialize_radio_summary_idxs(full_model: torch.nn.Module) -> None:
    """Materialize C-RADIO's summary-token indices only when loading left no buffer.

    A valid checkpoint-provided buffer is authoritative and must never be replaced.
    Some low-memory loading paths can instead leave the derived buffer on ``meta``,
    omit it, or materialize uninitialized out-of-range values; only those cases are
    reconstructed from the C-RADIO config.
    """
    radio = getattr(getattr(full_model, "vision_model", None), "radio_model", None)
    if radio is None:
        return

    device = _radio_device(radio)
    patch_generator = getattr(getattr(radio, "model", None), "patch_generator", None)
    num_cls_tokens = getattr(patch_generator, "num_cls_tokens", None)

    current = getattr(radio, "summary_idxs", None)
    invalid_materialized_buffer = False
    if isinstance(current, torch.Tensor) and current.device.type != "meta":
        # Transformers' low-memory loader can report this persistent buffer as
        # MISSING yet leave a materialized, uninitialized tensor behind. Preserve
        # any in-range checkpoint value (even if it differs from the config), but
        # rebuild values that cannot index C-RADIO's class-token dimension.
        if getattr(radio, "_modelopt_summary_idxs_validated", False):
            return
        # An empty buffer is broken whatever the config says: it is exactly what a
        # low-memory loader leaves behind, and C-RADIO would then return an empty
        # summary instead of raising. Range is only checkable when the class-token
        # count is known; when it is not, a non-empty checkpoint value is trusted.
        valid = current.dtype in (torch.int32, torch.int64) and current.numel() > 0
        if valid and num_cls_tokens is not None:
            valid = bool(((current >= 0) & (current < num_cls_tokens)).all().item())
        if valid:
            if device is not None and current.device != device:
                _register_radio_summary_idxs(radio, current.to(device=device), persistent=True)
            radio._modelopt_summary_idxs_validated = True
            return
        invalid_materialized_buffer = True

    vision_config = getattr(getattr(full_model, "vision_model", None), "config", None)
    args = getattr(vision_config, "args", None)
    teachers = (
        args.get("teachers") if isinstance(args, Mapping) else getattr(args, "teachers", None)
    )
    if teachers is None:
        warnings.warn(
            "C-RADIO summary-token indices are not materialized and cannot be rebuilt: "
            "vision_config.args.teachers is unavailable."
        )
        return
    if device is None:
        warnings.warn(
            "C-RADIO summary-token indices cannot be materialized because the vision "
            "module has no materialized parameter or buffer."
        )
        return

    expected = []
    for index, teacher in enumerate(teachers):
        if isinstance(teacher, Mapping):
            use_summary = teacher.get("use_summary", True)
        else:
            use_summary = getattr(teacher, "use_summary", True)
        if use_summary:
            expected.append(index)

    if invalid_materialized_buffer:
        warnings.warn(
            "C-RADIO summary-token indices are materialized but invalid for the "
            "available class tokens; rebuilding them from the vision configuration."
        )
    else:
        warnings.warn(
            "Materializing missing C-RADIO summary-token indices from the vision configuration."
        )
    summary_idxs = torch.tensor(expected, dtype=torch.int64, device=device)

    # Hold the rebuilt value to the same predicate that rejected the old one:
    # `expected` indexes teachers, which need not agree with the class tokens the
    # model actually has. Registering an out-of-range buffer here would surface as
    # an IndexError deep inside C-RADIO, right after a warning blaming the loader.
    out_of_range = num_cls_tokens is not None and (not expected or max(expected) >= num_cls_tokens)
    _register_radio_summary_idxs(radio, summary_idxs, persistent=False)
    if out_of_range:
        warnings.warn(
            "Rebuilt C-RADIO summary-token indices are inconsistent with the model: "
            f"the vision config yields indices {expected} but the patch generator "
            f"exposes only {num_cls_tokens} class token(s). The vision forward will "
            "likely fail; check vision_config.args.teachers against the checkpoint."
        )
        return
    radio._modelopt_summary_idxs_validated = True


def safe_nemotron_vl_forward(
    full_model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    calibrate_language_model: bool | None = None,
) -> None:
    """Run a minimal multimodal forward for Nemotron VL that avoids wrapper output packaging."""
    pixel_values = batch.get("pixel_values")
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    position_ids = batch.get("position_ids")
    image_flags = batch.get("image_flags")

    if pixel_values is None or input_ids is None:
        return

    # Nemotron Nano VL v2 expects `image_flags` in forward(), but the processor doesn't always emit it.
    # `pixel_values` is flattened across batch*images, so `image_flags` should align with pixel_values.shape[0].
    if image_flags is None and torch.is_tensor(pixel_values):
        image_flags = torch.ones(
            (pixel_values.shape[0], 1), device=pixel_values.device, dtype=torch.long
        )
    elif image_flags is None and isinstance(pixel_values, (list, tuple)) and pixel_values:
        first_pixel_value = pixel_values[0]
        # Nemotron tiles images, so each entry is [num_tiles, 3, H, W]. The flags
        # must align with the vision embeddings, i.e. with the total tile count,
        # not with the number of images.
        num_tiles = sum(value.shape[0] if value.dim() == 4 else 1 for value in pixel_values)
        image_flags = torch.ones((num_tiles, 1), device=first_pixel_value.device, dtype=torch.long)
    if image_flags is None:
        return

    # Match the model's preferred vision dtype (usually bf16).
    vision_dtype = None
    with contextlib.suppress(AttributeError, TypeError):
        vision_dtype = getattr(full_model.vision_model.config, "torch_dtype", None)
    if vision_dtype is None:
        with contextlib.suppress(AttributeError, TypeError):
            vision_dtype = getattr(full_model.language_model.config, "torch_dtype", None)
    if vision_dtype is not None:
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dtype != vision_dtype:
            pixel_values = pixel_values.to(dtype=vision_dtype)
        elif isinstance(pixel_values, (list, tuple)):
            if not all(isinstance(value, torch.Tensor) for value in pixel_values):
                raise TypeError("Nemotron pixel_values entries must be tensors.")
            pixel_values = type(pixel_values)(
                value.to(dtype=vision_dtype) if value.dtype != vision_dtype else value
                for value in pixel_values
            )

    materialize_radio_summary_idxs(full_model)

    # Vision embeddings. Note `extract_feature` also runs the `mlp1` projector, so
    # a vision-scope recipe has every one of its quantizers exercised by this call.
    vit_embeds = full_model.extract_feature(pixel_values)

    if calibrate_language_model is None:
        calibrate_language_model = resolve_nemotron_calibration_scope(full_model)
    # An undetermined scope (None) runs the decoder: collecting statistics that go
    # unused is recoverable, skipping a decoder that needed them is not.
    if calibrate_language_model is False:
        # A vision-only recipe has now exercised the vision tower and projector.
        # Nothing downstream collects calibration statistics, so avoid an
        # unnecessary decoder forward.
        return

    # Token embeddings are needed only when decoder quantizers are calibrated too.
    inputs_embeds = full_model.language_model.get_input_embeddings()(input_ids)
    image_flags_s = image_flags.to(device=vit_embeds.device).squeeze(-1)

    b, n, c = inputs_embeds.shape
    flat_embeds = inputs_embeds.reshape(b * n, c)
    flat_ids = input_ids.reshape(b * n)
    selected = flat_ids == full_model.img_context_token_id

    vit_embeds = vit_embeds[image_flags_s == 1]
    try:
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, c)
    except Exception as exc:
        # Truncating silently would feed the decoder mismatched vision embeddings
        # and quietly skew every activation statistic collected from this batch.
        warnings.warn(
            f"Vision embeddings do not line up with the image context tokens ({exc}); "
            "truncating to the available tokens. Calibration statistics from this "
            "batch may be unrepresentative."
        )
        vit_embeds = vit_embeds.reshape(-1, c)
        n_token = selected.sum()
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds[:n_token]

    inputs_embeds = flat_embeds.reshape(b, n, c)

    # LLM forward (drives activation stats)
    full_model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=False,
    )
