# Nemotron VL PTQ recipes

Nemotron VL is a vision-language model family (including Nemotron-Parse and
Nemotron 3 Nano Omni). Most recipes in this directory quantize only the decoder.
The opt-in FP8 vision recipe is validated on Nemotron 3 Nano Omni.

| File | What's model-specific |
|------|-----------------------|
| `disabled_quantizers.yaml` | Reusable unit (`QuantizerCfgListConfig`). Merges the standard `default_disabled_quantizers` exclusions with Nemotron-VL ones (`*vision*`, `*image*`, `*radio*`, `*visual*`, `*encoder*`, `*model_encoder*`). The last two patterns are required for Nemotron-Parse. Imported by recipes below as the single `disabled_quantizers` slot so they don't pull in two disabled-quantizer sets. |
| `nvfp4-kv_fp8_cast.yaml` | NVFP4 W4A4 model quantization + FP8 KV-cache cast (constant amax, no KV calibration). Identical numerics to the general `nvfp4` preset / `kv_fp8_cast` unit; what makes it model-specific is that it imports `disabled_quantizers.yaml` from this folder to skip the vision/encoder branches. |
| `vision_fp8.quant_cfg.yaml` | FP8 configuration for the known Nemotron 3 Nano Omni C-RADIO and vision-projector module layouts. Patch generation stays in high precision. |
| `fp8_vision-kv_none.yaml` | Vision-only FP8 recipe validated on Nemotron 3 Nano Omni. The language model, audio branch, and KV cache stay in high precision. |

Additional `<qformat>-kv_fp8_cast.yaml` recipes can be generated for other formats
if needed; only `nvfp4-kv_fp8_cast.yaml` is shipped by default.
