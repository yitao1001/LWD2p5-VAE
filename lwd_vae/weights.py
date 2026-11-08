"""共享的预训练权重读取与完整性检查；不会静默部分加载。"""
from __future__ import annotations
import hashlib
import pickle
from collections.abc import Mapping
from pathlib import Path
import torch
from torch import Tensor, nn


class _LegacyCheckpointMetadata:
    """旧 Lightning callback 的无行为标记；仅为读取官方 KL-f8 的元数据。

    官方包把 ModelCheckpoint 类作为 callbacks 字典的键保存。我们不导入
    或执行旧 Lightning 类，只用空类代替该标记；仍使用 weights_only=True。
    """


def _load_weights_file(path: Path) -> Mapping:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to read a .safetensors checkpoint") from exc
        checkpoint = load_file(str(path), device="cpu")
    else:
        legacy_marker = (_LegacyCheckpointMetadata,
                         "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint")
        try:
            with torch.serialization.safe_globals([legacy_marker]):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError("Cannot safely read this checkpoint. Supply a tensor state_dict "
                             "or the pinned official KL-f8 file; arbitrary pickle loading is disabled.") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a tensor state dict or contain 'state_dict'")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint 'state_dict' must be a mapping")
    return state


def _remove_wrapper_prefix(key: str) -> str:
    # Strip exact wrapper components only, not arbitrary substring replacements.
    while key.startswith(("first_stage_model.", "model.")):
        key = key.split(".", 1)[1]
    return key


def load_pretrained(model: nn.Module, path: str | Path, *, rgb_to_gray: str = "mean") -> dict:
    """Load a complete CompVis KL-f8 state, optionally adapting RGB boundaries.

    rgb_to_gray='mean' follows the CSEM paper and uploaded inverse-scattering
    script. 'sum' preserves the response to a gray image replicated into RGB
    and is retained only as the historical LWD baseline default. Decoder
    output channels and bias are averaged in both cases.

    Only loss/discriminator tensors may be ignored. Missing weights, unknown
    model weights, or incompatible interior shapes fail before modifying model.
    Use base_channels=128 and the default other settings for the original KL-f8.
    """
    if rgb_to_gray not in ("mean", "sum"):
        raise ValueError("rgb_to_gray must be mean or sum")
    path = Path(path).expanduser().resolve()
    source = _load_weights_file(path)
    expected = model.state_dict()
    loaded = {}
    ignored = []
    adapted = []
    unexpected = []
    mismatched = []
    for raw_key, value in source.items():
        if not isinstance(raw_key, str):
            raise ValueError("State dict keys must be strings")
        key = _remove_wrapper_prefix(raw_key)
        if key.startswith(("loss.", "discriminator.")):
            ignored.append(raw_key)
            continue
        if key not in expected:
            unexpected.append(raw_key)
            continue
        if key in loaded:
            raise ValueError(f"Duplicate normalized checkpoint key: {key}")
        if not isinstance(value, Tensor):
            raise ValueError(f"Checkpoint weight {raw_key!r} is not a tensor")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint weight {raw_key!r} must be finite floating point")
        target = expected[key]
        if value.shape != target.shape:
            if (key == "encoder.conv_in.weight" and value.ndim == 4
                    and value.shape == (target.shape[0], 3, *target.shape[2:])):
                value = value.mean(dim=1, keepdim=True) if rgb_to_gray == "mean" else value.sum(dim=1, keepdim=True)
                adapted.append({"key": key, "operation": f"{rgb_to_gray}_rgb_input_channels"})
            elif (key == "decoder.conv_out.weight" and value.ndim == 4
                  and value.shape == (3, *target.shape[1:])):
                value = value.mean(dim=0, keepdim=True)
                adapted.append({"key": key, "operation": "mean_rgb_output_channels"})
            elif key == "decoder.conv_out.bias" and value.shape == (3,) and target.shape == (1,):
                value = value.mean(dim=0, keepdim=True)
                adapted.append({"key": key, "operation": "mean_rgb_output_channels"})
            else:
                mismatched.append(f"{key}: checkpoint {tuple(value.shape)}, model {tuple(target.shape)}")
        loaded[key] = value
    missing = sorted(set(expected) - set(loaded))
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing ({len(missing)}): {missing[:10]}")
        if unexpected:
            details.append(f"unexpected ({len(unexpected)}): {unexpected[:10]}")
        if mismatched:
            details.append(f"shape mismatch ({len(mismatched)}): {mismatched[:10]}")
        raise ValueError("Incompatible pretrained checkpoint; no weights loaded. " + "; ".join(details))
    model.load_state_dict(loaded, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rgb_to_gray": rgb_to_gray,
        "matched_tensor_count": len(loaded),
        "adapted_tensor_count": len(adapted),
        "adapted": adapted,
        "ignored_tensor_count": len(ignored),
        "ignored": ignored,
    }
