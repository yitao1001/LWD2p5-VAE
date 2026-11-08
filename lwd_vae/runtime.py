"""Device selection, portable checkpoints and explicit dataset contracts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Supported devices are cpu, mps, cuda and cuda:N")
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError(f"Requested CUDA device is unavailable: {device}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng():
    np_state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
              "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG resume is unavailable")
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def data_contract(dataset, include_training_hashes=True):
    result = {"grid": dataset.config["grid"], "log10_bounds": list(dataset.log10_bounds),
              "coordinate_frame": dataset.coordinate_frame,
              "manifest_sha256": sha256_file(dataset.root / "manifest.json"),
              "array_order": "N,1,z,y", "parameterization": "log10(rho / (ohm m))"}
    if include_training_hashes:
        result["training_split_hashes"] = {}
        for name in ("train", "val"):
            split = dataset.manifest["splits"][name]
            result["training_split_hashes"][name] = {
                key: sha256_file(dataset.root / split[key]) for key in ("path", "metadata")}
    return result


def same_data_contract(first, second):
    """旧 checkpoint 未写 frame 字段，但原约定同样是全局 y/z；其余字段严格比较。"""
    return ({**first, "coordinate_frame": first.get("coordinate_frame", "global_aligned_yz")}
            == {**second, "coordinate_frame": second.get("coordinate_frame", "global_aligned_yz")})


def validate_data_contract(contract, dataset, require_manifest=False):
    if contract.get("coordinate_frame", "global_aligned_yz") != dataset.coordinate_frame:
        raise ValueError("Dataset coordinate frame differs from checkpoint; no implicit rotation is allowed")
    if contract["grid"] != dataset.config["grid"] or list(contract["log10_bounds"]) != list(dataset.log10_bounds):
        raise ValueError("Dataset grid/physical coordinates or log10 normalization differ from checkpoint; no implicit resize/rescale is allowed")
    if require_manifest:
        actual = data_contract(dataset, include_training_hashes="training_split_hashes" in contract)
        for key in ("manifest_sha256", "training_split_hashes"):
            if key in contract and actual.get(key) != contract[key]:
                raise ValueError(f"Resume dataset identity mismatch: {key}")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, state):
    """Atomically replace this run's checkpoint with CPU-portable tensors."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(cpu_tree(state), temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") not in ("lwd_vae_training_v1", "lwd_vae_finetuning_v2"):
        raise ValueError("Expected a training checkpoint from train_vae.py; use --pretrained for external weights")
    for key in ("model_spec", "model_state", "data_contract", "epoch", "training_config"):
        if key not in checkpoint:
            raise ValueError(f"Missing checkpoint field: {key}")
    if checkpoint["format"] == "lwd_vae_finetuning_v2":
        from .finetuning.model import CompVisKLF8, upstream_provenance
        if checkpoint.get("model_backend") != "compvis_original_core" or checkpoint.get("upstream_commit") != upstream_provenance()["commit"]:
            raise ValueError("Checkpoint CompVis backend/version differs from the installed reference source")
        model = CompVisKLF8(checkpoint["model_spec"])
    else:
        from .model import AutoencoderKL
        model = AutoencoderKL(checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(select_device(str(device)))
    return model, checkpoint


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
