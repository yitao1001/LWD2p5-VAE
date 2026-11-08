"""在训练前用真实数据和权重走一次前向；不创建实验目录或更新参数。"""
from __future__ import annotations

import torch

from ..dataset import LWDVAEDataset
from ..runtime import select_device
from ..weights import load_pretrained
from .config import missing_inputs, resolve_paths, validate_config
from .losses import VAELoss
from .model import CompVisKLF8


def check_setup(config, device_name="cpu"):
    config = validate_config(config)
    missing = missing_inputs(config)
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    paths, device = resolve_paths(config), select_device(device_name)
    torch.set_num_threads(config["training"]["cpu_threads"])
    splits = {}
    for split in ("train", "val"):
        with LWDVAEDataset(paths["dataset"], split, as_torch=True) as dataset:
            dataset.validate_all()
            if not len(dataset):
                raise ValueError(f"Empty {split} split")
            shape = dataset.image_shape[-2:]
            if any(v % 8 for v in shape) or min(shape) < 8:
                raise ValueError("Grid dimensions must be positive multiples of 8; no resize")
            if config["loss"]["feature"] != "none" and min(shape) < 32:
                raise ValueError("AlexNet requires H,W >= 32")
            splits[split] = {"samples": len(dataset), "shape": list(dataset.image_shape)}
            sample = dataset[0].unsqueeze(0).to(device)
    model = CompVisKLF8(config["model"])
    initialization = {"mode": config["initialization"]["mode"]}
    if initialization["mode"] == "pretrained":
        initialization.update(load_pretrained(model, paths["vae_weights"], rgb_to_gray="mean"))
    model.to(device).eval()
    criterion = VAELoss(config["loss"], paths["alexnet_weights"]).to(device)
    with torch.no_grad():
        reconstructed, posterior = model(sample, sample_posterior=False)
        terms = criterion(reconstructed, sample, posterior, 0)
        if reconstructed.shape != sample.shape or not all(torch.isfinite(v).all() for v in terms.values()):
            raise ValueError("Model/loss preflight failed")
    return {"purpose": config["purpose"], "device": str(device), "splits": splits,
            "initialization": initialization, "posterior_shape": list(posterior.mean.shape[1:]),
            "trainable_vae_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_feature_parameters": sum(p.numel() for p in criterion.parameters()),
            "sample_losses": {k: v.item() for k, v in terms.items()},
            "status": "weights, dataset and forward checked; no optimizer update"}
