"""Reproducible model-space VAE training; only train/val splits are opened."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import LWDVAEDataset
from .model import AutoencoderKL, load_pretrained
from .runtime import (capture_rng, cpu_tree, data_contract, load_checkpoint, restore_rng,
                      save_checkpoint, seed_everything, select_device, sha256_file,
                      validate_data_contract, write_json)

DEFAULT_TRAINING_KEYS = {"seed", "epochs", "batch_size", "learning_rate", "weight_decay",
    "reconstruction", "gradient_weight", "kl_weight", "kl_warmup_epochs", "grad_clip_norm",
    "patience", "min_delta", "workers", "cpu_threads", "precision", "max_train_samples", "max_val_samples"}


def validate_training_config(config):
    c = deepcopy(config)
    if not isinstance(c, dict) or set(c) != {"model", "training"}:
        raise ValueError("Training config must contain exactly model and training")
    t = c["training"]
    if not isinstance(t, dict) or set(t) != DEFAULT_TRAINING_KEYS:
        raise ValueError(f"training keys must be exactly {sorted(DEFAULT_TRAINING_KEYS)}")
    for key in ("seed", "epochs", "batch_size", "kl_warmup_epochs", "patience", "workers", "cpu_threads"):
        minimum = 1 if key in ("epochs", "batch_size", "cpu_threads") else 0
        if isinstance(t[key], bool) or not isinstance(t[key], int) or t[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("learning_rate", "weight_decay", "gradient_weight", "kl_weight", "grad_clip_norm", "min_delta"):
        value = t[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if t["learning_rate"] == 0 or t["grad_clip_norm"] == 0:
        raise ValueError("learning_rate and grad_clip_norm must be positive")
    for key in ("max_train_samples", "max_val_samples"):
        if t[key] is not None and (isinstance(t[key], bool) or not isinstance(t[key], int) or t[key] < 1):
            raise ValueError(f"{key} must be null or a positive integer")
    if t["precision"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("precision must be float32, float16 or bfloat16")
    if t["reconstruction"] not in ("mse", "l1"):
        raise ValueError("reconstruction must be mse or l1")
    return c


def load_training_config(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate config key: {key}")
            result[key] = value
        return result
    return validate_training_config(json.loads(Path(path).read_text(), object_pairs_hook=unique))


def reconstruction_terms(prediction, target, posterior, reconstruction="mse"):
    """Means per pixel/edge/latent element; weights don't hide image-size sums."""
    p, y = prediction.float(), target.float()
    pixel = F.mse_loss(p, y) if reconstruction == "mse" else F.l1_loss(p, y)
    gradient = 0.5 * (F.l1_loss(p[..., 1:, :] - p[..., :-1, :], y[..., 1:, :] - y[..., :-1, :]) +
                      F.l1_loss(p[..., :, 1:] - p[..., :, :-1], y[..., :, 1:] - y[..., :, :-1]))
    # Compute in float32 even when convolution uses autocast.
    mean, logvar = posterior.mean.float(), posterior.logvar.float()
    kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
    return pixel, gradient, kl


def kl_coefficient(settings, epoch):
    warmup = settings["kl_warmup_epochs"]
    return settings["kl_weight"] * (min(1.0, (epoch + 1) / warmup) if warmup else 1.0)


def _loader(dataset, settings, train, epoch=0):
    cap = settings["max_train_samples" if train else "max_val_samples"]
    subset = Subset(dataset, list(range(min(cap, len(dataset))))) if cap is not None else dataset
    if len(subset) == 0:
        raise ValueError("Train and validation splits must both be nonempty")
    generator = torch.Generator().manual_seed(settings["seed"] + epoch * 2 + int(train))
    return DataLoader(subset, batch_size=settings["batch_size"], shuffle=train,
                      num_workers=settings["workers"], drop_last=False, generator=generator,
                      persistent_workers=False)


def validate_epoch(model, loader, device, bounds):
    """Deterministic posterior mode, raw log10rho errors, no decoder clipping."""
    model.eval()
    scale = (bounds[1] - bounds[0]) / 2
    n_pixels, square_sum, absolute_sum, outside = 0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            target = batch.to(device)
            pred = model.decode(model.encode(target).mode())
            if not torch.isfinite(pred).all():
                raise FloatingPointError("Non-finite validation reconstruction")
            # Accumulate physical errors in CPU float64 (also works on MPS,
            # which has no float64 tensor support). Do not overflow FP32 squares.
            diff = (pred.detach().cpu().double() - target.detach().cpu().double()) * scale
            square_sum += float(diff.square().sum())
            absolute_sum += float(diff.abs().sum())
            n_pixels += diff.numel()
            outside += int(((pred < -1) | (pred > 1)).sum())
    return {"val_rmse_log10": math.sqrt(square_sum / n_pixels),
            "val_mae_log10": absolute_sum / n_pixels, "val_out_of_bounds_fraction": outside / n_pixels}


def _history_csv(path, history):
    with Path(path).open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(dataset_root, output, config=None, *, device_name="auto", pretrained=None, resume=None, epochs=None):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new run directory; refusing overwrite: {output}")
    if resume and (pretrained or config is not None):
        raise ValueError("--resume restores its saved config/weights; do not combine with --config or --pretrained")
    device = select_device(device_name)
    resume_state, source_best = None, None
    if resume:
        model, resume_state = load_checkpoint(resume, device="cpu")
        c = {"model": resume_state["model_spec"], "training": resume_state["training_config"]}
    elif config is not None:
        c = deepcopy(config)
    else:
        raise ValueError("A config is required for a new run")
    if epochs is not None:
        c["training"]["epochs"] = epochs
    c = validate_training_config(c)
    t = c["training"]
    if t["precision"] != "float32" and device.type != "cuda":
        raise ValueError("Mixed precision is supported only on CUDA; use float32 on CPU/MPS")
    if t["precision"] == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Selected CUDA device does not support bfloat16")
    torch.set_num_threads(t["cpu_threads"])
    seed_everything(t["seed"])
    train_ds = LWDVAEDataset(dataset_root, "train", as_torch=True)
    val_ds = LWDVAEDataset(dataset_root, "val", as_torch=True)
    try:
        train_ds.validate_all()
        val_ds.validate_all()
        if not len(train_ds) or not len(val_ds):
            raise ValueError("Train and validation splits must both be nonempty")
    finally:
        train_ds.close()
        val_ds.close()
    contract = data_contract(train_ds)
    start_epoch, global_step, history = 0, 0, []
    best_rmse, significant_best, best_epoch, bad_epochs = math.inf, math.inf, -1, 0
    initialization = {"mode": "scratch"}
    if resume_state:
        validate_data_contract(resume_state["data_contract"], train_ds, require_manifest=True)
        start_epoch = resume_state["epoch"] + 1
        if t["epochs"] <= start_epoch:
            raise ValueError(f"epochs is the total target and must be > completed {start_epoch}")
        best_rmse, best_epoch = resume_state["best_val_rmse"], resume_state["best_epoch"]
        significant_best, bad_epochs = resume_state["significant_best"], resume_state["bad_epochs"]
        global_step, history = resume_state["global_step"], resume_state["history"]
        initialization = {"mode": "resume", "source": str(Path(resume).resolve()), "sha256": sha256_file(resume)}
        if resume_state["epoch"] == best_epoch:
            source_best = resume_state
        else:
            source_path = Path(resume).resolve().parent / "best.pt"
            best_model, source_best = load_checkpoint(source_path, "cpu")
            del best_model
            from .runtime import same_data_contract
            if source_best["epoch"] != best_epoch or source_best["best_val_rmse"] != best_rmse or not same_data_contract(source_best["data_contract"], resume_state["data_contract"]):
                raise ValueError("Sibling best.pt does not match resumed checkpoint")
    else:
        model = AutoencoderKL(c["model"])
        if pretrained:
            initialization = {"mode": "pretrained", **load_pretrained(model, pretrained)}
    model.to(device)
    c["model"] = model.spec
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and t["precision"] == "float16")
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scaler.load_state_dict(resume_state["scaler_state"])
        restore_rng(resume_state["rng_state"])
    # All preflight checks precede directory creation, including shape compatibility.
    nz, ny = train_ds.image_shape[-2:]
    if nz % model.downsample_factor or ny % model.downsample_factor:
        raise ValueError("Dataset dimensions must be divisible by the model downsample factor")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_config.json", {"config": c, "data_contract": contract, "initialization": initialization,
        "dataset_root": str(train_ds.root), "device": str(device), "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "parameter_count": sum(p.numel() for p in model.parameters()),
        "posterior_shape": [4 if "latent_channels" not in model.spec else model.spec["latent_channels"], nz // model.downsample_factor, ny // model.downsample_factor],
        "selection_metric": "minimum validation RMSE in log10rho; deterministic posterior mode"})
    if source_best is not None:
        save_checkpoint(output / "best.pt", source_best)
    amp_dtype = torch.float16 if t["precision"] == "float16" else torch.bfloat16
    stopped_early = False
    try:
        for epoch in range(start_epoch, t["epochs"]):
            before = time.perf_counter()
            model.train()
            sums = {"loss": 0.0, "pixel": 0.0, "gradient": 0.0, "kl_per_latent": 0.0}
            sample_count = 0
            skipped_updates = 0
            beta = kl_coefficient(t, epoch)
            for batch in _loader(train_ds, t, train=True, epoch=epoch):
                target = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast("cuda", dtype=amp_dtype) if t["precision"] != "float32" else nullcontext()
                with context:
                    pred, posterior = model(target, sample_posterior=True)
                    pixel, gradient, kl = reconstruction_terms(pred, target, posterior, t["reconstruction"])
                    loss = pixel + t["gradient_weight"] * gradient + beta * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, step {global_step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # FP16 overflow is handled by GradScaler's skipped step and
                # reduced scale. FP32/BF16 nonfinite gradients remain errors.
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"], error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped_updates += int(scaler.get_scale() < previous_scale)
                count = target.shape[0]
                for key, value in zip(sums, (loss, pixel, gradient, kl)):
                    sums[key] += float(value.detach()) * count
                sample_count += count
                global_step += 1
            metrics = validate_epoch(model, _loader(val_ds, t, train=False), device, train_ds.log10_bounds)
            score = metrics["val_rmse_log10"]
            is_best = score < best_rmse
            if is_best:
                best_rmse, best_epoch = score, epoch
            if score < significant_best - t["min_delta"]:
                significant_best, bad_epochs = score, 0
            else:
                bad_epochs += 1
            row = {"epoch": epoch + 1, "global_step": global_step,
                   **{f"train_{k}": v / sample_count for k, v in sums.items()},
                   "kl_coefficient": beta, "skipped_optimizer_updates": skipped_updates,
                   **metrics, "best_val_rmse": best_rmse,
                   "seconds": time.perf_counter() - before}
            history.append(row)
            state = {"format": "lwd_vae_training_v1", "model_spec": model.spec,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(), "training_config": t, "data_contract": contract,
                "epoch": epoch, "global_step": global_step, "best_val_rmse": best_rmse,
                "best_epoch": best_epoch, "significant_best": significant_best, "bad_epochs": bad_epochs,
                "rng_state": capture_rng(), "history": history, "initialization": initialization}
            if is_best:
                save_checkpoint(output / "best.pt", state)
            save_checkpoint(output / "last.pt", state)
            _history_csv(output / "history.csv", history)
            print(f"epoch {epoch+1}/{t['epochs']} loss={row['train_loss']:.6g} val_RMSE(log10rho)={score:.6g} best={best_rmse:.6g}", flush=True)
            if t["patience"] and bad_epochs >= t["patience"]:
                stopped_early = True
                break
        summary = {"completed_epochs": history[-1]["epoch"], "best_epoch": best_epoch + 1,
            "best_val_rmse_log10": best_rmse, "stopped_early": stopped_early,
            "best_checkpoint": "best.pt", "last_checkpoint": "last.pt", "test_splits_used": False}
        write_json(output / "summary.json", summary)
        return summary
    finally:
        train_ds.close()
        val_ds.close()
