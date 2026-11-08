"""LWD 参考方案微调主循环。

阅读顺序：train() 的输入检查 → 网络/损失初始化 → 一个 epoch 的更新
→ 后验均值验证 → best/last 保存。测试与 OOD 数据不参与训练或选模型。
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import csv
import math
from pathlib import Path
import platform
import time

import torch
from torch.utils.data import DataLoader, Subset

from ..dataset import LWDVAEDataset
from ..runtime import (capture_rng, data_contract, load_checkpoint, restore_rng,
                       save_checkpoint, seed_everything, select_device, sha256_file, same_data_contract,
                       validate_data_contract, write_json)
from ..vendor.compvis.ldm.modules.ema import LitEma
from ..weights import load_pretrained
from .config import missing_inputs, project_path, resolve_paths, validate_config
from .losses import VAELoss, loss_coefficients
from .model import CompVisKLF8, upstream_provenance


CHECKPOINT_FORMAT = "lwd_vae_finetuning_v2"


def make_loader(dataset, settings, *, training, epoch=0):
    maximum = settings["max_train_samples" if training else "max_val_samples"]
    count = len(dataset) if maximum is None else min(maximum, len(dataset))
    if count == 0:
        raise ValueError("Train and validation splits must be nonempty")
    seed = settings["seed"] + epoch * 2 + int(training)
    return DataLoader(Subset(dataset, range(count)), batch_size=settings["batch_size"],
                      shuffle=training, num_workers=settings["workers"], drop_last=False,
                      generator=torch.Generator().manual_seed(seed))


def make_optimizer(model, settings):
    common = {"lr": settings["learning_rate"], "weight_decay": settings["weight_decay"]}
    if settings["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=settings["momentum"], **common)
    return torch.optim.AdamW(model.parameters(), **common)


def validate(model, loader, device, bounds):
    """在原始 log10ρ 单位统计误差；用后验均值，不裁切预测。"""
    model.eval()
    count, squared, absolute, outside = 0, 0.0, 0.0, 0
    scale = (bounds[1] - bounds[0]) / 2
    with torch.no_grad():
        for target in loader:
            target = target.to(device)
            prediction = model.decode(model.encode(target).mode())
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite validation prediction")
            difference = (prediction.cpu().double() - target.cpu().double()) * scale
            squared += difference.square().sum().item()
            absolute += difference.abs().sum().item()
            count += difference.numel()
            outside += ((prediction < -1) | (prediction > 1)).sum().item()
    return {"val_rmse_log10": math.sqrt(squared / count),
            "val_mae_log10": absolute / count, "val_out_of_bounds_fraction": outside / count}


def _write_history(path, rows):
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(config=None, *, device_name="auto", resume=None, output=None, dataset=None, epochs=None):
    """新训练传 config；续训仅传 resume，允许更换数据路径/输出路径和总轮数。

    续训检查数据指纹，并从 checkpoint 恢复 VAE、损失网络、优化器、EMA
    和随机状态，因此不需要再次提供最初的外部权重文件。
    """
    if resume and config is not None:
        raise ValueError("Resume restores experiment_config; do not also supply a config")
    saved, inherited_best = None, None
    if resume:
        if output is None:
            raise ValueError("Resume requires a new --output directory")
        model, saved = load_checkpoint(project_path(resume), device="cpu")
        if saved["format"] != CHECKPOINT_FORMAT:
            raise ValueError("Use train_baseline.py to resume legacy baseline checkpoints")
        c = deepcopy(saved["experiment_config"])
    elif config is not None:
        c = deepcopy(config)
    else:
        raise ValueError("A configuration is required for a new run")
    if output is not None:
        c["paths"]["output"] = str(output)
    if dataset is not None:
        c["paths"]["dataset"] = str(dataset)
    if epochs is not None:
        c["training"]["epochs"] = epochs
    c = validate_config(c)
    paths, settings = resolve_paths(c), c["training"]
    output_path = paths["output"]
    if output_path.exists():
        raise FileExistsError(f"Choose a new output directory; refusing overwrite: {output_path}")
    missing = missing_inputs(c, resumed=bool(resume))
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))
    device = select_device(device_name)
    if settings["precision"] != "float32" and device.type != "cuda":
        raise ValueError("float16/bfloat16 training requires CUDA; use float32 for CPU/MPS")
    if settings["precision"] == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Selected CUDA device does not support bfloat16")
    torch.set_num_threads(settings["cpu_threads"])
    seed_everything(settings["seed"])

    # 1. 只打开 train/val，检查形状、数值、物理网格和归一化约定。
    with LWDVAEDataset(paths["dataset"], "train", as_torch=True) as train_ds, \
         LWDVAEDataset(paths["dataset"], "val", as_torch=True) as val_ds:
        train_ds.validate_all()
        val_ds.validate_all()
        if not len(train_ds) or not len(val_ds):
            raise ValueError("Train/val must be nonempty")
        contract = data_contract(train_ds)
        nz, ny = train_ds.image_shape[-2:]
        if nz % 8 or ny % 8:
            raise ValueError("LWD image height and width must be divisible by 8")
        if c["loss"]["feature"] != "none" and min(nz, ny) < 32:
            raise ValueError("AlexNet feature losses need both image dimensions >= 32; no implicit resize")

        # 2. 建立原版网络、冻结特征损失和优化器；预训练缺失时不退回随机初始化。
        initialization = {"mode": "scratch", "rgb_to_gray": c["initialization"]["rgb_to_gray"]}
        if not saved:
            model = CompVisKLF8(c["model"])
            if c["initialization"]["mode"] == "pretrained":
                initialization = {"mode": "pretrained", **load_pretrained(
                    model, paths["vae_weights"], rgb_to_gray=c["initialization"]["rgb_to_gray"])}
        else:
            validate_data_contract(saved["data_contract"], train_ds, require_manifest=True)
            if saved["model_spec"] != c["model"]:
                raise ValueError("Checkpoint model_spec and experiment_config disagree")
            initialization = {"mode": "resume", "checkpoint": str(project_path(resume)),
                              "sha256": sha256_file(project_path(resume)),
                              "original": saved["initialization"]}
        model.to(device)
        criterion = VAELoss(c["loss"], paths["alexnet_weights"],
                            restored_state=saved["loss_state"] if saved else None).to(device)
        optimizer = make_optimizer(model, c["optimizer"])
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and settings["precision"] == "float16")
        ema = LitEma(model, decay=c["ema"]["decay"]).to(device) if c["ema"]["decay"] else None
        feature_provenance = saved["feature_provenance"] if saved else {
            "kind": c["loss"]["feature"],
            "alexnet_path": str(paths["alexnet_weights"]) if c["loss"]["feature"] != "none" else None,
            "alexnet_sha256": sha256_file(paths["alexnet_weights"]) if c["loss"]["feature"] != "none" else None,
        }
        if c["loss"]["feature"] == "lpips_alex":
            feature_provenance["lpips_calibration_sha256"] = criterion.feature_network.calibration_sha256
        start, global_step, history = 0, 0, []
        best, significant_best, best_epoch, bad_epochs = math.inf, math.inf, -1, 0
        if saved:
            start = saved["epoch"] + 1
            if settings["epochs"] <= start:
                raise ValueError("--epochs is the total target and must exceed completed epochs")
            optimizer.load_state_dict(saved["optimizer_state"])
            scaler.load_state_dict(saved["scaler_state"])
            if ema is not None:
                ema.load_state_dict(saved["ema_state"], strict=True)
            elif saved["ema_state"] is not None:
                raise ValueError("Checkpoint EMA settings disagree")
            global_step, history = saved["global_step"], deepcopy(saved["history"])
            best, best_epoch = saved["best_val_rmse"], saved["best_epoch"]
            significant_best, bad_epochs = saved["significant_best"], saved["bad_epochs"]
            if saved["epoch"] == best_epoch:
                inherited_best = saved
            else:
                best_path = project_path(resume).parent / "best.pt"
                _, inherited_best = load_checkpoint(best_path)
                if (inherited_best["epoch"] != best_epoch or inherited_best["best_val_rmse"] != best
                        or not same_data_contract(inherited_best["data_contract"], saved["data_contract"])):
                    raise ValueError("Sibling best.pt does not match the resumed run")
        else:
            # 在建立 run 目录前检查一次真实 LWD 输入，及时发现网络/特征维度错误。
            model.eval()
            with torch.no_grad():
                sample = train_ds[0].unsqueeze(0).to(device)
                prediction, posterior = model(sample, sample_posterior=False)
                if prediction.shape != sample.shape or not torch.isfinite(criterion(prediction, sample, posterior, 0)["loss"]):
                    raise ValueError("Model/loss preflight failed on the LWD grid")
        if saved:
            # 初始化网络和损失会消耗 RNG；在所有构造完成后再恢复。
            restore_rng(saved["rng_state"])

        # 3. 所有输入检查通过，才发布此次实验的目录。
        checkpoint_dir, log_dir = output_path / "checkpoints", output_path / "logs"
        output_path.mkdir(parents=True, exist_ok=False)
        checkpoint_dir.mkdir()
        log_dir.mkdir()
        write_json(output_path / "run_config.json", {
            "config": c, "resolved_paths": {k: str(v) if v else None for k, v in paths.items()},
            "initialization": initialization, "feature_provenance": feature_provenance,
            "data_contract": contract, "upstream": upstream_provenance(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "posterior_shape": [c["model"]["latent_channels"], nz // 8, ny // 8],
            "device": str(device), "torch_version": str(torch.__version__), "python_version": platform.python_version(),
            "selection_metric": "minimum validation RMSE in raw log10rho, posterior mode, after optional epoch EMA copyback",
        })
        if inherited_best:
            save_checkpoint(checkpoint_dir / "best.pt", inherited_best)
        stopped_early = False
        amp_dtype = torch.float16 if settings["precision"] == "float16" else torch.bfloat16

        # 4. 每轮：后验采样训练 → 可选 EMA → 后验均值验证 → 保存。
        for epoch in range(start, settings["epochs"]):
            before = time.perf_counter()
            model.train()
            sums, count, skipped = {}, 0, 0
            for target in make_loader(train_ds, settings, training=True, epoch=epoch):
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                amp = torch.autocast("cuda", dtype=amp_dtype) if settings["precision"] != "float32" else nullcontext()
                with amp:
                    prediction, posterior = model(target, sample_posterior=True)
                    terms = criterion(prediction, target, posterior, epoch)
                if not all(torch.isfinite(value).all() for value in terms.values()):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}, step {global_step}")
                scaler.scale(terms["loss"]).backward()
                scaler.unscale_(optimizer)
                # grad_clip_norm=null 表示不裁剪，但仍检查非 AMP 梯度是否有限。
                clip = settings["grad_clip_norm"] or math.inf
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < previous_scale)
                for name, value in terms.items():
                    sums[name] = sums.get(name, 0.0) + value.detach().item() * len(target)
                count += len(target)
                global_step += 1
            if ema is not None:
                ema(model)
                ema.copy_to(model)
            metrics = validate(model, make_loader(val_ds, settings, training=False), device, train_ds.log10_bounds)
            score = metrics["val_rmse_log10"]
            is_best = score < best
            if is_best:
                best, best_epoch = score, epoch
            if score < significant_best - settings["min_delta"]:
                significant_best, bad_epochs = score, 0
            else:
                bad_epochs += 1
            feature_weight, kl_weight = loss_coefficients(c["loss"], epoch)
            row = {"epoch": epoch + 1, "global_step": global_step,
                   **{f"train_{key}": value / count for key, value in sums.items()},
                   "feature_coefficient": feature_weight, "kl_coefficient": kl_weight,
                   "learning_rate": optimizer.param_groups[0]["lr"], "skipped_optimizer_updates": skipped,
                   **metrics, "best_val_rmse": best, "seconds": time.perf_counter() - before}
            history.append(row)
            state = {
                "format": CHECKPOINT_FORMAT, "model_backend": "compvis_original_core",
                "upstream_commit": upstream_provenance()["commit"], "model_spec": model.spec,
                "model_state": model.state_dict(), "loss_state": criterion.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                "ema_state": ema.state_dict() if ema is not None else None,
                "training_config": settings, "experiment_config": c, "data_contract": contract,
                "epoch": epoch, "global_step": global_step, "history": history,
                "best_val_rmse": best, "best_epoch": best_epoch,
                "significant_best": significant_best, "bad_epochs": bad_epochs,
                "rng_state": capture_rng(), "initialization": initialization,
                "feature_provenance": feature_provenance,
            }
            if is_best:
                save_checkpoint(checkpoint_dir / "best.pt", state)
            save_checkpoint(checkpoint_dir / "last.pt", state)
            _write_history(log_dir / "history.csv", history)
            print(f"epoch {epoch+1}/{settings['epochs']} loss={row['train_loss']:.6g} val_RMSE(log10rho)={score:.6g} best={best:.6g}", flush=True)
            if settings["patience"] and bad_epochs >= settings["patience"]:
                stopped_early = True
                break
        summary = {"completed_epochs": history[-1]["epoch"], "best_epoch": best_epoch + 1,
                   "best_val_rmse_log10": best, "stopped_early": stopped_early,
                   "best_checkpoint": "checkpoints/best.pt", "last_checkpoint": "checkpoints/last.pt",
                   "test_splits_used": False, "reference": c["reference"]}
        write_json(output_path / "summary.json", summary)
        return summary
