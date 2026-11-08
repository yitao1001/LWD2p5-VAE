"""读取实验配置，并统一把相对路径解释为相对于项目根目录。

只依赖标准库，因此尚未安装 PyTorch 时也可以查看配置。配置中不允许
未知字段，避免参数拼错后被静默忽略。
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/finetuning/csem_reference.json"


def project_path(value: str | Path | None) -> Path | None:
    """绝对路径保持不变；相对路径始终以项目目录为基准，与启动位置无关。"""
    if value is None:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(expected))}")


def _number(value, label, *, minimum=0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"{label} must be {'>' if positive else '>='} {minimum}")


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def validate_config(config: dict) -> dict:
    c = deepcopy(config)
    _keys(c, ("schema_version", "reference", "paths", "model", "initialization", "loss", "optimizer", "training", "ema"), "config")
    if type(c["schema_version"]) is not int or c["schema_version"] != 2:
        raise ValueError("Expected finetuning config schema_version=2")
    _keys(c["reference"], ("name", "basis", "status", "assumptions"), "reference")
    for key in ("name", "basis", "status"):
        if not isinstance(c["reference"][key], str) or not c["reference"][key].strip():
            raise ValueError(f"reference.{key} must be a nonempty string")
    if not isinstance(c["reference"]["assumptions"], list) or not all(isinstance(x, str) for x in c["reference"]["assumptions"]):
        raise ValueError("reference.assumptions must be a list of strings")
    _keys(c["paths"], ("dataset", "output", "vae_weights", "alexnet_weights"), "paths")
    for key, value in c["paths"].items():
        if value is None and key.endswith("weights"):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a nonempty path")
    m = c["model"]
    _keys(m, ("base_channels", "channel_mult", "num_res_blocks", "latent_channels", "dropout"), "model")
    _integer(m["base_channels"], "model.base_channels", 32)
    if m["base_channels"] % 32:
        raise ValueError("Original CompVis GroupNorm requires base_channels to be divisible by 32")
    if not isinstance(m["channel_mult"], list) or len(m["channel_mult"]) != 4:
        raise ValueError("channel_mult must have four levels (8-fold downsampling)")
    for multiplier in m["channel_mult"]:
        _integer(multiplier, "channel multiplier", 1)
    for key in ("num_res_blocks", "latent_channels"):
        _integer(m[key], f"model.{key}", 1)
    _number(m["dropout"], "model.dropout")
    if m["dropout"] >= 1:
        raise ValueError("dropout must be < 1")
    init = c["initialization"]
    _keys(init, ("mode", "rgb_to_gray"), "initialization")
    if init["mode"] not in ("pretrained", "scratch") or init["rgb_to_gray"] not in ("mean", "sum"):
        raise ValueError("initialization: mode=pretrained/scratch; rgb_to_gray=mean/sum")
    if init["mode"] == "pretrained" and not c["paths"]["vae_weights"]:
        raise ValueError("pretrained mode requires paths.vae_weights; no random-init fallback")
    if init["mode"] == "scratch" and c["paths"]["vae_weights"] is not None:
        raise ValueError("scratch mode requires vae_weights=null, so a checkpoint is never silently ignored")
    loss = c["loss"]
    _keys(loss, ("pixel", "feature", "feature_weight", "feature_layers", "kl_weight", "kl_reduction", "kl_warmup_epochs", "feature_decay_after_epoch", "feature_decay_factor"), "loss")
    if loss["pixel"] not in ("mse", "l1") or loss["feature"] not in ("alexnet_l2", "lpips_alex", "none"):
        raise ValueError("loss: pixel=mse/l1; feature=alexnet_l2/lpips_alex/none")
    if loss["kl_reduction"] not in ("sum_per_sample", "mean_per_element"):
        raise ValueError("kl_reduction must be sum_per_sample or mean_per_element")
    for key in ("feature_weight", "kl_weight", "feature_decay_factor"):
        _number(loss[key], f"loss.{key}")
    _integer(loss["kl_warmup_epochs"], "loss.kl_warmup_epochs")
    if loss["feature_decay_after_epoch"] is not None:
        _integer(loss["feature_decay_after_epoch"], "loss.feature_decay_after_epoch", 1)
    layers = loss["feature_layers"]
    if not isinstance(layers, list) or not layers or any(type(x) is not int or x not in (1, 4, 7, 9, 11) for x in layers) or sorted(set(layers)) != layers:
        raise ValueError("feature_layers must be an increasing subset of AlexNet ReLU indices [1,4,7,9,11]")
    if loss["feature"] == "none" and loss["feature_weight"] != 0:
        raise ValueError("feature=none requires feature_weight=0")
    if loss["feature"] != "none" and (not c["paths"]["alexnet_weights"] or loss["feature_weight"] <= 0):
        raise ValueError("Feature loss requires local alexnet_weights and a positive feature_weight")
    if loss["feature"] == "lpips_alex" and layers != [1, 4, 7, 9, 11]:
        raise ValueError("LPIPS uses all five official AlexNet slices; feature_layers cannot override them")
    optimizer = c["optimizer"]
    _keys(optimizer, ("name", "learning_rate", "momentum", "weight_decay"), "optimizer")
    if optimizer["name"] not in ("sgd", "adamw"):
        raise ValueError("optimizer.name must be sgd or adamw")
    _number(optimizer["learning_rate"], "optimizer.learning_rate", positive=True)
    _number(optimizer["weight_decay"], "optimizer.weight_decay")
    _number(optimizer["momentum"], "optimizer.momentum")
    if optimizer["momentum"] >= 1:
        raise ValueError("momentum must be < 1")
    if optimizer["name"] == "adamw" and optimizer["momentum"] != 0:
        raise ValueError("AdamW uses its standard betas; set the unused momentum field to 0")
    t = c["training"]
    _keys(t, ("seed", "epochs", "batch_size", "workers", "cpu_threads", "precision", "grad_clip_norm", "patience", "min_delta", "max_train_samples", "max_val_samples"), "training")
    for key in ("seed", "epochs", "batch_size", "workers", "cpu_threads", "patience"):
        _integer(t[key], f"training.{key}", 1 if key in ("epochs", "batch_size", "cpu_threads") else 0)
    for key in ("max_train_samples", "max_val_samples"):
        if t[key] is not None:
            _integer(t[key], f"training.{key}", 1)
    if t["grad_clip_norm"] is not None:
        _number(t["grad_clip_norm"], "training.grad_clip_norm", positive=True)
    _number(t["min_delta"], "training.min_delta")
    if t["precision"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("precision must be float32, float16 or bfloat16")
    _keys(c["ema"], ("decay", "policy"), "ema")
    _number(c["ema"]["decay"], "ema.decay")
    if c["ema"]["decay"] >= 1 or c["ema"]["policy"] != "epoch_copyback":
        raise ValueError("EMA: 0 <= decay < 1; policy must be epoch_copyback (uploaded script convention)")
    return c


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return validate_config(json.loads(project_path(path).read_text(encoding="utf8"), object_pairs_hook=unique_keys))


def resolve_paths(config: dict) -> dict[str, Path | None]:
    return {key: project_path(value) for key, value in config["paths"].items()}


def missing_inputs(config: dict, *, resumed=False) -> list[str]:
    """一次列出全部缺失材料，避免逐个启动才发现下一项缺失。"""
    paths = resolve_paths(config)
    problems = []
    if not (paths["dataset"] / "manifest.json").is_file():
        problems.append(f"dataset manifest: {paths['dataset'] / 'manifest.json'}")
    if not resumed:
        for key, required in (("vae_weights", config["initialization"]["mode"] == "pretrained"),
                              ("alexnet_weights", config["loss"]["feature"] != "none")):
            if required and (paths[key] is None or not paths[key].is_file()):
                problems.append(f"{key}: {paths[key]}")
    return problems
