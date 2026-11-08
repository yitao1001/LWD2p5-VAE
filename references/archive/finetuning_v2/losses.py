"""将重构、特征与 KL 三项明确组合；损失系数和数值归约均写入日志。"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .features import build_feature_loss


def loss_coefficients(settings: dict, epoch: int) -> tuple[float, float]:
    """epoch 为从 0 开始的轮次；配置中的 decay_after_epoch 从 1 开始。"""
    warmup = settings["kl_warmup_epochs"]
    kl_weight = settings["kl_weight"] * (min(1, (epoch + 1) / warmup) if warmup else 1)
    feature_weight = settings["feature_weight"]
    decay_after = settings["feature_decay_after_epoch"]
    if decay_after is not None and epoch + 1 > decay_after:
        feature_weight *= settings["feature_decay_factor"]
    return float(feature_weight), float(kl_weight)


class VAELoss(nn.Module):
    def __init__(self, settings: dict, alexnet_weights: Path | None, *, restored_state=None):
        super().__init__()
        self.settings = dict(settings)
        self.feature_network = build_feature_loss(settings, alexnet_weights, restoring=restored_state is not None)
        if restored_state is not None:
            self.load_state_dict(restored_state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, prediction, target, posterior, epoch: int):
        # 卷积可用 AMP；KL 的指数和图像特征损失统一用 FP32，避免半精度溢出。
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            pred, true = prediction.float(), target.float()
            pixel = F.mse_loss(pred, true) if self.settings["pixel"] == "mse" else F.l1_loss(pred, true)
            feature = pred.new_zeros(()) if self.feature_network is None else self.feature_network(pred, true)
            mean, logvar = posterior.mean.float(), posterior.logvar.float()
            element_kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar)
            kl_per_latent = element_kl.mean()
            kl = element_kl.flatten(1).sum(1).mean() if self.settings["kl_reduction"] == "sum_per_sample" else kl_per_latent
            feature_weight, kl_weight = loss_coefficients(self.settings, epoch)
            total = pixel + feature_weight * feature + kl_weight * kl
        return {"loss": total, "pixel": pixel, "feature": feature, "kl": kl,
                "kl_per_latent": kl_per_latent,
                "weighted_feature": feature_weight * feature, "weighted_kl": kl_weight * kl}
