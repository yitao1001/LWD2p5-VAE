"""将重构、特征与 KL 三项明确组合；损失系数和数值归约均写入日志。"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .features import build_feature_loss


def loss_coefficients(settings: dict, epoch: int = 0) -> tuple[float, float]:
    """本路线使用固定系数，不混入逆散射脚本的后期衰减或旧基线 KL warmup。"""
    return float(settings["feature_weight"]), float(settings["kl_weight"])


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
            pixel = F.mse_loss(pred, true)
            feature = pred.new_zeros(()) if self.feature_network is None else self.feature_network(pred, true)
            mean, logvar = posterior.mean.float(), posterior.logvar.float()
            element_kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar)
            kl_per_latent = element_kl.mean()
            kl = element_kl.flatten(1).sum(1).mean()
            feature_weight, kl_weight = loss_coefficients(self.settings, epoch)
            total = pixel + feature_weight * feature + kl_weight * kl
        return {"loss": total, "pixel": pixel, "feature": feature, "kl": kl,
                "kl_per_latent": kl_per_latent,
                "weighted_feature": feature_weight * feature, "weighted_kl": kl_weight * kl}
