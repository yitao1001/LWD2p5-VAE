"""原版 CompVis 编码器/解码器的薄适配层。

网络层来自 vendor/compvis 中固定版本的原始文件；这里仅组装 KL VAE，
增加 LWD 张量检查。encode/decode 对应 references/compvis/autoencoder_original.py，
不使用旧的 AI 重写网络，也不依赖旧版 Lightning 的训练生命周期。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ..vendor.compvis.ldm.modules.diffusionmodules.model import Encoder, Decoder
from ..vendor.compvis.ldm.modules.distributions.distributions import DiagonalGaussianDistribution


def upstream_provenance() -> dict:
    path = Path(__file__).resolve().parents[1] / "vendor/compvis/provenance.json"
    return json.loads(path.read_text())


class CompVisKLF8(nn.Module):
    """输入 [B,1,H,W]；潜变量 [B,4,H/8,W/8]，支持矩形网格。"""
    downsample_factor = 8

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        ddconfig = {
            "double_z": True, "z_channels": spec["latent_channels"],
            "resolution": 128, "in_channels": 1, "out_ch": 1,
            "ch": spec["base_channels"], "ch_mult": tuple(spec["channel_mult"]),
            "num_res_blocks": spec["num_res_blocks"], "attn_resolutions": [],
            "dropout": spec["dropout"],
        }
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        channels = spec["latent_channels"]
        # 与原 AutoencoderKL 相同的 1x1 posterior/latent 映射及参数命名。
        self.quant_conv = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.post_quant_conv = nn.Conv2d(channels, channels, 1)

    def encode(self, image):
        if image.ndim != 4 or image.shape[1] != 1 or any(n < 8 or n % 8 for n in image.shape[-2:]):
            raise ValueError("Expected [B,1,H,W], with H/W positive multiples of 8; no resize")
        moments = self.quant_conv(self.encoder(image))
        # exp(logvar) 在 FP16 中容易溢出；仅后验统计用 FP32，卷积仍支持 AMP。
        if moments.dtype in (torch.float16, torch.bfloat16):
            moments = moments.float()
        return DiagonalGaussianDistribution(moments)

    def decode(self, latent):
        if latent.ndim != 4 or latent.shape[1] != self.spec["latent_channels"] or min(latent.shape[-2:]) < 1:
            raise ValueError("Invalid spatial latent tensor")
        return self.decoder(self.post_quant_conv(latent))

    def forward(self, image, sample_posterior=True):
        posterior = self.encode(image)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(latent), posterior
