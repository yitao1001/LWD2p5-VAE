"""A grayscale, rectangular-image KL autoencoder with CompVis-compatible keys.

The encoder/decoder topology follows CompVis latent-diffusion's MIT-licensed
AutoencoderKL (see THIRD_PARTY_NOTICES.md). There is no Lightning, ``ldm``,
perceptual-loss, diffusion-scaling, or fixed square-image dependency.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_MODEL_SPEC = {
    "base_channels": 32,
    "channel_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "latent_channels": 4,
    "dropout": 0.0,
}


def canonical_model_spec(spec: Mapping | None = None) -> dict:
    """Validate and materialize the five architecture settings saved in checkpoints."""
    supplied = {} if spec is None else dict(spec)
    unknown = set(supplied) - set(DEFAULT_MODEL_SPEC)
    if unknown:
        raise ValueError(f"Unknown model settings: {sorted(unknown)}")
    result = {**DEFAULT_MODEL_SPEC, **supplied}
    for name in ("base_channels", "num_res_blocks", "latent_channels"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    multipliers = result["channel_mult"]
    if not isinstance(multipliers, (list, tuple)) or len(multipliers) != 4:
        raise ValueError("channel_mult must contain four entries for downsampling factor 8")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in multipliers):
        raise ValueError("channel_mult entries must be positive integers")
    result["channel_mult"] = list(multipliers)
    dropout = result["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("dropout must be a finite number in [0, 1)")
    result["dropout"] = float(dropout)
    return result


def _norm(channels: int) -> nn.GroupNorm:
    # The pretrained 128-channel architecture always uses the original 32 groups.
    # gcd also permits an intentionally small model for CPU smoke checks.
    return nn.GroupNorm(math.gcd(32, channels), channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.norm1 = _norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        shortcut = self.nin_shortcut(x) if hasattr(self, "nin_shortcut") else x
        return shortcut + h


class AttnBlock(nn.Module):
    """Single-head spatial attention, present only at the encoder/decoder middle."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        batch, channels, height, width = h.shape
        q = self.q(h).reshape(batch, channels, height * width).transpose(1, 2)
        k = self.k(h).reshape(batch, channels, height * width)
        weights = F.softmax(torch.bmm(q, k) * channels ** -0.5, dim=2)
        v = self.v(h).reshape(batch, channels, height * width)
        h = torch.bmm(v, weights.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # CompVis uses right/bottom padding before its strided convolution.
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def _middle(channels: int, dropout: float) -> nn.Module:
    mid = nn.Module()
    mid.block_1 = ResnetBlock(channels, channels, dropout)
    mid.attn_1 = AttnBlock(channels)
    mid.block_2 = ResnetBlock(channels, channels, dropout)
    return mid


def _run_middle(mid: nn.Module, h: Tensor) -> Tensor:
    return mid.block_2(mid.attn_1(mid.block_1(h)))


class Encoder(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        base = spec["base_channels"]
        self.conv_in = nn.Conv2d(1, base, 3, padding=1)
        self.down = nn.ModuleList()
        channels = base
        for level, multiplier in enumerate(spec["channel_mult"]):
            down = nn.Module()
            down.block = nn.ModuleList()
            down.attn = nn.ModuleList()
            for _ in range(spec["num_res_blocks"]):
                target = base * multiplier
                down.block.append(ResnetBlock(channels, target, spec["dropout"]))
                channels = target
            if level < len(spec["channel_mult"]) - 1:
                down.downsample = Downsample(channels)
            self.down.append(down)
        self.mid = _middle(channels, spec["dropout"])
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv2d(channels, 2 * spec["latent_channels"], 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                h = block(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        h = _run_middle(self.mid, h)
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        base = spec["base_channels"]
        channels = base * spec["channel_mult"][-1]
        self.conv_in = nn.Conv2d(spec["latent_channels"], channels, 3, padding=1)
        self.mid = _middle(channels, spec["dropout"])
        self.up = nn.ModuleList()
        for level in reversed(range(len(spec["channel_mult"]))):
            up = nn.Module()
            up.block = nn.ModuleList()
            up.attn = nn.ModuleList()
            target = base * spec["channel_mult"][level]
            for _ in range(spec["num_res_blocks"] + 1):
                up.block.append(ResnetBlock(channels, target, spec["dropout"]))
                channels = target
            if level != 0:
                up.upsample = Upsample(channels)
            self.up.insert(0, up)  # Match CompVis decoder.up.* checkpoint numbering.
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = _run_middle(self.mid, self.conv_in(z))
        for up in reversed(self.up):
            for block in up.block:
                h = block(h)
            if hasattr(up, "upsample"):
                h = up.upsample(h)
        # Intentionally unbounded, as in the original KL-f8 AutoencoderKL.
        return self.conv_out(F.silu(self.norm_out(h)))


class DiagonalGaussian:
    """Reparameterized diagonal posterior; KL is summed per image, not averaged."""
    def __init__(self, parameters: Tensor):
        if parameters.ndim != 4 or parameters.shape[1] % 2:
            raise ValueError("Posterior parameters must have shape [B, 2*C, H, W]")
        # Autocast may produce FP16 moments, but exp(+20) exceeds the FP16
        # range. Keep posterior statistics and reparameterization in FP32;
        # decoder convolutions can still use autocast. Preserve float64 inputs.
        if parameters.dtype in (torch.float16, torch.bfloat16):
            parameters = parameters.float()
        self.mean, logvar = parameters.chunk(2, dim=1)
        self.logvar = logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(self.mean.shape, device=self.mean.device,
                            dtype=self.mean.dtype, generator=generator)
        return self.mean + self.std * noise

    def mode(self) -> Tensor:
        return self.mean

    def kl(self) -> Tensor:
        return 0.5 * (self.mean.square() + self.var - 1.0 - self.logvar).sum(dim=(1, 2, 3))


class AutoencoderKL(nn.Module):
    """One-channel KL-f8 VAE; 48x120 inputs produce 4x6x15 default latents."""
    downsample_factor = 8

    def __init__(self, spec: Mapping | None = None):
        super().__init__()
        self.spec = canonical_model_spec(spec)
        self.encoder = Encoder(self.spec)
        self.decoder = Decoder(self.spec)
        latent_channels = self.spec["latent_channels"]
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def encode(self, x: Tensor) -> DiagonalGaussian:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Input must have shape [B, 1, H, W]")
        if any(size < 8 or size % 8 for size in x.shape[-2:]):
            raise ValueError("Input height and width must be positive multiples of 8; no resizing is applied")
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))

    def decode(self, z: Tensor) -> Tensor:
        if z.ndim != 4 or z.shape[1] != self.spec["latent_channels"]:
            raise ValueError(f"Latent must have shape [B, {self.spec['latent_channels']}, H, W]")
        if min(z.shape[-2:]) < 1:
            raise ValueError("Latent height and width must be positive")
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: Tensor, sample_posterior: bool = True) -> tuple[Tensor, DiagonalGaussian]:
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior


# Historical import path retained for baseline scripts and checkpoints.
from ..weights import load_pretrained as _load_pretrained

def load_pretrained(model, path):
    return _load_pretrained(model, path, rgb_to_gray="sum")
