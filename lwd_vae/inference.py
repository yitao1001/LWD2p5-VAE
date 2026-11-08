"""Differentiable raw-log10rho interface for a trained LWD VAE.

This wrapper does not call an EM solver or detach latent-space gradients.
Outputs are unbounded and never silently clipped to the training interval.
"""
import torch
from torch import nn

from .runtime import load_checkpoint


class LWDVAE(nn.Module):
    def __init__(self, checkpoint, device="cpu", freeze_weights=True):
        super().__init__()
        self.model, saved = load_checkpoint(checkpoint, device=device)
        self.data_contract = saved["data_contract"]
        grid = self.data_contract["grid"]
        self.image_shape = (1, grid["nz"], grid["ny"])
        self.latent_shape = (self.model.spec["latent_channels"], grid["nz"] // 8, grid["ny"] // 8)
        self.register_buffer("bounds", torch.tensor(self.data_contract["log10_bounds"], dtype=torch.float32, device=next(self.model.parameters()).device))
        if freeze_weights:
            self.model.requires_grad_(False)
        self.eval()

    def encode(self, log10rho):
        """Return the posterior from raw [B,1,nz,ny] log10rho tensors."""
        if log10rho.ndim != 4 or tuple(log10rho.shape[1:]) != self.image_shape:
            raise ValueError(f"Expected [B,{','.join(map(str,self.image_shape))}] on the checkpoint grid")
        if not torch.isfinite(log10rho).all():
            raise ValueError("log10rho contains nonfinite values")
        if bool((log10rho < self.bounds[0] - 1e-6).any() | (log10rho > self.bounds[1] + 1e-6).any()):
            raise ValueError("Input exceeds trained normalization bounds; no clipping is applied")
        normalized = 2 * (log10rho - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) - 1
        return self.model.encode(normalized)

    def decode(self, latent):
        """Return raw log10rho; gradient flows from cells to latent."""
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError(f"Expected latent [B,{','.join(map(str,self.latent_shape))}]")
        decoded = self.model.decode(latent)
        return self.bounds[0] + (decoded + 1) * (self.bounds[1] - self.bounds[0]) / 2

    def decode_inversion_vector(self, latent):
        """[B,nz*ny] vectors with z fastest, preserving autograd."""
        image = self.decode(latent)[:, 0]
        return image.transpose(-2, -1).reshape(image.shape[0], -1)

    def forward(self, log10rho):
        return self.decode(self.encode(log10rho).mode())
