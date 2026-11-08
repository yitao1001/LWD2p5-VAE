"""Numerical and checkpoint regression checks for the LWD KL autoencoder."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from lwd_vae.model import AutoencoderKL, DiagonalGaussian, load_pretrained


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        return AutoencoderKL({"base_channels": 8, "num_res_blocks": 1})

    def test_rectangular_forward_backward_and_inverse_gradient(self):
        torch.manual_seed(17)
        model = self.model()
        x = torch.randn(2, 1, 48, 120)
        reconstructed, posterior = model(x)
        self.assertEqual(reconstructed.shape, x.shape)
        self.assertEqual(posterior.mean.shape, (2, 4, 6, 15))
        loss = F.mse_loss(reconstructed, x) + 1e-5 * posterior.kl().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        # The frozen decoder still exposes nonzero d(model)/d(latent), as needed
        # when the future LWD inversion updates z instead of decoder parameters.
        model.requires_grad_(False)
        z = torch.randn(1, 4, 6, 15, requires_grad=True)
        model.decode(z).square().mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0)

    def test_eval_posterior_mean_is_deterministic(self):
        model = self.model().eval()
        x = torch.randn(1, 1, 48, 120)
        with torch.no_grad():
            first, p = model(x, sample_posterior=False)
            second, _ = model(x, sample_posterior=False)
            direct = model.decode(p.mean)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, direct, rtol=0, atol=0)

    def test_gaussian_kl_is_analytic_sum(self):
        means = torch.full((2, 4, 6, 15), 2.0)
        logvar = torch.full_like(means, torch.log(torch.tensor(3.0)).item())
        posterior = DiagonalGaussian(torch.cat([means, logvar], dim=1))
        expected_per_element = 0.5 * (4.0 + 3.0 - 1.0 - torch.log(torch.tensor(3.0)))
        torch.testing.assert_close(posterior.kl(), torch.full((2,), expected_per_element * 360))
        standard = DiagonalGaussian(torch.zeros(2, 8, 6, 15))
        torch.testing.assert_close(standard.kl(), torch.zeros(2))
        g1, g2 = torch.Generator().manual_seed(8), torch.Generator().manual_seed(8)
        torch.testing.assert_close(posterior.sample(g1), posterior.sample(g2))

    def test_input_and_architecture_validation(self):
        model = self.model()
        for bad in (torch.zeros(1, 48, 120), torch.zeros(1, 3, 48, 120), torch.zeros(1, 1, 47, 120)):
            with self.assertRaises(ValueError):
                model.encode(bad)
        with self.assertRaises(ValueError):
            model.decode(torch.zeros(1, 3, 6, 15))
        for spec in ({"base_channels": 0}, {"dropout": float("nan")}, {"channel_mult": [1, 2]}, {"typo": 4}):
            with self.assertRaises(ValueError):
                AutoencoderKL(spec)

    def test_half_posterior_statistics_do_not_overflow(self):
        moments = torch.cat([torch.zeros(1, 4, 6, 15), torch.full((1, 4, 6, 15), 20.0)], dim=1).half()
        posterior = DiagonalGaussian(moments)
        self.assertEqual(posterior.mean.dtype, torch.float32)
        self.assertTrue(torch.isfinite(posterior.var).all())
        self.assertTrue(torch.isfinite(posterior.kl()).all())
        self.assertTrue(torch.isfinite(posterior.sample()).all())

    def test_compvis_parameter_names(self):
        keys = set(self.model().state_dict())
        for key in ("encoder.down.0.block.0.norm1.weight", "encoder.down.2.downsample.conv.weight",
                    "encoder.mid.attn_1.q.weight", "decoder.mid.block_2.conv2.weight",
                    "decoder.up.3.block.1.norm2.weight", "decoder.up.1.upsample.conv.weight",
                    "quant_conv.weight", "post_quant_conv.bias"):
            self.assertIn(key, keys)
        self.assertFalse(any("temb" in name for name in keys))

    def test_rgb_checkpoint_adaptation_and_complete_load(self):
        model = self.model()
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        in_key, out_key, bias_key = "encoder.conv_in.weight", "decoder.conv_out.weight", "decoder.conv_out.bias"
        state[in_key] = torch.randn(state[in_key].shape[0], 3, 3, 3)
        state[out_key] = torch.randn(3, state[out_key].shape[1], 3, 3)
        state[bias_key] = torch.randn(3)
        wrapped = {f"model.first_stage_model.{key}": value for key, value in state.items()}
        wrapped["loss.discriminator.weight"] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.ckpt"
            torch.save({"state_dict": wrapped}, path)
            report = load_pretrained(model, path)
            self.assertEqual(report["adapted_tensor_count"], 3)
            self.assertEqual(report["matched_tensor_count"], len(state))
            self.assertEqual(report["ignored_tensor_count"], 1)
            self.assertEqual(report["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        loaded = model.state_dict()
        torch.testing.assert_close(loaded[in_key], state[in_key].sum(1, keepdim=True))
        torch.testing.assert_close(loaded[out_key], state[out_key].mean(0, keepdim=True))
        torch.testing.assert_close(loaded[bias_key], state[bias_key].mean(0, keepdim=True))
        gray = torch.randn(1, 1, 12, 14)
        rgb_response = F.conv2d(gray.repeat(1, 3, 1, 1), state[in_key], state["encoder.conv_in.bias"], padding=1)
        torch.testing.assert_close(model.encoder.conv_in(gray), rgb_response, rtol=1e-5, atol=1e-5)

    def test_reject_partial_or_wrong_architecture_without_mutation(self):
        model = self.model()
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            incomplete = dict(original)
            incomplete.pop("quant_conv.bias")
            incomplete["encoder.conv_in.bias"] = original["encoder.conv_in.bias"] + 100
            torch.save(incomplete, path)
            with self.assertRaisesRegex(ValueError, "missing"):
                load_pretrained(model, path)
            bad_shape = dict(original)
            bad_shape["quant_conv.weight"] = torch.zeros(2, 2, 1, 1)
            torch.save(bad_shape, path)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_pretrained(model, path)
            unknown = dict(original)
            unknown["encoder.nonexistent.weight"] = torch.zeros(1)
            torch.save(unknown, path)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_pretrained(model, path)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
