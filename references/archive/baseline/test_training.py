"""CPU integration checks for training, exact continuation and held-out testing.

The existing smoke dataset is read-only. All runs and deliberately malformed
dataset copies live in a TemporaryDirectory and are removed after the tests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lwd_vae.dataset import LWDVAEDataset
from lwd_vae.inference import LWDVAE
from lwd_vae.model import DiagonalGaussian
from lwd_vae.runtime import load_checkpoint, validate_data_contract, same_data_contract
from lwd_vae.training import (
    kl_coefficient,
    load_training_config,
    reconstruction_terms,
    train,
    validate_training_config,
)
from test_vae import run as evaluate


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "smoke_v1"


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="lwd_vae_training_tests_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.config = load_training_config(ROOT / "configs" / "train_smoke.json")
        cls.config["training"].update(
            epochs=1, batch_size=2, max_train_samples=4, max_val_samples=2,
            cpu_threads=1, workers=0, precision="float32",
        )
        # Exercise restoration of both posterior sampling and dropout RNG state.
        cls.config["model"]["dropout"] = 0.1
        cls.one_epoch = cls.work / "one_epoch"
        cls.summary = train(DATASET, cls.one_epoch, cls.config, device_name="cpu")

    def output(self, name):
        return self.work / name

    def evaluation_args(self, output, dataset=DATASET):
        return argparse.Namespace(
            dataset=dataset, checkpoint=self.one_epoch / "best.pt", output=output,
            splits=["test"], device="cpu", batch_size=2, workers=0,
            max_samples=2, save_reconstructions=True, no_plots=True,
        )

    def copied_dataset(self, name):
        destination = self.output(name)
        shutil.copytree(DATASET, destination)
        return destination

    def test_reconstruction_loss_has_finite_gradients_and_mean_kl(self):
        prediction = torch.linspace(-1, 1, 24).reshape(2, 1, 3, 4).requires_grad_()
        target = torch.zeros_like(prediction)
        means = torch.full((2, 4, 2, 3), 2.0, requires_grad=True)
        logvar = torch.full_like(means, float(np.log(3)), requires_grad=True)
        posterior = DiagonalGaussian(torch.cat([means, logvar], dim=1))
        pixel, gradient, kl = reconstruction_terms(prediction, target, posterior)
        torch.testing.assert_close(pixel, prediction.square().mean())
        expected_kl = 0.5 * (4 + 3 - 1 - np.log(3))
        self.assertAlmostEqual(kl.item(), expected_kl, places=6)
        self.assertGreater(gradient.item(), 0)
        (pixel + 0.05 * gradient + 0.001 * kl).backward()
        for value in (prediction, means, logvar):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        l1, _, _ = reconstruction_terms(prediction, target, posterior, "l1")
        torch.testing.assert_close(l1, prediction.abs().mean())

    def test_warmup_and_training_checkpoint_contents(self):
        settings = self.config["training"]
        self.assertEqual(kl_coefficient(settings, 0), settings["kl_weight"] / 2)
        self.assertEqual(kl_coefficient(settings, 1), settings["kl_weight"])
        self.assertEqual(kl_coefficient({**settings, "kl_warmup_epochs": 0}, 0), settings["kl_weight"])
        self.assertEqual(self.summary["completed_epochs"], 1)
        self.assertFalse(self.summary["test_splits_used"])
        for name in ("best.pt", "last.pt"):
            model, checkpoint = load_checkpoint(self.one_epoch / name)
            self.assertEqual(checkpoint["format"], "lwd_vae_training_v1")
            self.assertEqual(checkpoint["epoch"], 0)
            self.assertEqual(checkpoint["global_step"], 2)
            self.assertEqual(len(checkpoint["history"]), 1)
            self.assertEqual(checkpoint["initialization"]["mode"], "scratch")
            self.assertEqual(set(checkpoint["data_contract"]["training_split_hashes"]), {"train", "val"})
            self.assertTrue(all(value.device.type == "cpu" for value in model.state_dict().values()))
            self.assertTrue(all(torch.isfinite(value).all() for value in model.state_dict().values()))
            self.assertTrue(all(state["step"].item() == 2 for state in checkpoint["optimizer_state"]["state"].values()))
        config = json.loads((self.one_epoch / "run_config.json").read_text())
        self.assertEqual(config["posterior_shape"], [4, 6, 15])
        with (self.one_epoch / "history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        for key in ("train_loss", "train_pixel", "train_gradient", "train_kl_per_latent", "val_rmse_log10"):
            self.assertTrue(np.isfinite(float(rows[0][key])), key)

    def test_legacy_frame_field_defaults_without_ignoring_data_identity(self):
        _, checkpoint = load_checkpoint(self.one_epoch / 'last.pt')
        modern = checkpoint['data_contract']
        legacy = {k:v for k,v in modern.items() if k != 'coordinate_frame'}
        self.assertTrue(same_data_contract(legacy, modern))
        self.assertTrue(same_data_contract(modern, legacy))
        self.assertFalse(same_data_contract(legacy, {**modern, 'coordinate_frame':'tool_aligned_yz'}))
        self.assertFalse(same_data_contract(legacy, {**modern, 'manifest_sha256':'changed'}))

    def test_resume_matches_uninterrupted_model_and_optimizer_exactly(self):
        resumed = self.output("resumed")
        uninterrupted = self.output("uninterrupted")
        train(DATASET, resumed, device_name="cpu", resume=self.one_epoch / "last.pt", epochs=2)
        train(DATASET, uninterrupted, self.config, device_name="cpu", epochs=2)
        _, a = load_checkpoint(resumed / "last.pt")
        _, b = load_checkpoint(uninterrupted / "last.pt")
        self.assertEqual(a["global_step"], 4)
        self.assertEqual(a["global_step"], b["global_step"])
        self.assertEqual(a["best_epoch"], b["best_epoch"])
        self.assertEqual(a["best_val_rmse"], b["best_val_rmse"])
        self.assertEqual(a["initialization"]["mode"], "resume")
        for key, value in a["model_state"].items():
            torch.testing.assert_close(value, b["model_state"][key], rtol=0, atol=0, msg=key)
        for parameter_id, state in a["optimizer_state"]["state"].items():
            other = b["optimizer_state"]["state"][parameter_id]
            self.assertEqual(state["step"].item(), 4)
            for key, value in state.items():
                torch.testing.assert_close(value, other[key], rtol=0, atol=0)
        self.assertEqual(a["optimizer_state"]["param_groups"], b["optimizer_state"]["param_groups"])
        for first, second in zip(a["history"], b["history"]):
            self.assertEqual({k: v for k, v in first.items() if k != "seconds"},
                             {k: v for k, v in second.items() if k != "seconds"})

    def test_evaluation_is_repeatable_and_writes_real_metrics(self):
        first_output, second_output = self.output("evaluate_a"), self.output("evaluate_b")
        first = evaluate(self.evaluation_args(first_output))
        second = evaluate(self.evaluation_args(second_output))
        self.assertEqual(first["splits"], second["splits"])
        summary = first["splits"]["test"]
        self.assertEqual(summary["evaluated_count"], 2)
        self.assertTrue(summary["partial_split"])
        self.assertTrue(np.isfinite(summary["overall"]["rmse_log10"]))
        first_array = np.load(first_output / "test" / "reconstructed_log10rho.npy")
        second_array = np.load(second_output / "test" / "reconstructed_log10rho.npy")
        self.assertEqual(first_array.shape, (2, 1, 48, 120))
        np.testing.assert_array_equal(first_array, second_array)
        self.assertEqual(json.loads((first_output / "metrics.json").read_text())["splits"], first["splits"])
        with (first_output / "test" / "sample_metrics.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_inversion_wrapper_preserves_raw_scale_order_and_latent_gradient(self):
        wrapper = LWDVAE(self.one_epoch / "best.pt", device="cpu")
        self.assertFalse(wrapper.training)
        self.assertFalse(wrapper.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.parameters()))
        with LWDVAEDataset(DATASET, "test", as_torch=True) as dataset:
            lo, hi = dataset.log10_bounds
            raw_log10rho = lo + (dataset[0].unsqueeze(0) + 1) * (hi - lo) / 2
        posterior = wrapper.encode(raw_log10rho)
        self.assertEqual(tuple(posterior.mean.shape), (1, 4, 6, 15))
        latent = posterior.mode().detach().requires_grad_(True)
        decoded = wrapper.decode(latent)
        self.assertEqual(tuple(decoded.shape), (1, 1, 48, 120))
        vector = wrapper.decode_inversion_vector(latent)
        self.assertEqual(tuple(vector.shape), (1, 48 * 120))
        expected_vector = decoded.detach().cpu().numpy()[0, 0].reshape(-1, order="F")
        np.testing.assert_array_equal(vector.detach().cpu().numpy()[0], expected_vector)
        # Decode must invert the dataset-wide affine normalization without
        # detaching or changing the cell order needed by the future EM solver.
        normalized_decoded = wrapper.model.decode(latent)
        torch.testing.assert_close(decoded, lo + (normalized_decoded + 1) * (hi - lo) / 2)
        vector.square().mean().backward()
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in wrapper.parameters()))
        for bad_input in (raw_log10rho.squeeze(0), raw_log10rho[..., :-1],
                          torch.full_like(raw_log10rho, hi + 1)):
            with self.subTest(shape=tuple(bad_input.shape)), self.assertRaises(ValueError):
                wrapper.encode(bad_input)
        with self.assertRaises(ValueError):
            wrapper.decode(torch.zeros(1, 4, 6, 14))

    def test_grid_mismatch_rejected_before_train_or_evaluation_output(self):
        changed = self.copied_dataset("changed_grid")
        path = changed / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["config"]["grid"]["y_bounds_ft"] = [-6.0, 6.0]
        path.write_text(json.dumps(manifest))
        for name, action in (
            ("invalid_grid_train", lambda out: train(changed, out, device_name="cpu", resume=self.one_epoch / "last.pt", epochs=2)),
            ("invalid_grid_eval", lambda out: evaluate(self.evaluation_args(out, changed))),
        ):
            with self.subTest(name=name):
                output = self.output(name)
                with self.assertRaisesRegex(ValueError, "grid|coordinates"):
                    action(output)
                self.assertFalse(output.exists())

    def test_resume_rejects_manifest_or_training_data_change(self):
        _, checkpoint = load_checkpoint(self.one_epoch / "last.pt")
        for kind in ("manifest", "array"):
            changed = self.copied_dataset(f"changed_{kind}")
            path = changed / "manifest.json"
            manifest = json.loads(path.read_text())
            if kind == "manifest":
                manifest["test_note"] = "Changed dataset identity"
                path.write_text(json.dumps(manifest))
            else:
                array = np.load(changed / manifest["splits"]["train"]["path"], mmap_mode="r+")
                array[0, 0, 0, 0] = 0.123
                array.flush()
                del array
            output = self.output(f"reject_changed_{kind}")
            with self.assertRaisesRegex(ValueError, "dataset identity mismatch"):
                train(changed, output, device_name="cpu", resume=self.one_epoch / "last.pt", epochs=2)
            self.assertFalse(output.exists())
            # New held-out sets may differ in identity but must keep the geometry
            # and normalization contract. Evaluation deliberately permits this.
            with LWDVAEDataset(changed, "test") as dataset:
                validate_data_contract(checkpoint["data_contract"], dataset, require_manifest=False)

    def test_existing_outputs_are_never_overwritten(self):
        sentinel = (self.one_epoch / "last.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            train(DATASET, self.one_epoch, self.config, device_name="cpu")
        with self.assertRaises(FileExistsError):
            evaluate(self.evaluation_args(self.one_epoch))
        self.assertEqual((self.one_epoch / "last.pt").read_bytes(), sentinel)

    def test_invalid_config_and_nonfinite_inputs_are_rejected(self):
        for key, value in (("learning_rate", float("nan")), ("kl_weight", float("inf")),
                           ("epochs", True), ("batch_size", 0), ("max_train_samples", 0),
                           ("gradient_weight", -1.0), ("reconstruction", "typo")):
            config = deepcopy(self.config)
            config["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training_config(config)
        duplicate = self.output("duplicate_config.json")
        duplicate.write_text('{"model": {}, "model": {}, "training": {}}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_training_config(duplicate)
        changed = self.copied_dataset("nonfinite_dataset")
        manifest = json.loads((changed / "manifest.json").read_text())
        array = np.load(changed / manifest["splits"]["train"]["path"], mmap_mode="r+")
        array[0, 0, 0, 0] = np.nan
        array.flush()
        del array
        output = self.output("nonfinite_run")
        with self.assertRaisesRegex(ValueError, "finite"):
            train(changed, output, self.config, device_name="cpu")
        self.assertFalse(output.exists())

    def test_nonfinite_training_loss_never_becomes_a_checkpoint(self):
        output = self.output("nonfinite_loss")
        nonfinite_terms = (torch.tensor(float("nan"), requires_grad=True), torch.tensor(0.0), torch.tensor(0.0))
        with patch("lwd_vae.training.reconstruction_terms", return_value=nonfinite_terms):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
                train(DATASET, output, self.config, device_name="cpu")
        self.assertFalse((output / "last.pt").exists())
        self.assertFalse((output / "best.pt").exists())

    def test_evaluation_cli_smoke(self):
        output = self.output("evaluation_cli")
        result = subprocess.run(
            [sys.executable, str(ROOT / "test_vae.py"), "--dataset", str(DATASET),
             "--checkpoint", str(self.one_epoch / "best.pt"), "--output", str(output),
             "--splits", "test", "--device", "cpu", "--max-samples", "2",
             "--batch-size", "2", "--no-plots"],
            cwd=ROOT, text=True, capture_output=True, timeout=120,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "metrics.json").read_text())
        self.assertEqual(report["splits"]["test"]["evaluated_count"], 2)


if __name__ == "__main__":
    unittest.main()
