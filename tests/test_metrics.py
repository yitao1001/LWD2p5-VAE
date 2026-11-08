"""Numerical metric tests independent of PyTorch and training quality."""

import json
import math
import unittest

import numpy as np

from lwd_vae.evaluation import reconstruction_metrics, summarize_metrics, unrestricted_denormalize


class MetricsTests(unittest.TestCase):
    def test_identity(self):
        image = np.arange(12, dtype=float).reshape(1, 1, 3, 4) / 4
        metric = reconstruction_metrics(image, image, [-1, 3])[0]
        for field in ("rmse_log10", "mae_log10", "edge_mae_log10_per_cell", "relative_rho_l2", "out_of_bounds_fraction"):
            self.assertEqual(metric[field], 0.0)
        self.assertIsNone(metric["log10_relative_rho_l2"])
        json.dumps(metric, allow_nan=False)

    def test_known_constant_offset_and_zero_edges(self):
        truth = np.full((2, 1, 3, 4), 1.0)
        metric = reconstruction_metrics(truth, truth + math.log10(2), [-1, 3])[0]
        self.assertAlmostEqual(metric["rmse_log10"], math.log10(2))
        self.assertAlmostEqual(metric["mae_log10"], math.log10(2))
        self.assertAlmostEqual(metric["relative_rho_l2"], 1.0)
        self.assertEqual(metric["edge_mae_log10_per_cell"], 0.0)

    def test_edge_metric_pools_axes_by_edge_count(self):
        truth = np.zeros((1, 1, 2, 3))
        prediction = np.array([[[[0, 1, 2], [0, 1, 2]]]])
        metric = reconstruction_metrics(truth, prediction, [-1, 3])[0]
        self.assertAlmostEqual(metric["edge_mae_log10_per_cell"], 4 / 7)

    def test_group_and_overall_aggregate_samples_not_batches(self):
        truth = np.zeros((3, 1, 2, 2))
        prediction = np.broadcast_to(np.array([0, 0, 3])[:, None, None, None], truth.shape)
        rows = reconstruction_metrics(truth, prediction, [-1, 3])
        for i, row in enumerate(rows):
            row.update(family="layered" if i < 2 else "faulted", contrast_class="weak" if i < 2 else "strong")
        summary = summarize_metrics(rows)
        self.assertAlmostEqual(summary["overall"]["rmse_log10"], math.sqrt(3))
        self.assertAlmostEqual(summary["overall"]["mae_log10"], 1)
        self.assertEqual(summary["by_family"]["layered"]["count"], 2)
        self.assertEqual(summary["by_family"]["faulted"]["rmse_log10"], 3)
        self.assertEqual(summary["by_contrast_class"]["weak"]["mae_log10"], 0)

    def test_decoder_out_of_bounds_not_clipped(self):
        raw = unrestricted_denormalize(np.array([[[[2.0, -2.0]]]]), [-1, 3])
        np.testing.assert_array_equal(raw, [[[[5.0, -3.0]]]])
        rows = reconstruction_metrics(np.zeros_like(raw), raw, [-1, 3])
        self.assertEqual(rows[0]["out_of_bounds_fraction"], 1.0)
        self.assertAlmostEqual(rows[0]["rmse_log10"], math.sqrt(17))

    def test_resistivity_overflow_is_explicit_and_json_safe(self):
        truth = np.zeros((1, 1, 2, 2))
        rows = reconstruction_metrics(truth, truth + 1000, [-1, 3])
        self.assertIsNone(rows[0]["relative_rho_l2"])
        self.assertTrue(rows[0]["relative_rho_l2_overflow"])
        self.assertAlmostEqual(rows[0]["log10_relative_rho_l2"], 1000)
        self.assertEqual(rows[0]["rho_exponent_overflow_pixels"], 4)
        summary = summarize_metrics(rows)
        self.assertIsNone(summary["overall"]["mean_relative_rho_l2"])
        self.assertEqual(summary["overall"]["relative_rho_l2_overflow_count"], 1)
        json.dumps(summary, allow_nan=False)
        json.dumps(rows, allow_nan=False)

    def test_negative_large_prediction_does_not_overflow(self):
        truth = np.zeros((1, 1, 2, 2))
        metric = reconstruction_metrics(truth, truth - 1000, [-1, 3])[0]
        self.assertAlmostEqual(metric["relative_rho_l2"], 1)
        self.assertEqual(metric["rho_exponent_underflow_pixels"], 4)
        self.assertFalse(metric["relative_rho_l2_overflow"])

    def test_nonfinite_and_shape_errors(self):
        image = np.zeros((1, 1, 2, 2))
        for bad in (math.inf, math.nan, -math.inf):
            with self.assertRaises(ValueError):
                reconstruction_metrics(image, np.full_like(image, bad), [-1, 3])
            with self.assertRaises(ValueError):
                unrestricted_denormalize([bad], [-1, 3])
        with self.assertRaises(ValueError):
            reconstruction_metrics(image, image[0], [-1, 3])
        with self.assertRaises(ValueError):
            reconstruction_metrics(image, image, [3, -1])
        with self.assertRaises(ValueError):
            reconstruction_metrics(image[:0], image[:0], [-1, 3])


if __name__ == "__main__":
    unittest.main()
