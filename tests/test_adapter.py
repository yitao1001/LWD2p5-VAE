"""Regression checks for normalization, physical coordinates, and ordering."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lwd_vae.dataset import LWDVAEDataset, denormalize_log10rho, normalize_log10rho
from lwd_vae.mapping import from_inversion_vector, map_to_inversion_grid, to_inversion_vector


class AdapterTests(unittest.TestCase):
    def test_asymmetric_fortran_order(self):
        image = np.array([[11, 12, 13], [21, 22, 23]], dtype=np.float32)
        expected = np.array([11, 21, 12, 22, 13, 23])
        np.testing.assert_array_equal(to_inversion_vector(image), expected)
        np.testing.assert_array_equal(from_inversion_vector(expected, (2, 3)), image)
        with self.assertRaises(ValueError):
            from_inversion_vector(expected, (3, 3))
        with self.assertRaises(ValueError):
            to_inversion_vector(image[None])

    def test_global_normalization_roundtrip(self):
        raw = np.array([-1.0, -0.5, 0.0, 1.0, 2.75, 3.0], dtype=np.float32)
        normalized = normalize_log10rho(raw, [-1, 3])
        np.testing.assert_allclose(normalized[[0, 3, -1]], [-1, 0, 1])
        np.testing.assert_allclose(denormalize_log10rho(normalized, [-1, 3]), raw, atol=1e-6)
        # A constant image retains its absolute value; it is not normalized per image.
        np.testing.assert_array_equal(normalize_log10rho(np.full((2, 3), 2.0), [-1, 3]), 0.5)
        for bad in ([np.nan], [np.inf], [-1.1], [3.1]):
            with self.assertRaises(ValueError):
                normalize_log10rho(bad, [-1, 3])
        with self.assertRaises(ValueError):
            denormalize_log10rho([1.2], [-1, 3])

    def test_physical_bilinear_on_nonuniform_grid(self):
        sy, sz = np.array([-5, -3, 0, 2, 5.0]), np.array([-2, -1, 0.5, 2.0])
        yy, zz = np.meshgrid((sy[:-1] + sy[1:]) / 2, (sz[:-1] + sz[1:]) / 2)
        image = 2.0 + 0.1 * yy - 0.3 * zz
        ty, tz = np.array([-3.0, -1.0, 1.0, 3.0]), np.array([-1.0, 0.0, 1.0])
        target_y, target_z = np.meshgrid((ty[:-1] + ty[1:]) / 2, (tz[:-1] + tz[1:]) / 2)
        expected = 2.0 + 0.1 * target_y - 0.3 * target_z
        mapped = map_to_inversion_grid(image, sy, sz, ty, tz)
        np.testing.assert_allclose(mapped, expected, atol=1e-12)
        vector = map_to_inversion_grid(image, sy, sz, ty, tz, as_vector=True)
        np.testing.assert_allclose(vector, expected.ravel(order="F"), atol=1e-12)
        np.testing.assert_allclose(map_to_inversion_grid(image, sy, sz, sy, sz), image)

    def test_domain_and_edge_policy(self):
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        edges = [0.0, 1.0, 2.0]
        with self.assertRaisesRegex(ValueError, "physical domain"):
            map_to_inversion_grid(image, edges, edges, [-0.1, 1.0], [0, 1])
        with self.assertRaises(ValueError):
            map_to_inversion_grid(image, edges, edges, [0, 1, 0.5], [0, 1])
        with self.assertRaises(ValueError):
            map_to_inversion_grid(image, [0, 1], edges, [0, 1], [0, 1])
        # Centers in the outer half cell use the nearest source center.
        mapped = map_to_inversion_grid(image, edges, edges, [0, 0.2], [0, 0.2])
        np.testing.assert_array_equal(mapped, [[1.0]])
        single = map_to_inversion_grid([[3.0]], [0, 2], [0, 2], [0, 1, 2], [0, 1, 2])
        np.testing.assert_array_equal(single, np.full((2, 2), 3.0))

    def _fixture(self, root, data=None):
        if data is None:
            data = np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3) / 3.0 - 1.0
        (root / "train").mkdir(exist_ok=True)
        np.save(root / "train/log10rho.npy", data)
        manifest = {
            "schema_version": 1,
            "config": {
                "grid": {"nz": 2, "ny": 3, "y_bounds_ft": [-5, 5], "z_bounds_ft": [-2, 2]},
                "resistivity": {"log10_bounds": [-1, 3]},
            },
            "splits": {"train": {"count": 2, "path": "train/log10rho.npy", "metadata": "train/metadata.jsonl"}},
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "train/metadata.jsonl").write_text('{"id": 0}\n{"id": 1}\n', encoding="utf-8")
        return data

    def test_loader_memmap_metadata_and_worker_reopening(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = self._fixture(root)
            dataset = LWDVAEDataset(root, return_metadata=True)
            self.assertIsNone(dataset._array)
            self.assertEqual(len(dataset), 2)
            image, metadata = dataset[0]
            self.assertEqual(image.shape, (1, 2, 3))
            self.assertEqual(image.dtype, np.float32)
            self.assertTrue(image.flags.writeable)
            self.assertIsInstance(dataset._array, np.memmap)
            self.assertEqual(metadata["id"], 0)
            np.testing.assert_allclose(denormalize_log10rho(image, [-1, 3]), raw[0], atol=1e-6)
            self.assertEqual(dataset[-1][1]["id"], 1)
            with self.assertRaises(IndexError):
                _ = dataset[2]
            reopened = pickle.loads(pickle.dumps(dataset))
            self.assertIsNone(reopened._array)
            np.testing.assert_array_equal(reopened[0][0], image)
            old_array = reopened._array
            reopened._array_pid = -1  # Simulate receiving the object after fork.
            np.testing.assert_array_equal(reopened[0][0], image)
            self.assertIsNot(reopened._array, old_array)
            dataset.validate_all(chunk_size=1)
            dataset.close()
            reopened.close()

    def test_malformed_arrays_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for bad in (np.zeros((2, 1, 3, 2), np.float32), np.zeros((2, 1, 2, 3), np.float64)):
                self._fixture(root, bad)
                with self.assertRaises(ValueError):
                    LWDVAEDataset(root)
            for bad_value in (np.nan, np.inf, 4.0):
                data = np.zeros((2, 1, 2, 3), np.float32)
                data[1, 0, 0, 0] = bad_value
                self._fixture(root, data)
                with LWDVAEDataset(root) as dataset:
                    _ = dataset[0]
                    with self.assertRaisesRegex(ValueError, "invalid sample 1"):
                        _ = dataset[1]
                    with self.assertRaises(ValueError):
                        dataset.validate_all()
            self._fixture(root)
            (root / "train/metadata.jsonl").write_text('{"id": 0}\n', encoding="utf-8")
            with LWDVAEDataset(root) as dataset:
                with self.assertRaisesRegex(ValueError, "expected 2"):
                    _ = dataset.metadata


if __name__ == "__main__":
    unittest.main()
