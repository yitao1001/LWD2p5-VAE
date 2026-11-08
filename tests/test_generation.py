"""Physical/data contracts, deterministic replay, no cross-split leakage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lwd_vae.dataset import LWDVAEDataset
from lwd_vae.generation import FAMILIES, GENERATOR_VERSION, SPLITS, generate_dataset, grid_coordinates, load_config, sample_model, validate_config
from test_vae import evaluation_splits


CONFIG = Path(__file__).resolve().parents[1] / "configs/lwd_default.json"


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_family_physics_and_replay(self):
        for family in FAMILIES:
            rng = np.random.default_rng(40)
            image, params = sample_model(self.config, rng, family)
            other, other_params = sample_model(self.config, np.random.default_rng(40), family)
            np.testing.assert_array_equal(image, other)
            self.assertEqual(params, other_params)
            self.assertEqual(image.shape, (48, 120))
            self.assertEqual(image.dtype, np.float32)
            self.assertTrue(np.isfinite(image).all())
            self.assertGreaterEqual(image.min(), -0.302)
            self.assertLessEqual(image.max(), 2.302)
            if family == "homogeneous":
                self.assertEqual(float(np.ptp(image)), 0)
            else:
                self.assertGreater(float(np.ptp(image)), 0)
                self.assertTrue(np.all(np.diff(params["stratigraphic_boundaries_ft"]) > 0))

    def test_invalid_config_fails_early(self):
        for path, value in [(("grid", "ny"), 119), (("grid", "nz"), True),
                            (("geometry", "bed_thickness_ft"), [0.01, 1.0]),
                            (("family_weights", "layered"), 0.7),
                            (("splits", "train"), -1),
                            (("geometry", "background_bedding"), "randomly_rotated"),
                            (("geometry", "dip_abs_deg"), [0, 15]),
                            (("geometry", "anticline_amplitude_ft"), [0.01, 0.75]),
                            (("geometry", "anticline_limb_width_ft"), [0.01, 3.5]),
                            (("geometry", "anticline_center_y_fraction"), [-0.6, 0.1]),
                            (("geometry", "anticline_fault_tilt_abs_deg"), [20, 75]),
                            (("ood", "dip_abs_deg"), [25, 45]),
                            (("ood", "delta_log10"), [1, 2]),
                            (("resistivity", "sampling_log10_bounds"), [-2, 3])]:
            config = deepcopy(self.config)
            config[path[0]][path[1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_config(config)
        config = deepcopy(self.config)
        config["unknown"] = True
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{"seed": 1, "seed": 2}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_config(path)

    def test_global_background_has_no_overall_tilt(self):
        for seed in range(20):
            for family in FAMILIES:
                with self.subTest(seed=seed, family=family):
                    image, params = sample_model(self.config, np.random.default_rng(seed), family)
                    self.assertEqual(params['coordinate_frame'], 'global_aligned_yz')
                    self.assertEqual(params['overall_rotation_deg'], 0)
                    self.assertNotIn('dip_deg', params)
                    if family in ('homogeneous', 'layered', 'fluid_contact'):
                        # 每一行数值恒定，所有背景层界严格水平。
                        np.testing.assert_array_equal(image, np.repeat(image[:, :1], image.shape[1], axis=1))
        for family in ('curved', 'anticline', 'faulted', 'faulted_anticline', 'pinchout', 'unconformity'):
            image, _ = sample_model(self.config, np.random.default_rng(40), family)
            self.assertTrue(np.any(image != image[:, :1]), family)

    def test_removed_body_families_cannot_be_requested_or_reenabled(self):
        for family in ('block', 'lens'):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, 'Unknown geological family'):
                sample_model(self.config, np.random.default_rng(1), family)
            for weight in (0.0, 0.05):
                config = deepcopy(self.config)
                config['family_weights'][family] = weight
                config['family_weights']['layered'] -= weight
                with self.subTest(family=family, weight=weight), self.assertRaisesRegex(ValueError, f'{family} family was removed'):
                    validate_config(config)

    def test_anticline_horizons_arch_upward_and_preserve_layer_spacing(self):
        config = deepcopy(self.config)
        config['geometry']['boundary_count'] = [1, 2]
        checked = 0
        for seed in range(30):
            image, meta = sample_model(config, np.random.default_rng(seed), 'anticline')
            traces = [np.flatnonzero(np.diff(image[:, col])) for col in range(image.shape[1])]
            n = len(meta['stratigraphic_boundaries_ft'])
            # 只在所有层界都位于窗口内时比较完整两翼，避免把裁切当作消失。
            if any(len(trace) != n for trace in traces):
                continue
            traces = np.array(traces).T
            for trace in traces:
                crest = np.argmin(trace)
                self.assertGreater(trace[0], trace[crest])
                self.assertGreater(trace[-1], trace[crest])
                self.assertTrue(np.all(np.diff(trace[:crest+1]) <= 0))
                self.assertTrue(np.all(np.diff(trace[crest:]) >= 0))
            if n > 1:
                # 连续层界共享位移；离散后的层间距只允许一个单元的取整差。
                self.assertTrue(np.all(np.ptp(np.diff(traces, axis=0), axis=1) <= 1))
            checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_faulted_anticline_displaces_only_one_side_of_inclined_fault(self):
        _, _, y, z = grid_coordinates(self.config)
        tilt_signs, throw_signs = set(), set()
        for seed in range(20):
            plain, plain_meta = sample_model(self.config, np.random.default_rng(seed), 'anticline')
            faulted, meta = sample_model(self.config, np.random.default_rng(seed), 'faulted_anticline')
            for key in ('anticline_amplitude_ft', 'anticline_center_y_ft', 'anticline_left_width_ft', 'anticline_right_width_ft'):
                self.assertEqual(meta[key], plain_meta[key])
            slope = np.tan(np.deg2rad(meta['fault_tilt_deg']))
            moved = y > meta['fault_y_ft'] + (z - meta['fault_reference_z_ft']) * slope
            np.testing.assert_array_equal(plain[~moved], faulted[~moved])
            self.assertTrue(np.any(plain[moved] != faulted[moved]))
            # 滑移向量与断层面平行，两侧样本必须覆盖相反倾向和错动方向。
            self.assertAlmostEqual(meta['fault_lateral_shift_ft'] / meta['throw_ft'], slope)
            tilt_signs.add(np.sign(meta['fault_tilt_deg']))
            throw_signs.add(np.sign(meta['throw_ft']))
        self.assertEqual(tilt_signs, {-1, 1})
        self.assertEqual(throw_signs, {-1, 1})

    def test_old_rotation_config_is_rejected_with_migration_hint(self):
        old = CONFIG.parent.parent / 'references/dataset_v1/lwd_default_original.json'
        with self.assertRaisesRegex(ValueError, 'schema_version=2'):
            load_config(old)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            sample_model(self.config, np.random.default_rng(1), 'layered', 'ood_dip')

    def test_local_window_translation_keeps_global_axes_and_cell_size(self):
        shifted = deepcopy(self.config)
        shifted['grid']['y_bounds_ft'] = [v+255 for v in shifted['grid']['y_bounds_ft']]
        shifted['grid']['z_bounds_ft'] = [v+1780 for v in shifted['grid']['z_bounds_ft']]
        for family in FAMILIES:
            a, _ = sample_model(self.config, np.random.default_rng(40), family)
            b, _ = sample_model(shifted, np.random.default_rng(40), family)
            np.testing.assert_array_equal(a, b)

    def test_default_evaluation_splits_follow_dataset_version(self):
        root = CONFIG.parent.parent
        self.assertEqual(evaluation_splits(root/'datasets/smoke_v1', None), ['test', 'ood_dip', 'ood_contrast'])
        self.assertEqual(evaluation_splits(root/'datasets/smoke_global_v2', None), ['test', 'ood_contrast'])
        with self.assertRaisesRegex(ValueError, 'absent'):
            evaluation_splits(root/'datasets/smoke_global_v2', ['ood_dip'])

    def test_end_to_end_split_contract_and_exact_replay(self):
        config = deepcopy(self.config)
        config["splits"] = {s: 200 if s == "train" else 20 for s in SPLITS}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dataset"
            manifest = generate_dataset(config, output)
            self.assertEqual(manifest['schema_version'], 2)
            self.assertEqual(manifest['generator_version'], GENERATOR_VERSION)
            self.assertEqual(manifest['coordinate_frame'], 'global_aligned_yz')
            self.assertFalse(manifest['geometry_policy']['random_overall_rotation'])
            self.assertFalse(manifest['geometry_policy']['isolated_block_bodies'])
            self.assertFalse(manifest['geometry_policy']['isolated_lens_bodies'])
            self.assertNotIn('ood_dip', manifest['splits'])
            all_hashes = set()
            for split in SPLITS:
                with LWDVAEDataset(output, split, return_metadata=True) as dataset:
                    dataset.validate_all()
                    self.assertEqual(len(dataset), config["splits"][split])
                    records = dataset.metadata
                    self.assertNotIn('block', manifest['splits'][split]['family_counts'])
                    self.assertNotIn('block', {r['family'] for r in records})
                    self.assertNotIn('lens', manifest['splits'][split]['family_counts'])
                    self.assertNotIn('lens', {r['family'] for r in records})
                    if split == "train":
                        self.assertEqual({r["family"] for r in records}, set(FAMILIES))
                        self.assertEqual(manifest['splits'][split]['family_counts'],
                                         {f: int(200 * weight) for f, weight in config['family_weights'].items()})
                    array = np.load(output / manifest["splits"][split]["path"], mmap_mode="r")
                    for record in records:
                        raw = array[record["index"], 0]
                        digest = hashlib.sha256(raw.astype("<f4").tobytes()).hexdigest()
                        self.assertEqual(digest, record["model_sha256"])
                        self.assertNotIn(digest, all_hashes)
                        all_hashes.add(digest)
                        self.assertEqual(record['parameters']['overall_rotation_deg'], 0)
                        if split == "ood_contrast":
                            self.assertGreaterEqual(np.ptp(raw), 1.8 - 1e-6)
                    record = records[-1]
                    rng = np.random.default_rng(np.random.SeedSequence(record["seed_sequence"]))
                    for _ in range(record["accepted_attempt"] + 1):
                        replay, _ = sample_model(config, rng, record["family"], split)
                    np.testing.assert_array_equal(replay, array[record["index"], 0])
                    del array
            self.assertEqual(len(all_hashes), sum(config["splits"].values()))
            before = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                generate_dataset(config, output)
            self.assertEqual(before, (output / "manifest.json").read_bytes())

    def test_loader_rejects_tool_frame_mislabeled_as_global(self):
        config = deepcopy(self.config)
        config['splits'] = {s: int(s == 'train') for s in SPLITS}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'dataset'
            manifest = generate_dataset(config, output)
            manifest['coordinate_frame'] = 'tool_aligned_yz'
            (output/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'coordinate frame'):
                LWDVAEDataset(output, 'train')

    def test_failed_generation_does_not_publish_partial_dataset(self):
        config = deepcopy(self.config)
        config["splits"] = {s: int(s == "train") for s in SPLITS}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dataset"
            with patch("lwd_vae.generation.sample_model", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    generate_dataset(config, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
