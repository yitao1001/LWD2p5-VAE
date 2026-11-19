"""Compositional generator (lwd_v4): physical/data contracts, component geometry, replay, no leakage."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lwd_vae.dataset import LWDVAEDataset
from lwd_vae.generation import (_WATER, CONTRAST_CLASSES, DIP_CLASSES, GENERATOR_VERSION, SPLITS, TAGS,
                                RejectedSample, _draw, _pinchout_interval, generate_dataset, grid_coordinates,
                                load_config, replay_sample, sample_model, validate_config)
from lwd_vae.mapping import to_inversion_vector
from test_vae import evaluation_splits


CONFIG = Path(__file__).resolve().parents[1] / "configs/lwd_default.json"
COMPONENTS = ("undulation", "fault", "pinchout", "erosion", "fluid_contact")


def only(config, *names, gentle=None):
    """Config with every optional component off except `names` (probability 1)."""
    c = deepcopy(config)
    for name in COMPONENTS:
        c["components"][name]["probability"] = 1.0 if name in names else 0.0
    c["stratigraphy"]["thick_bed_probability"] = 0.0
    if gentle is not None:
        c["dip"]["gentle_probability"] = gentle
    return c


def draws(config, count, split="train", seed=0):
    """Accepted draws (image, info, scene, (y, z)) from independent streams."""
    out, i = [], 0
    while len(out) < count:
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        i += 1
        try:
            out.append(_draw(config, rng, split))
        except RejectedSample:
            continue
    return out


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_grid_is_square_one_inch_window_and_fortran_order(self):
        grid = self.config["grid"]
        self.assertEqual((grid["nz"], grid["ny"]), (128, 128))
        ye, ze, _, _ = grid_coordinates(self.config)
        np.testing.assert_allclose(np.diff(ye), 1 / 12, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.diff(ze), 1 / 12, rtol=0, atol=1e-12)
        np.testing.assert_allclose([ye[0], ye[-1], ze[0], ze[-1]], [-16 / 3, 16 / 3, -16 / 3, 16 / 3], atol=1e-12)
        image = np.arange(128 * 128, dtype=float).reshape(128, 128)  # value = iz*128 + iy
        vector = to_inversion_vector(image)
        iy, iz = 5, 7
        self.assertEqual(vector[iy * 128 + iz], image[iz, iy])  # z fastest

    def test_replay_bounds_and_labels_for_every_split(self):
        lo, hi = self.config["resistivity"]["sampling_log10_bounds"]
        for split in SPLITS:
            for seed in range(6):
                with self.subTest(split=split, seed=seed):
                    try:
                        image, info = sample_model(self.config, np.random.default_rng(seed), split)
                    except RejectedSample:
                        continue
                    again, info_again = sample_model(self.config, np.random.default_rng(seed), split)
                    np.testing.assert_array_equal(image, again)
                    self.assertEqual(info, info_again)
                    self.assertEqual((image.shape, image.dtype), ((128, 128), np.float32))
                    self.assertTrue(np.isfinite(image).all())
                    self.assertGreaterEqual(image.min(), lo - 1e-5)
                    self.assertLessEqual(image.max(), hi + 1e-5)
                    self.assertTrue(set(info["tags"]) <= set(TAGS) and info["tags"])
                    self.assertIn(info["contrast_class"], CONTRAST_CLASSES)
                    self.assertIn(info["dip_class"], DIP_CLASSES)
                    json.dumps(info)  # metadata must be JSON serializable

    def test_contrast_and_dip_limits_per_split(self):
        strong = self.config["resistivity"]["strong_span_log10"][1]
        for image, info, _, _ in draws(self.config, 60, "train"):
            self.assertLessEqual(float(np.ptp(image)), strong + 1e-5)
            self.assertLessEqual(abs(info["parameters"]["dip_deg"]), self.config["dip"]["max_abs_deg"])
        ood_lo, ood_hi = self.config["ood"]["contrast_span_log10"]
        for image, info, _, _ in draws(self.config, 30, "ood_contrast"):
            self.assertGreaterEqual(float(np.ptp(image)), ood_lo - 1e-5)
            self.assertLessEqual(float(np.ptp(image)), ood_hi + 1e-5)
            self.assertEqual(info["contrast_class"], "ood")
        for _, info, _, _ in draws(self.config, 30, "ood_dip"):
            self.assertTrue(15 <= abs(info["parameters"]["dip_deg"]) <= 25)
            self.assertEqual(info["dip_class"], "ood")

    def test_stack_palette_thickness_and_gradational_rules(self):
        clip = self.config["stratigraphy"]["thickness_clip_ft"]
        for image, info, scene, _ in draws(self.config, 40):
            if scene is None:
                continue
            stack = scene.lower
            self.assertTrue(np.all(np.diff(stack.palette_index) != 0))  # adjacent beds differ in lithology
            self.assertTrue(np.all((stack.thickness >= clip[0] - 1e-12) & (stack.thickness <= clip[1] + 1e-12)))
            for j in np.flatnonzero(stack.gradational):
                self.assertLessEqual(stack.gradational[j], 0.5 * min(stack.thickness[j - 1], stack.thickness[j]) + 1e-12)
            palette = np.asarray(info["parameters"]["palette_log10rho"])
            self.assertTrue(np.all(np.diff(palette) >= self.config["resistivity"]["min_palette_gap_log10"] - 1e-12))

    def test_measured_dip_matches_parameter(self):
        config = only(self.config, gentle=0.0)
        config["properties"].update(trend_probability=0.0, gradational_probability=0.0)
        cell = 1 / 12
        checked = 0
        for image, info, _, (y, z) in draws(config, 12, seed=3):
            if len(np.unique(image)) < 2:
                continue
            dip = info["parameters"]["dip_deg"]
            angles = np.arange(-20, 20.001, 0.1)
            scores = []
            for angle in angles:
                s = z * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
                _, bins = np.unique(np.floor(s / (cell / 2)), return_inverse=True)
                counts = np.bincount(bins.ravel())
                sums = np.bincount(bins.ravel(), image.ravel())
                squares = np.bincount(bins.ravel(), image.ravel().astype(float) ** 2)
                scores.append(float(np.sum(squares - sums ** 2 / counts)))
            self.assertLess(abs(angles[int(np.argmin(scores))] - dip), 0.5, info["parameters"])
            checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_fault_moves_only_one_side_and_covers_both_senses(self):
        senses, tilts = set(), []
        for image, info, scene, (y, z) in draws(only(self.config, "fault"), 25, seed=4):
            self.assertIn("fault", info["tags"])
            for i, fault in enumerate(scene.faults):
                tilts.append(abs(fault["tilt_from_vertical_deg"]))
                senses.add(fault["sense"])
                without, _, _ = scene.evaluate(y, z, skip_fault=i)
                full, _, _ = scene.evaluate(y, z)
                slope = math.tan(math.radians(fault["tilt_from_vertical_deg"]))
                moved = y > fault["anchor_y_ft"] + (z - fault["anchor_z_ft"]) * slope
                if len(scene.faults) == 1:
                    np.testing.assert_array_equal(full[~moved], without[~moved])
                self.assertGreaterEqual(np.count_nonzero(full != without), self.config["quality"]["min_visible_cells"])
        self.assertEqual(senses, {"normal", "reverse"})
        lo, hi = self.config["components"]["fault"]["tilt_from_vertical_deg"]
        self.assertTrue(all(lo <= t <= hi for t in tilts))

    def test_pinchout_tapers_to_zero_with_random_tip_and_mode(self):
        modes, tip_inside = set(), set()
        limit = self.config["components"]["pinchout"]["max_wedge_angle_deg"]
        for image, info, scene, _ in draws(only(self.config, "pinchout"), 40, seed=5):
            p = info["parameters"]["pinchout"]
            if p is None:
                continue
            modes.add(p["mode"])
            tip_inside.add(p["tip_in_window"])
            self.assertLessEqual(p["wedge_angle_deg"], limit + 1e-9)
            beyond = p["tip_u_ft"] + p["direction"] * np.array([0.01, 1.0, 5.0])
            full = p["tip_u_ft"] - p["direction"] * (p["taper_length_ft"] + np.array([0.0, 1.0]))
            low, high = _pinchout_interval(p, scene.lower, beyond)
            np.testing.assert_allclose(high - low, 0, atol=1e-12)  # thickness zero beyond the tip
            low, high = _pinchout_interval(p, scene.lower, full)
            np.testing.assert_allclose(high - low, p["host_thickness_ft"], atol=1e-9)
        self.assertEqual(modes, {"top_descends", "base_rises", "symmetric"})
        self.assertEqual(tip_inside, {True, False})

    def test_fluid_contact_is_horizontal_across_dipping_beds(self):
        config = only(self.config, "fluid_contact", gentle=0.0)
        checked = 0
        for image, info, scene, (y, z) in draws(config, 40, seed=7):
            f = scene.fluid
            if f is None or abs(info["parameters"]["dip_deg"]) < 3:
                continue
            _, dry_unit, _ = scene.evaluate(y, z, skip=("fluid_contact",))
            _, unit, _ = scene.evaluate(y, z)
            reservoir = scene.palette_index(dry_unit) == f["palette_index"]
            wet = unit >= _WATER
            self.assertFalse(np.any(wet & ~reservoir))
            self.assertFalse(np.any(wet & (z <= f["contact_z_ft"] - f["transition_ft"])))
            self.assertTrue(np.all(wet[reservoir & (z >= f["contact_z_ft"])]))  # one horizontal level for all beds
            checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_erosion_truncates_lower_stack(self):
        checked = 0
        for image, info, scene, (y, z) in draws(only(self.config, "erosion"), 15, seed=8):
            if scene.erosion is None:
                continue
            u, s = scene.material(y, z)
            cover = scene.eroded(u, s)
            _, unit, _ = scene.evaluate(y, z)
            self.assertTrue(np.all(unit[cover] >= 10_000) and np.all(unit[~cover] < 10_000))
            checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_default_config_produces_every_tag(self):
        seen = set()
        for _, info, _, _ in draws(self.config, 300, seed=9):
            seen.update(info["tags"])
        self.assertEqual(seen, set(TAGS))

    def test_invalid_config_fails_early(self):
        cases = [(("grid", "ny"), 121), (("grid", "nz"), True),
                 (("stratigraphy", "thickness_clip_ft"), [0.1, 12.0]),
                 (("resistivity", "palette_size_probabilities"), {"2": 0.5, "3": 0.4}),
                 (("resistivity", "sampling_log10_bounds"), [-2, 3]),
                 (("ood", "contrast_span_log10"), [2.0, 3.0]),
                 (("ood", "dip_abs_deg"), [10.0, 25.0]),
                 (("dip", "gentle_abs_deg"), 20.0),
                 (("splits", "train"), -1)]
        for path, value in cases:
            config = deepcopy(self.config)
            config[path[0]][path[1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_config(config)
        for broken in ({"unknown": True}, {"family_weights": {"layered": 1.0}}):
            config = deepcopy(self.config)
            config.update(broken)
            with self.assertRaises(ValueError):
                validate_config(config)
        config = deepcopy(self.config)
        del config["splits"]["ood_dip"]
        with self.assertRaises(ValueError):
            validate_config(config)
        config = deepcopy(self.config)
        config["components"]["pinchout"]["max_wedge_angle_deg"] = 75
        with self.assertRaises(ValueError):
            validate_config(config)
        old = deepcopy(self.config)
        old["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema_version=3"):
            validate_config(old)
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            sample_model(self.config, np.random.default_rng(1), "ood_family")

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{"seed": 1, "seed": 2}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_config(path)

    def test_window_translation_keeps_geology(self):
        shifted = deepcopy(self.config)
        shifted["grid"]["y_bounds_ft"] = [v + 255 for v in shifted["grid"]["y_bounds_ft"]]
        shifted["grid"]["z_bounds_ft"] = [v + 1780 for v in shifted["grid"]["z_bounds_ft"]]
        for seed in range(5):
            a, _ = sample_model(self.config, np.random.default_rng(seed))
            b, _ = sample_model(shifted, np.random.default_rng(seed))
            np.testing.assert_array_equal(a, b)

    def test_default_evaluation_splits_follow_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for names in (["test", "ood_contrast", "ood_dip"], ["test"]):
                (root / "manifest.json").write_text(json.dumps({"splits": {n: {"count": 1} for n in names}}))
                self.assertEqual(evaluation_splits(root, None), names)
            with self.assertRaisesRegex(ValueError, "absent"):
                evaluation_splits(root, ["ood_dip"])

    def test_end_to_end_split_contract_and_exact_replay(self):
        config = deepcopy(self.config)
        config["splits"] = {s: 60 if s == "train" else 12 for s in SPLITS}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dataset"
            manifest = generate_dataset(config, output)
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(manifest["generator_version"], GENERATOR_VERSION)
            self.assertEqual(manifest["coordinate_frame"], "global_aligned_yz")
            self.assertFalse(manifest["geometry_policy"]["image_rotation"])
            self.assertTrue(manifest["geometry_policy"]["formation_dip_in_image"])
            self.assertEqual(manifest["grid_contract"]["shape_nz_ny"], [128, 128])
            all_hashes = set()
            for split in SPLITS:
                with LWDVAEDataset(output, split, return_metadata=True) as dataset:
                    dataset.validate_all()
                    self.assertEqual(len(dataset), config["splits"][split])
                    records = dataset.metadata
                    info = manifest["splits"][split]
                    self.assertEqual(sum(info["contrast_counts"].values()), len(records))
                    self.assertEqual(sum(info["dip_counts"].values()), len(records))
                    array = np.load(output / info["path"], mmap_mode="r")
                    for record in records:
                        raw = array[record["index"], 0]
                        digest = hashlib.sha256(raw.astype("<f4").tobytes()).hexdigest()
                        self.assertEqual(digest, record["model_sha256"])
                        self.assertNotIn(digest, all_hashes)
                        all_hashes.add(digest)
                        if split == "ood_contrast":
                            self.assertGreaterEqual(np.ptp(raw), 2.5 - 1e-5)
                        if split == "ood_dip":
                            self.assertEqual(record["dip_class"], "ood")
                    replay, _ = replay_sample(config, records[-1], split)
                    np.testing.assert_array_equal(replay, array[records[-1]["index"], 0])
                    del array
            self.assertEqual(len(all_hashes), sum(config["splits"].values()))
            before = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                generate_dataset(config, output)
            self.assertEqual(before, (output / "manifest.json").read_bytes())

    def test_loader_rejects_tool_frame_mislabeled_as_global(self):
        config = deepcopy(self.config)
        config["splits"] = {s: int(s == "train") for s in SPLITS}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dataset"
            manifest = generate_dataset(config, output)
            manifest["coordinate_frame"] = "tool_aligned_yz"
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "coordinate frame"):
                LWDVAEDataset(output, "train")

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
