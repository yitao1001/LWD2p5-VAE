"""Continuous, piecewise-constant 2-D geology sampled at cell centers.

Coordinates are a globally aligned local y,z window in ft: axes never rotate.
Background bedding is horizontal; z increases downward and x is invariant.
Tool inclination belongs to the forward model, not this geological generator.
Values are log10(rho / (ohm m)), not conductivity or normalized pixels.
The generator never imports or runs an electromagnetic solver.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

FAMILIES = ("homogeneous", "layered", "curved", "anticline", "faulted",
            "faulted_anticline", "pinchout", "unconformity", "fluid_contact")
SPLITS = ("train", "val", "test", "ood_contrast")
COORDINATE_FRAME = "global_aligned_yz"
GENERATOR_VERSION = "2.2.0"


def _keys(obj, expected, name):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise ValueError(f"{name} must have exactly these keys: {list(expected)}")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _range(values, name, positive=False):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{name} must be a two-element list")
    lo, hi = [_number(v, name) for v in values]
    if lo >= hi or (positive and lo <= 0):
        raise ValueError(f"Invalid range for {name}: {values}")


def validate_config(config):
    """Fail early on misspelled options and physically unresolved geometry."""
    c = deepcopy(config)
    if c.get("schema_version") != 2:
        raise ValueError("Expected dataset config schema_version=2; use configs/lwd_default.json. Legacy generator/config are archived in references/dataset_v1/.")
    _keys(c, ("schema_version", "coordinate_frame", "seed", "grid", "resistivity", "geometry", "family_weights", "splits", "ood", "quality"), "config")
    if type(c["schema_version"]) is not int or c["coordinate_frame"] != COORDINATE_FRAME:
        raise ValueError("Dataset v2 requires coordinate_frame=global_aligned_yz")
    _integer(c["seed"], "seed", 0)
    g, r, geo = c["grid"], c["resistivity"], c["geometry"]
    _keys(g, ("ny", "nz", "y_bounds_ft", "z_bounds_ft"), "grid")
    for key in ("ny", "nz"):
        _integer(g[key], key, 16)
        if g[key] % 8:
            raise ValueError(f"{key} must be divisible by 8 for kl-f8")
    for key in ("y_bounds_ft", "z_bounds_ft"):
        _range(g[key], key)
    _keys(r, ("log10_bounds", "sampling_log10_bounds", "weak_contrast_probability", "weak_delta_log10", "strong_delta_log10"), "resistivity")
    for key in ("log10_bounds", "sampling_log10_bounds", "weak_delta_log10", "strong_delta_log10"):
        _range(r[key], key, positive="delta" in key)
    lo, hi = r["log10_bounds"]
    slo, shi = r["sampling_log10_bounds"]
    if not lo <= slo < shi <= hi:
        raise ValueError("Sampling bounds must lie inside fixed normalization bounds")
    if not 0 <= _number(r["weak_contrast_probability"], "weak_contrast_probability") <= 1:
        raise ValueError("weak_contrast_probability must be in [0,1]")
    if isinstance(c["family_weights"], dict):
        for removed in ("block", "lens"):
            if removed in c["family_weights"]:
                raise ValueError(f"The {removed} family was removed; remove its key and redistribute its weight. See configs/lwd_default.json.")
    _keys(geo, ("background_bedding", "boundary_count", "bed_thickness_ft", "curve_amplitude_ft", "curve_wavelength_ft", "fault_throw_ft", "fault_tilt_from_vertical_deg",
                "anticline_amplitude_ft", "anticline_limb_width_ft", "anticline_center_y_fraction", "anticline_fault_tilt_abs_deg"), "geometry")
    if geo["background_bedding"] != "horizontal":
        raise ValueError("Dataset v2 uses horizontal background bedding; no random overall tilt")
    for key, values in geo.items():
        if key != "background_bedding":
            _range(values, key, positive=key not in ("fault_tilt_from_vertical_deg", "boundary_count", "anticline_center_y_fraction"))
    for v in geo["boundary_count"]:
        _integer(v, "boundary_count", 1)
    dz = (g["z_bounds_ft"][1] - g["z_bounds_ft"][0]) / g["nz"]
    dy = (g["y_bounds_ft"][1] - g["y_bounds_ft"][0]) / g["ny"]
    height = np.ptp(g["z_bounds_ft"])
    width = np.ptp(g["y_bounds_ft"])
    if geo["bed_thickness_ft"][0] < 2 * dz or geo["bed_thickness_ft"][1] > height:
        raise ValueError("Beds must be at least two z cells thick and at most the window height")
    if 0.6 * height / geo["boundary_count"][1] < 2 * dz:
        raise ValueError("Too many boundaries for this grid")
    if geo["anticline_amplitude_ft"][0] < 2 * dz or geo["anticline_amplitude_ft"][1] > height / 2:
        raise ValueError("Anticline amplitude must be at least two z cells and at most half the window height")
    if geo["anticline_limb_width_ft"][0] < 2 * dy or geo["anticline_limb_width_ft"][1] > width / 2:
        raise ValueError("Anticline limb width must be at least two y cells and at most half the window width")
    if max(abs(v) for v in geo["anticline_center_y_fraction"]) > 0.4:
        raise ValueError("Anticline center fractions must lie within [-0.4,0.4] of the window width")
    if geo["anticline_fault_tilt_abs_deg"][1] >= 70:
        raise ValueError("Anticline fault tilt must stay below 70 degrees from vertical")
    _keys(c["family_weights"], FAMILIES, "family_weights")
    weights = [_number(v, "family weight") for v in c["family_weights"].values()]
    if min(weights) < 0 or not np.isclose(sum(weights), 1.0):
        raise ValueError("Family weights must be nonnegative and sum to one")
    _keys(c["splits"], SPLITS, "splits")
    for name, count in c["splits"].items():
        _integer(count, name, 0)
    if c["splits"]["train"] < 1:
        raise ValueError("At least one training model is required")
    if c["splits"]["ood_contrast"] and sum(c["family_weights"][f] for f in FAMILIES if f != "homogeneous") <= 0:
        raise ValueError("OOD splits require a non-homogeneous family")
    _keys(c["ood"], ("delta_log10",), "ood")
    for key, val in c["ood"].items():
        _range(val, key, positive=True)
    if max(abs(v) for v in geo["fault_tilt_from_vertical_deg"]) >= 70:
        raise ValueError("Fault tilt must stay within +/-70 degrees from vertical")
    if c["ood"]["delta_log10"][0] <= max(r["weak_delta_log10"][1], r["strong_delta_log10"][1]):
        raise ValueError("OOD contrast must exceed training contrasts")
    if max(r["weak_delta_log10"][1], r["strong_delta_log10"][1], c["ood"]["delta_log10"][1]) >= shi - slo:
        raise ValueError("Contrast must fit within sampling log10 bounds without clipping")
    _keys(c["quality"], ("min_minority_fraction", "max_attempts"), "quality")
    if not 0 < _number(c["quality"]["min_minority_fraction"], "min_minority_fraction") < 0.5:
        raise ValueError("min_minority_fraction must be in (0,0.5)")
    _integer(c["quality"]["max_attempts"], "max_attempts", 1)
    return c


def load_config(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate config key: {key}")
            obj[key] = value
        return obj
    return validate_config(json.loads(Path(path).read_text(), object_pairs_hook=unique))


def grid_coordinates(config):
    g = config["grid"]
    ye = np.linspace(*g["y_bounds_ft"], g["ny"] + 1)
    ze = np.linspace(*g["z_bounds_ft"], g["nz"] + 1)
    return ye, ze, *np.meshgrid((ye[:-1] + ye[1:]) / 2, (ze[:-1] + ze[1:]) / 2)


def _anticline_uplift(y, center, amplitude, left_width, right_width):
    """Positive upward displacement in ft, shared by all stratigraphic horizons.

    Independent limb widths produce a skewed arch with a smooth crest. Since z
    increases downwards, an undeformed horizon at z0 moves to z0 - uplift(y).
    """
    limb_width = np.where(y < center, left_width, right_width)
    return amplitude * np.exp(-0.5 * ((y - center) / limb_width) ** 2)


def sample_model(config, rng, family, profile="train"):
    """Return (nz,ny) raw log10rho and sampled geological parameters."""
    if family not in FAMILIES or profile not in SPLITS:
        raise ValueError("Unknown geological family or split")
    c, r, geo = config, config["resistivity"], config["geometry"]
    ye, ze, yy, zz = grid_coordinates(c)
    width, height = np.ptp(ye), np.ptp(ze)
    yc, zc = (ye[0] + ye[-1]) / 2, (ze[0] + ze[-1]) / 2
    y, z = yy - yc, zz - zc
    uniform = lambda limits: float(rng.uniform(*limits))
    if profile == "ood_contrast":
        delta = uniform(c["ood"]["delta_log10"])
        contrast_class = "ood"
    elif rng.random() < r["weak_contrast_probability"]:
        delta = uniform(r["weak_delta_log10"])
        contrast_class = "weak"
    else:
        delta = uniform(r["strong_delta_log10"])
        contrast_class = "strong"
    slo, shi = r["sampling_log10_bounds"]
    low = float(rng.uniform(slo, shi - delta))
    high = low + delta
    values = np.array([low, high])
    if rng.random() < 0.5:
        values = values[::-1]
    offset = float(rng.uniform(-0.2, 0.2) * height)
    # 所有家族先从水平背景层序开始。局部弯曲/断层等由下方分支显式定义，
    # 不把已知工具井斜混入地质，也不对整张模型随机旋转。
    strat = z - offset
    meta = {"offset_ft": offset,
            "endmembers_log10rho": values.tolist(), "delta_log10": delta,
            "contrast_class": contrast_class}
    n = int(rng.integers(geo["boundary_count"][0], geo["boundary_count"][1] + 1))
    # Well-separated ordered interfaces, shifted together by a shared deformation.
    bounds = np.array([0.0]) if n == 1 else np.linspace(-0.3 * height, 0.3 * height, n)
    if n > 1:
        bounds += rng.uniform(-0.05, 0.05, n) * height / n
    meta["stratigraphic_boundaries_ft"] = bounds.tolist()

    if family == "homogeneous":
        value = uniform(r["sampling_log10_bounds"])
        meta = {"homogeneous_log10rho": value, "contrast_class": "none"}
        model = np.full(y.shape, value)
    elif family in ("layered", "curved", "faulted"):
        if family == "curved":
            amp, wavelength = uniform(geo["curve_amplitude_ft"]), uniform(geo["curve_wavelength_ft"])
            phase = float(rng.uniform(0, 2 * np.pi))
            strat -= amp * np.sin(2 * np.pi * y / wavelength + phase)
            meta.update(amplitude_ft=amp, wavelength_ft=wavelength, phase_rad=phase)
        elif family == "faulted":
            fy = float(rng.uniform(-0.25, 0.25) * width)
            tilt = uniform(geo["fault_tilt_from_vertical_deg"])
            throw = uniform(geo["fault_throw_ft"]) * rng.choice([-1, 1])
            strat -= throw * (y > fy + z * np.tan(np.deg2rad(tilt)))
            meta.update(fault_y_ft=fy + yc, fault_tilt_deg=tilt, throw_ft=float(throw))
        model = values[np.searchsorted(bounds, strat) % 2]
    elif family in ("anticline", "faulted_anticline"):
        # 整组层界共用一个上拱位移；两翼宽度独立抽样，形成左右不对称的背斜。
        amplitude = uniform(geo["anticline_amplitude_ft"])
        center_y = uniform(geo["anticline_center_y_fraction"]) * width
        left_width = uniform(geo["anticline_limb_width_ft"])
        right_width = uniform(geo["anticline_limb_width_ft"])
        material_y, material_z = y, z
        meta.update(anticline_amplitude_ft=amplitude, anticline_center_y_ft=center_y + yc,
                    anticline_left_width_ft=left_width, anticline_right_width_ft=right_width)
        if family == "faulted_anticline":
            # 在当前全局坐标判断断层侧，再沿断层面反向追踪到变形前坐标。
            # dy = dz*tan(alpha) 保证位移平行于断层；throw 是垂向分量。
            fy = float(rng.uniform(-0.15, 0.15) * width)
            tilt = uniform(geo["anticline_fault_tilt_abs_deg"]) * int(rng.choice([-1, 1]))
            throw = uniform(geo["fault_throw_ft"]) * int(rng.choice([-1, 1]))
            lateral_shift = throw * np.tan(np.deg2rad(tilt))
            displaced_side = y > fy + z * np.tan(np.deg2rad(tilt))
            material_y = y - lateral_shift * displaced_side
            material_z = z - throw * displaced_side
            meta.update(fault_y_ft=fy + yc, fault_tilt_deg=tilt, throw_ft=throw,
                        fault_lateral_shift_ft=float(lateral_shift), fault_reference_z_ft=zc)
        uplift = _anticline_uplift(material_y, center_y, amplitude, left_width, right_width)
        strat = material_z - offset + uplift
        model = values[np.searchsorted(bounds, strat) % 2]
    elif family == "pinchout":
        thickness = uniform(geo["bed_thickness_ft"])
        tip = float(rng.uniform(-0.2, 0.3) * width)
        taper_length = float(rng.uniform(0.25, 0.7) * width)
        direction = int(rng.choice([-1, 1]))
        local_thickness = thickness * np.clip(direction * (tip - y) / taper_length, 0, 1)
        mask = (strat >= -thickness / 2) & (strat < -thickness / 2 + local_thickness) & (local_thickness > 0)
        model = np.where(mask, values[1], values[0])
        meta.update(thickness_ft=thickness, tip_y_ft=tip + yc, taper_length_ft=taper_length, direction=direction)
    elif family == "unconformity":
        # 起伏侵蚀面截断水平老层；这是侵蚀不整合，不再加入角度不整合。
        erosion_amp = uniform(geo["curve_amplitude_ft"])
        erosion = offset + erosion_amp * np.cos(2 * np.pi * y / width)
        model = values[np.searchsorted(bounds, strat) % 2]
        cover = z < erosion
        model[cover] = (low + high) / 2
        meta.update(erosion_amplitude_ft=erosion_amp, cover_log10rho=(low + high) / 2)
    else:  # fluid_contact: flat contact inside a horizontal reservoir.
        thickness = uniform(geo["bed_thickness_ft"])
        contact = offset + float(rng.uniform(-0.2, 0.2)) * thickness
        reservoir = np.abs(strat) < thickness / 2
        model = np.full(y.shape, (low + high) / 2)
        model[reservoir & (z < contact)] = high
        model[reservoir & (z >= contact)] = low
        meta.update(thickness_ft=thickness, contact_z_ft=contact + zc)
    meta.update(coordinate_frame=COORDINATE_FRAME, background_bedding="horizontal",
                overall_rotation_deg=0.0)
    return model.astype(np.float32), meta


def _schedule(config, count, split, rng):
    weights = np.array([config["family_weights"][f] for f in FAMILIES])
    if split.startswith("ood_"):
        weights[0] = 0  # 均匀介质没有物性对比度外推的意义。
    if weights.sum() == 0:
        raise ValueError("OOD splits require a non-homogeneous family")
    exact = weights / weights.sum() * count
    quota = np.floor(exact).astype(int)
    quota[np.argsort(-(exact - quota), kind="stable")[:count - quota.sum()]] += 1
    schedule = np.repeat(np.arange(len(FAMILIES)), quota)
    rng.shuffle(schedule)
    return schedule


def generate_dataset(config, output):
    """Stream to NPY memmaps; publish a complete dataset atomically, never overwrite."""
    c = validate_config(config)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    g = c["grid"]
    size = sum(c["splits"].values()) * g["ny"] * g["nz"] * 4
    if shutil.disk_usage(output.parent).free < 1.15 * size + 10_000_000:
        raise OSError(f"Insufficient disk space; model arrays alone need {size / 2**30:.2f} GiB")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    serialized = json.dumps(c, sort_keys=True, ensure_ascii=False)
    manifest = {"schema_version": 2, "generator_version": GENERATOR_VERSION, "config": c,
                "config_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "units": {"coordinates": "ft", "resistivity": "ohm m"},
                "array_order": "N,1,z,y", "inversion_flatten_order": "F",
                "parameterization": "log10(rho / (ohm m))",
                "normalization": "2*(log10rho-lo)/(hi-lo)-1; fixed bounds in config",
                "strike_invariant_axis": "x", "z_positive": "down",
                "coordinate_frame": COORDINATE_FRAME,
                "geometry_policy": {"background_bedding": "horizontal", "random_overall_rotation": False,
                                    "tool_inclination_included": False,
                                    "isolated_block_bodies": False,
                                    "isolated_lens_bodies": False,
                                    "local_structure": "Continuous curved/anticlinal horizons, inclined fault offsets, tapering and erosion; no standalone block or lens bodies."},
                "anisotropy": False, "forward_responses_included": False,
                "split_policy": "Independent geometry per sample, independent seed streams per split; OOD excluded from fit/tuning.",
                "splits": {}}
    hashes = set()
    try:
        ye, ze, _, _ = grid_coordinates(c)
        np.savez(stage / "grid.npz", y_edges_ft=ye, z_edges_ft=ze)
        for split_id, split in enumerate(SPLITS):
            count = c["splits"][split]
            if not count:
                continue
            folder = stage / split
            folder.mkdir()
            data = np.lib.format.open_memmap(folder / "log10rho.npy", mode="w+", dtype=np.float32,
                                            shape=(count, 1, g["nz"], g["ny"]))
            scheduler = np.random.default_rng(np.random.SeedSequence([c["seed"], split_id, 4294967295]))
            schedule = _schedule(c, count, split, scheduler)
            family_counts, contrast_counts = Counter(), Counter()
            minimum, maximum, square_sum, total_sum = np.inf, -np.inf, 0.0, 0.0
            rejected = 0
            with (folder / "metadata.jsonl").open("w", encoding="utf8") as stream:
                for i, family_idx in enumerate(schedule):
                    family = FAMILIES[family_idx]
                    seed = [c["seed"], split_id, i]
                    rng = np.random.default_rng(np.random.SeedSequence(seed))
                    for attempt in range(c["quality"]["max_attempts"]):
                        model, parameters = sample_model(c, rng, family, split)
                        _, counts = np.unique(model, return_counts=True)
                        minority = float(1 - counts.max() / model.size)
                        if family != "homogeneous" and minority < c["quality"]["min_minority_fraction"]:
                            rejected += 1
                            continue
                        # Cropping or erosion may remove an endmember. OOD must
                        # remain high contrast in the actual stored image.
                        if split == "ood_contrast" and np.ptp(model) < c["ood"]["delta_log10"][0] - 1e-6:
                            rejected += 1
                            continue
                        digest = hashlib.sha256(model.astype("<f4").tobytes()).hexdigest()
                        if digest in hashes:
                            rejected += 1
                            continue
                        break
                    else:
                        raise RuntimeError(f"Could not generate valid {split}/{i}/{family}; revise config")
                    if not np.isfinite(model).all() or model.min() < c["resistivity"]["log10_bounds"][0] - 1e-6 or model.max() > c["resistivity"]["log10_bounds"][1] + 1e-6:
                        raise RuntimeError("Generated values violate normalization contract")
                    hashes.add(digest)
                    data[i, 0] = model
                    family_counts[family] += 1
                    contrast_counts[parameters["contrast_class"]] += 1
                    minimum, maximum = min(minimum, float(model.min())), max(maximum, float(model.max()))
                    total_sum += float(model.sum(dtype=np.float64))
                    square_sum += float(np.square(model.astype(np.float64)).sum())
                    record = {"index": i, "sample_id": f"{split}:{i:08d}", "family": family,
                              "seed_sequence": seed, "accepted_attempt": attempt, "model_sha256": digest,
                              "minority_fraction": minority, "observed_delta_log10": float(np.ptp(model)),
                              "parameters": parameters}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if (i + 1) % 1000 == 0:
                        print(f"{split}: {i+1}/{count}", flush=True)
            data.flush()
            del data
            cells = count * g["nz"] * g["ny"]
            manifest["splits"][split] = {"count": count, "path": f"{split}/log10rho.npy",
                "metadata": f"{split}/metadata.jsonl", "family_counts": dict(family_counts),
                "contrast_counts": dict(contrast_counts), "rejected_draws": rejected,
                "min_log10rho": minimum, "max_log10rho": maximum,
                "mean_log10rho": total_sum / cells,
                "std_log10rho": math.sqrt(max(0, square_sum / cells - (total_sum / cells)**2))}
            print(f"Completed {split}: {count} models; {rejected} rejected draws", flush=True)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        # A pre-existing empty directory must not silently be replaced either.
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest
