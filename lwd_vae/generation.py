"""Compositional 2-D formation models (datasets lwd_v2 onward), sampled at cell centres.

Frame: global y-z, z positive downward, x strike-invariant. The image grid never
rotates. Formation dip is geology (the stratigraphic frame is rotated inside the
fixed grid); tool inclination belongs to the forward model and never enters an
image. Values are log10(rho / (ohm m)). No EM solver is imported or run.

Every cell centre is mapped back through the geological history:
  1. lower layer stack drawn from a per-sample lithology palette;
  2. optional erosion surface with an upper (cover) stack of the same bedding dip;
  3. optional pinchout of one lower-stack layer;
  4. regional dip and optional undulation (folding) of both stacks;
  5. optional faults (one or two); slip is parallel to the fault plane;
  6. optional fluid contact, horizontal in the present-day global frame.
Lengths in sample parameters are ft relative to the window centre.
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

SCHEMA_VERSION = 3
# 组合式地质生成器（lwd_v2 起）。2.0.0 生成 lwd_v2/lwd_v3（两者只差层厚中位数配置）；
# 2.1.0（lwd_v4）去掉钙质结核，其余组件与参数不变。
GENERATOR_VERSION = "2.1.0"
COORDINATE_FRAME = "global_aligned_yz"
SPLITS = ("train", "val", "test", "ood_contrast", "ood_dip")
TAGS = ("planar", "undulation", "fault", "pinchout", "erosion", "fluid_contact",
        "gradational", "trend", "thick_bed")
CONTRAST_CLASSES = ("none", "weak", "moderate", "high", "ood")
DIP_CLASSES = ("gentle", "moderate", "ood")
# Per-cell unit ids: lower-stack layer k is k; the other ranges are offsets.
_UPPER, _WATER = 10_000, 100_000
_STACK_COVER_FT = (-24.0, 24.0)

_SPEC = {
    "grid": ("ny", "nz", "y_bounds_ft", "z_bounds_ft"),
    "resistivity": ("log10_bounds", "sampling_log10_bounds", "palette_size_probabilities",
                    "min_palette_gap_log10", "weak_span_probability", "weak_span_log10",
                    "strong_span_log10", "contrast_class_edges_log10"),
    "stratigraphy": ("median_thickness_ft", "thickness_log_sigma", "thickness_clip_ft", "thick_bed_probability"),
    "properties": ("layer_jitter_log10", "trend_probability", "trend_max_log10",
                   "gradational_probability", "gradational_width_ft"),
    "dip": ("gentle_probability", "gentle_abs_deg", "max_abs_deg"),
    "ood": ("contrast_span_log10", "dip_abs_deg"),
    "quality": ("min_visible_cells", "max_attempts", "position_attempts"),
}
_COMPONENTS = {
    "undulation": ("probability", "count", "wavelength_ft", "amplitude_ft", "max_slope"),
    "fault": ("probability", "second_fault_probability", "tilt_from_vertical_deg", "throw_ft"),
    "pinchout": ("probability", "taper_length_ft", "min_host_thickness_ft", "max_wedge_angle_deg"),
    "erosion": ("probability", "amplitude_ft", "wavelength_ft", "max_slope", "max_relative_tilt_deg"),
    "fluid_contact": ("probability", "min_reservoir_thickness_ft", "drop_log10", "transition_ft"),
}


class RejectedSample(Exception):
    """A draw broke a quality rule; the caller draws again from the same stream."""


# --------------------------------------------------------------------------- config

def _keys(obj, expected, name):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise ValueError(f"{name} must have exactly these keys: {list(expected)}")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _range(values, name, minimum=None, maximum=None, integer=False):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{name} must be a two-element list")
    if integer:
        for v in values:
            _integer(v, name, 0)
    lo, hi = (_number(v, name) for v in values)
    if lo > hi or (lo == hi and not integer):
        raise ValueError(f"Invalid range for {name}: {values}")
    if (minimum is not None and lo < minimum) or (maximum is not None and hi > maximum):
        raise ValueError(f"{name} must lie within [{minimum}, {maximum}]")
    return lo, hi


def _probability(value, name):
    if not 0 <= _number(value, name) <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def validate_config(config):
    """Fail early on misspelled options and geometry the grid cannot resolve."""
    c = deepcopy(config)
    if not isinstance(c, dict):
        raise ValueError("Dataset config must be a JSON object")
    if c.get("schema_version") != SCHEMA_VERSION or type(c.get("schema_version")) is not int:
        raise ValueError("Expected dataset config schema_version=3 (lwd_v2); older family-based configs are retired. "
                         "Use configs/lwd_default.json.")
    _keys(c, ("schema_version", "coordinate_frame", "seed", "grid", "resistivity", "stratigraphy", "properties",
              "dip", "components", "splits", "ood", "quality"), "config")
    if c["coordinate_frame"] != COORDINATE_FRAME:
        raise ValueError("Dataset requires coordinate_frame=global_aligned_yz")
    _integer(c["seed"], "seed", 0)
    for section, keys in _SPEC.items():
        _keys(c[section], keys, section)
    _keys(c["components"], tuple(_COMPONENTS), "components")
    for name, keys in _COMPONENTS.items():
        _keys(c["components"][name], keys, f"components.{name}")
        _probability(c["components"][name]["probability"], f"components.{name}.probability")

    g = c["grid"]
    for key in ("ny", "nz"):
        _integer(g[key], f"grid.{key}", 16)
        if g[key] % 8:
            raise ValueError(f"grid.{key} must be divisible by 8 for kl-f8")
    _range(g["y_bounds_ft"], "grid.y_bounds_ft")
    _range(g["z_bounds_ft"], "grid.z_bounds_ft")
    cell = max(np.ptp(g["y_bounds_ft"]) / g["ny"], np.ptp(g["z_bounds_ft"]) / g["nz"])
    two_cells = 2 * cell - 1e-9

    r = c["resistivity"]
    lo, hi = _range(r["log10_bounds"], "resistivity.log10_bounds")
    slo, shi = _range(r["sampling_log10_bounds"], "resistivity.sampling_log10_bounds", lo, hi)
    sizes = r["palette_size_probabilities"]
    if not isinstance(sizes, dict) or not sizes or any(k not in ("2", "3", "4", "5", "6") for k in sizes):
        raise ValueError("palette_size_probabilities keys must be palette sizes '2'..'6'")
    for key, value in sizes.items():
        _probability(value, f"palette_size_probabilities[{key}]")
    if not np.isclose(sum(sizes.values()), 1.0):
        raise ValueError("palette_size_probabilities must sum to one")
    gap = _number(r["min_palette_gap_log10"], "min_palette_gap_log10")
    _probability(r["weak_span_probability"], "weak_span_probability")
    weak = _range(r["weak_span_log10"], "weak_span_log10", gap)
    strong = _range(r["strong_span_log10"], "strong_span_log10", weak[1])
    edges = _range(r["contrast_class_edges_log10"], "contrast_class_edges_log10", 0, strong[1])
    if gap <= 0 or edges[0] != weak[1]:
        raise ValueError("min_palette_gap_log10 must be positive and the first class edge must equal the weak span maximum")
    ood_span = _range(c["ood"]["contrast_span_log10"], "ood.contrast_span_log10", strong[1])
    if ood_span[1] > shi - slo:
        raise ValueError("OOD contrast must fit inside the sampling bounds")

    st = c["stratigraphy"]
    median = _range(st["median_thickness_ft"], "median_thickness_ft", two_cells)
    clip = _range(st["thickness_clip_ft"], "thickness_clip_ft", two_cells)
    if not clip[0] <= median[0] <= median[1] <= clip[1]:
        raise ValueError("median_thickness_ft must lie inside thickness_clip_ft")
    if _number(st["thickness_log_sigma"], "thickness_log_sigma") < 0:
        raise ValueError("thickness_log_sigma must be >= 0")
    _probability(st["thick_bed_probability"], "thick_bed_probability")

    p = c["properties"]
    for key in ("layer_jitter_log10", "trend_max_log10"):
        if _number(p[key], key) < 0:
            raise ValueError(f"{key} must be >= 0")
    _probability(p["trend_probability"], "trend_probability")
    _probability(p["gradational_probability"], "gradational_probability")
    _range(p["gradational_width_ft"], "gradational_width_ft", two_cells)

    d = c["dip"]
    _probability(d["gentle_probability"], "dip.gentle_probability")
    gentle, maximum = _number(d["gentle_abs_deg"], "gentle_abs_deg"), _number(d["max_abs_deg"], "max_abs_deg")
    ood_dip = _range(c["ood"]["dip_abs_deg"], "ood.dip_abs_deg", maximum, 45)
    if not 0 < gentle <= maximum <= ood_dip[0]:
        raise ValueError("Dip limits must satisfy 0 < gentle_abs_deg <= max_abs_deg <= ood.dip_abs_deg[0]")

    k = c["components"]
    _range(k["undulation"]["count"], "undulation.count", 1, 6, integer=True)
    _range(k["undulation"]["wavelength_ft"], "undulation.wavelength_ft", 4 * cell)
    _range(k["undulation"]["amplitude_ft"], "undulation.amplitude_ft", 0)
    for name in ("undulation", "erosion"):
        if not 0 < _number(k[name]["max_slope"], f"{name}.max_slope") <= 1:
            raise ValueError(f"{name}.max_slope must be in (0, 1]")
    _probability(k["fault"]["second_fault_probability"], "fault.second_fault_probability")
    _range(k["fault"]["tilt_from_vertical_deg"], "fault.tilt_from_vertical_deg", 0, 70)
    _range(k["fault"]["throw_ft"], "fault.throw_ft", two_cells)
    _range(k["pinchout"]["taper_length_ft"], "pinchout.taper_length_ft", two_cells)
    if _number(k["pinchout"]["min_host_thickness_ft"], "pinchout.min_host_thickness_ft") < two_cells:
        raise ValueError("pinchout.min_host_thickness_ft must be at least two cells")
    if not 0 < _number(k["pinchout"]["max_wedge_angle_deg"], "pinchout.max_wedge_angle_deg") < 60:
        raise ValueError("pinchout.max_wedge_angle_deg must be in (0, 60)")
    _range(k["erosion"]["amplitude_ft"], "erosion.amplitude_ft", 0)
    _range(k["erosion"]["wavelength_ft"], "erosion.wavelength_ft", 4 * cell)
    if not 0 <= _number(k["erosion"]["max_relative_tilt_deg"], "erosion.max_relative_tilt_deg") < 30:
        raise ValueError("erosion.max_relative_tilt_deg must be in [0, 30)")
    if _number(k["fluid_contact"]["min_reservoir_thickness_ft"], "fluid_contact.min_reservoir_thickness_ft") < two_cells:
        raise ValueError("fluid_contact.min_reservoir_thickness_ft must be at least two cells")
    _range(k["fluid_contact"]["drop_log10"], "fluid_contact.drop_log10", 0)
    _range(k["fluid_contact"]["transition_ft"], "fluid_contact.transition_ft", 0)

    _keys(c["splits"], SPLITS, "splits")
    for name, count in c["splits"].items():
        _integer(count, f"splits.{name}", 0)
    if c["splits"]["train"] < 1:
        raise ValueError("At least one training model is required")
    q = c["quality"]
    for key in ("min_visible_cells", "max_attempts", "position_attempts"):
        _integer(q[key], f"quality.{key}", 1)
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
    """Cell edges and centre meshes (ft) of the configured window."""
    g = config["grid"]
    ye = np.linspace(*g["y_bounds_ft"], g["ny"] + 1)
    ze = np.linspace(*g["z_bounds_ft"], g["nz"] + 1)
    return ye, ze, *np.meshgrid((ye[:-1] + ye[1:]) / 2, (ze[:-1] + ze[1:]) / 2)


# ------------------------------------------------------------------- geometry parts

def _u(rng, limits):
    return float(rng.uniform(limits[0], limits[1]))


def _log_u(rng, limits):
    return float(math.exp(rng.uniform(math.log(limits[0]), math.log(limits[1]))))


def _sign(rng):
    return 1.0 if rng.random() < 0.5 else -1.0


class _Frame:
    """Rotation into bedding-parallel u and bedding-normal s (s positive down).

    A bedding surface s=const satisfies dz/dy = tan(dip): positive dip deepens toward +y.
    """

    def __init__(self, dip_deg, y0, z0):
        self.cos, self.sin = math.cos(math.radians(dip_deg)), math.sin(math.radians(dip_deg))
        self.y0, self.z0 = y0, z0

    def to_us(self, y, z):
        dy, dz = y - self.y0, z - self.z0
        return dy * self.cos + dz * self.sin, dz * self.cos - dy * self.sin

    def from_us(self, u, s):
        return self.y0 + u * self.cos - s * self.sin, self.z0 + u * self.sin + s * self.cos


class _Stack:
    """Layers between increasing edges in stratigraphic s (ft, downward)."""

    def __init__(self, edges, palette_index, base, trend, anchor, gradational, envelope):
        self.edges, self.palette_index = edges, palette_index
        self.base, self.trend, self.anchor = base, trend, anchor
        self.gradational, self.envelope = gradational, envelope
        self.thickness = np.diff(edges)

    def layer(self, s):
        return np.clip(np.searchsorted(self.edges, s, side="right") - 1, 0, len(self.base) - 1)

    def value(self, s, k):
        top, bottom = self.edges[k], self.edges[k + 1]
        fraction = np.clip((s - top) / (bottom - top), 0.0, 1.0)
        return np.clip(self.base[k] + self.trend[k] * (fraction - self.anchor[k]), *self.envelope)

    def evaluate(self, s):
        k = self.layer(s)
        values = self.base[k] + self.trend[k] * (
            np.clip((s - self.edges[k]) / self.thickness[k], 0.0, 1.0) - self.anchor[k])
        blended = np.zeros(s.shape, dtype=bool)
        for j in np.flatnonzero(self.gradational):
            width = self.gradational[j]
            near = np.abs(s - self.edges[j]) < width / 2
            if near.any():
                t = (s[near] - self.edges[j] + width / 2) / width
                above = self.base[j - 1] + self.trend[j - 1] * (1.0 - self.anchor[j - 1])
                below = self.base[j] + self.trend[j] * (0.0 - self.anchor[j])
                values[near] = above + (below - above) * t
                blended |= near
        return np.clip(values, *self.envelope), k, blended


def _palette(rng, c, span):
    """Sorted lithology values; extremes span exactly `span`, members >= gap apart."""
    r = c["resistivity"]
    slo, shi = r["sampling_log10_bounds"]
    gap = r["min_palette_gap_log10"]
    sizes = sorted(r["palette_size_probabilities"], key=int)
    size = int(rng.choice([int(k) for k in sizes], p=[r["palette_size_probabilities"][k] for k in sizes]))
    size = max(2, min(size, 1 + int(span / gap + 1e-9)))
    low = float(rng.uniform(slo, shi - span))
    for _ in range(50):
        values = np.concatenate(([low], np.sort(rng.uniform(low, low + span, size - 2)), [low + span]))
        if np.all(np.diff(values) >= gap):
            return values
    return np.linspace(low, low + span, size)


def _build_stack(rng, c, palette, median):
    """Lognormal bed thicknesses around a per-sample median; adjacent beds differ in lithology.

    Jitter and trends of the two extreme lithologies point inward, so every
    layer stays inside the palette envelope and the palette span bounds contrast.
    """
    st, pr = c["stratigraphy"], c["properties"]
    lo_clip, hi_clip = st["thickness_clip_ft"]
    edges = [_STACK_COVER_FT[0] - float(rng.uniform(0, 1)) * median]
    while edges[-1] < _STACK_COVER_FT[1]:
        thickness = median * math.exp(st["thickness_log_sigma"] * rng.standard_normal())
        edges.append(edges[-1] + float(np.clip(thickness, lo_clip, hi_clip)))
    edges = np.asarray(edges)
    n, size = len(edges) - 1, len(palette)
    span = float(palette[-1] - palette[0])
    index = np.empty(n, dtype=int)
    index[0] = rng.integers(size)
    for k in range(1, n):
        draw = int(rng.integers(size - 1))
        index[k] = draw + (draw >= index[k - 1])
    jitter_amp = min(pr["layer_jitter_log10"], 0.2 * span)
    trend_amp = min(pr["trend_max_log10"], 0.5 * span)
    base, trend, anchor = np.empty(n), np.zeros(n), np.full(n, 0.5)
    for k in range(n):
        p = index[k]
        jitter = float(rng.uniform(-jitter_amp, jitter_amp))
        jitter = -abs(jitter) if p == size - 1 else abs(jitter) if p == 0 else jitter
        base[k] = palette[p] + jitter
        if rng.random() < pr["trend_probability"]:
            g = float(rng.uniform(-trend_amp, trend_amp))
            trend[k] = g
            if p == size - 1:
                anchor[k] = 1.0 if g > 0 else 0.0
            elif p == 0:
                anchor[k] = 0.0 if g > 0 else 1.0
    gradational = np.zeros(n + 1)
    for j in range(1, n):
        if rng.random() < pr["gradational_probability"]:
            width = _u(rng, pr["gradational_width_ft"])
            if width <= 0.5 * min(edges[j] - edges[j - 1], edges[j + 1] - edges[j]):
                gradational[j] = width
    return _Stack(edges, index, base, trend, anchor, gradational, (float(palette[0]), float(palette[-1])))


def _undulation(rng, cfg, u_range):
    """Gaussian arches/sags (centres may lie outside the window) and sine folds; slope-capped."""
    count = int(rng.integers(cfg["count"][0], cfg["count"][1] + 1))
    terms = []
    for _ in range(count):
        wavelength = _u(rng, cfg["wavelength_ft"])
        amplitude = _u(rng, cfg["amplitude_ft"]) / math.sqrt(count)
        sign = _sign(rng)
        if rng.random() < 0.5:
            width = wavelength / 4
            amplitude = min(amplitude, cfg["max_slope"] * width * math.sqrt(math.e))
            centre = float(rng.uniform(u_range[0] - 2 * width, u_range[1] + 2 * width))
            terms.append({"kind": "gaussian", "amplitude_ft": sign * amplitude, "centre_u_ft": centre, "width_ft": width})
        else:
            amplitude = min(amplitude, cfg["max_slope"] * wavelength / (2 * math.pi))
            terms.append({"kind": "sine", "amplitude_ft": sign * amplitude, "wavelength_ft": wavelength,
                          "phase_rad": float(rng.uniform(0, 2 * math.pi))})
    return terms


def _uplift(terms, u):
    """Upward displacement (ft) of stratigraphic surfaces at bedding coordinate u."""
    lift = np.zeros(np.shape(u))
    for t in terms:
        if t["kind"] == "gaussian":
            lift = lift + t["amplitude_ft"] * np.exp(-0.5 * ((u - t["centre_u_ft"]) / t["width_ft"]) ** 2)
        else:
            lift = lift + t["amplitude_ft"] * np.sin(2 * np.pi * u / t["wavelength_ft"] + t["phase_rad"])
    return lift


def _fault_side(fault, y, z):
    slope = math.tan(math.radians(fault["tilt_from_vertical_deg"]))
    return y > fault["anchor_y_ft"] + (z - fault["anchor_z_ft"]) * slope


def _move(fault, y, z, sign):
    """sign=-1 undoes the slip (present -> material), +1 applies it. The side test
    is identical in both frames because the slip vector is parallel to the plane."""
    side = _fault_side(fault, y, z)
    slope = math.tan(math.radians(fault["tilt_from_vertical_deg"]))
    return y + sign * side * fault["throw_ft"] * slope, z + sign * side * fault["throw_ft"]


def _erosion_surface(erosion, u):
    return (erosion["level_s_ft"] + erosion["amplitude_ft"] * np.sin(2 * np.pi * u / erosion["wavelength_ft"] + erosion["phase_rad"])
            + u * math.tan(math.radians(erosion["relative_tilt_deg"])))


def _pinchout_interval(p, stack, u):
    k = p["host_layer"]
    top, bottom = stack.edges[k], stack.edges[k + 1]
    local = (bottom - top) * np.clip(p["direction"] * (p["tip_u_ft"] - u) / p["taper_length_ft"], 0.0, 1.0)
    if p["mode"] == "top_descends":
        return bottom - local, bottom
    if p["mode"] == "base_rises":
        return top, top + local
    middle = (top + bottom) / 2
    return middle - local / 2, middle + local / 2



class _Scene:
    """All geological parameters of one sample, evaluated on present-day points."""

    def __init__(self, frame, lower, palette):
        self.frame, self.lower, self.upper, self.palette = frame, lower, None, palette
        self.undulation, self.faults, self.erosion, self.pinchout = [], [], None, None
        self.fluid = None

    def material(self, y, z, skip_fault=None):
        for i in range(len(self.faults) - 1, -1, -1):
            if i != skip_fault:
                y, z = _move(self.faults[i], y, z, -1)
        u, s = self.frame.to_us(y, z)
        return u, s + _uplift(self.undulation, u)

    def present(self, u, s_strat):
        """Forward map of material points (used for pinchout tips)."""
        y, z = self.frame.from_us(u, s_strat - _uplift(self.undulation, u))
        for fault in self.faults:
            y, z = _move(fault, y, z, +1)
        return y, z

    def eroded(self, u, s_strat):
        return np.zeros(np.shape(u), bool) if self.erosion is None else s_strat < _erosion_surface(self.erosion, u)

    def evaluate(self, y, z, skip=(), skip_fault=None):
        u, s = self.material(y, z, skip_fault)
        values, unit, blended = self.lower.evaluate(s)
        if self.pinchout is not None and "pinchout" not in skip:
            k = self.pinchout["host_layer"]
            low, high = _pinchout_interval(self.pinchout, self.lower, u)
            host = unit == k
            # 被尖灭让出的空间由相邻层填充（顶面下降→上覆层，底面上升→下伏层）。
            for vacated, neighbour in ((host & (s < low), k - 1), (host & (s >= high), k + 1)):
                if vacated.any():
                    values[vacated] = self.lower.value(s[vacated], np.full(vacated.sum(), neighbour))
                    unit[vacated] = neighbour
                    blended[vacated] = False
        if self.erosion is not None and "erosion" not in skip:
            cover = self.eroded(u, s)
            if cover.any():
                v2, k2, b2 = self.upper.evaluate(s[cover])
                values[cover], unit[cover], blended[cover] = v2, _UPPER + k2, b2
        if self.fluid is not None and "fluid_contact" not in skip:
            values, unit = self._apply_fluid(values, unit, z)
        return values, unit, blended

    def palette_index(self, unit):
        index = np.full(unit.shape, -1)
        lower = unit < _UPPER
        index[lower] = self.lower.palette_index[unit[lower]]
        if self.upper is not None:
            upper = (unit >= _UPPER) & (unit < _WATER)
            index[upper] = self.upper.palette_index[unit[upper] - _UPPER]
        return index

    def layer_thickness(self, unit):
        thickness = np.zeros(unit.shape)
        lower = unit < _UPPER
        thickness[lower] = self.lower.thickness[unit[lower]]
        if self.upper is not None:
            upper = (unit >= _UPPER) & (unit < _WATER)
            thickness[upper] = self.upper.thickness[unit[upper] - _UPPER]
        return thickness

    def _ramp(self, z):
        f = self.fluid
        if f["transition_ft"] > 0:
            return np.clip((z - (f["contact_z_ft"] - f["transition_ft"])) / f["transition_ft"], 0.0, 1.0)
        return (z >= f["contact_z_ft"]).astype(float)

    def _apply_fluid(self, values, unit, z):
        reservoir = self.palette_index(unit) == self.fluid["palette_index"]
        ramp = self._ramp(z)
        wet = reservoir & (ramp > 0)
        values = values.copy()
        values[wet] -= self.fluid["drop_log10"] * ramp[wet]
        unit = unit.copy()
        unit[wet] += _WATER
        return values, unit


# ---------------------------------------------------------------------- sampling

def _inside(window, y, z):
    (ya, yb), (za, zb) = window
    return (y >= ya) & (y <= yb) & (z >= za) & (z <= zb)


def _visible_change(scene, y, z, reference, minimum, **skip):
    other, _, _ = scene.evaluate(y, z, **skip)
    return int(np.count_nonzero(np.abs(other - reference) > 1e-9)) >= minimum


def _sample_erosion(rng, c, scene, y, z, window, s_centre, height, minimum):
    cfg = c["components"]["erosion"]
    for _ in range(c["quality"]["position_attempts"]):
        wavelength = _u(rng, cfg["wavelength_ft"])
        amplitude = min(_u(rng, cfg["amplitude_ft"]), cfg["max_slope"] * wavelength / (2 * math.pi))
        scene.erosion = {"level_s_ft": s_centre + float(rng.uniform(-0.3, 0.3)) * height,
                         "amplitude_ft": _sign(rng) * amplitude, "wavelength_ft": wavelength,
                         "phase_rad": float(rng.uniform(0, 2 * math.pi)),
                         "relative_tilt_deg": float(rng.uniform(-cfg["max_relative_tilt_deg"], cfg["max_relative_tilt_deg"]))}
        values, _, _ = scene.evaluate(y, z)
        u, s = scene.material(y, z)
        cover = scene.eroded(u, s)
        if (min(cover.sum(), (~cover).sum()) >= minimum
                and _visible_change(scene, y, z, values, minimum, skip=("erosion",))):
            return True
    scene.erosion = None
    return False


def _window_layers(scene, y, z):
    _, unit, _ = scene.evaluate(y, z, skip=("pinchout", "erosion", "fluid_contact"))
    return set(np.unique(unit[unit < _UPPER]).tolist())


def _sample_pinchout(rng, c, scene, y, z, window, u_range, cell, minimum):
    cfg = c["components"]["pinchout"]
    n = len(scene.lower.base)
    slope = math.tan(math.radians(cfg["max_wedge_angle_deg"]))
    for _ in range(c["quality"]["position_attempts"]):
        visible = _window_layers(scene, y, z)
        mode = ("top_descends", "base_rises", "symmetric")[int(rng.integers(3))]
        moving = 0.5 if mode == "symmetric" else 1.0  # share of the thickness carried by one surface
        # 楔形角上限：移动的层面相对层理的斜率 <= tan(max_wedge_angle)，即楔形长度 >= 厚度/斜率。
        hosts = [k for k in sorted(visible) if 1 <= k <= n - 2
                 and scene.lower.thickness[k] >= cfg["min_host_thickness_ft"]
                 and moving * scene.lower.thickness[k] / slope <= cfg["taper_length_ft"][1]]
        if not hosts:
            continue
        k = int(rng.choice(hosts))
        shortest = max(cfg["taper_length_ft"][0], moving * scene.lower.thickness[k] / slope)
        taper = _log_u(rng, (shortest, cfg["taper_length_ft"][1])) if shortest < cfg["taper_length_ft"][1] else shortest
        direction = int(_sign(rng))
        # The taper zone must overlap the window's bedding-parallel extent; the tip may lie outside.
        tip = float(rng.uniform(u_range[0], u_range[1] + taper) if direction > 0 else rng.uniform(u_range[0] - taper, u_range[1]))
        scene.pinchout = {"host_layer": k, "host_thickness_ft": float(scene.lower.thickness[k]), "mode": mode,
                          "direction": direction, "tip_u_ft": tip, "taper_length_ft": taper,
                          "wedge_angle_deg": math.degrees(math.atan(moving * scene.lower.thickness[k] / taper))}
        values, _, _ = scene.evaluate(y, z)
        # A pinchout must change a visible area (three times the generic minimum), not a corner sliver.
        if _visible_change(scene, y, z, values, 3 * minimum, skip=("pinchout",)):
            top, bottom = scene.lower.edges[k], scene.lower.edges[k + 1]
            tip_s = {"top_descends": bottom, "base_rises": top, "symmetric": (top + bottom) / 2}[mode]
            ty, tz = scene.present(np.array([tip]), np.array([tip_s]))
            scene.pinchout["tip_in_window"] = bool(_inside(window, ty, tz)[0] and not scene.eroded(np.array([tip]), np.array([tip_s]))[0])
            # Neighbours of the host keep sharp contacts so the taper stays well defined.
            scene.lower.gradational[[k, k + 1]] = 0.0
            return True
    scene.pinchout = None
    return False


def _sample_faults(rng, c, scene, y, z, width, height, minimum):
    cfg = c["components"]["fault"]
    count = 2 if rng.random() < cfg["second_fault_probability"] else 1
    for _ in range(count):
        for _ in range(c["quality"]["position_attempts"]):
            fault = {"tilt_from_vertical_deg": _u(rng, cfg["tilt_from_vertical_deg"]) * _sign(rng),
                     "throw_ft": _u(rng, cfg["throw_ft"]) * _sign(rng),
                     "anchor_y_ft": float(rng.uniform(-0.35, 0.35)) * width,
                     "anchor_z_ft": float(rng.uniform(-0.3, 0.3)) * height}
            # The moved side is the hanging wall when the plane leans toward +y.
            fault["sense"] = "normal" if (fault["tilt_from_vertical_deg"] > 0) == (fault["throw_ft"] > 0) else "reverse"
            scene.faults.append(fault)
            values, _, _ = scene.evaluate(y, z)
            if _visible_change(scene, y, z, values, minimum, skip_fault=len(scene.faults) - 1):
                break
            scene.faults.pop()
    return bool(scene.faults)




def _sample_fluid(rng, c, scene, y, z, cell, limit, minimum):
    cfg = c["components"]["fluid_contact"]
    slo, _ = c["resistivity"]["sampling_log10_bounds"]
    values, unit, _ = scene.evaluate(y, z)
    pindex = scene.palette_index(unit)
    thick = scene.layer_thickness(unit)
    median_value = float(np.median(scene.palette))
    candidates = [p for p in range(len(scene.palette))
                  if scene.palette[p] >= median_value
                  and np.count_nonzero((pindex == p) & (thick >= cfg["min_reservoir_thickness_ft"])) >= 2 * minimum]
    if not candidates:
        return False
    p = int(rng.choice(candidates))
    reservoir_cells = (pindex == p) & (thick >= cfg["min_reservoir_thickness_ft"])
    z_lo, z_hi = np.percentile(z[reservoir_cells], [10, 90])
    reservoir = pindex == p
    for _ in range(c["quality"]["position_attempts"]):
        contact = float(rng.uniform(z_lo, z_hi))
        transition = _u(rng, cfg["transition_ft"])
        water = reservoir & (z >= contact)
        oil = reservoir & (z < contact - transition)
        if min(water.sum(), oil.sum()) < minimum:
            continue
        floor = max(slo, float(values.max()) - limit)
        wet = reservoir & (z > contact - transition)
        top = min(cfg["drop_log10"][1], float(values[wet].min()) - floor)
        if top < cfg["drop_log10"][0]:
            return False
        scene.fluid = {"palette_index": p, "contact_z_ft": contact, "transition_ft": transition,
                       "drop_log10": float(rng.uniform(cfg["drop_log10"][0], top))}
        return True
    return False


def _contrast_class(c, span, interfaces):
    if not interfaces:
        return "none"
    weak, moderate = c["resistivity"]["contrast_class_edges_log10"]
    if span > c["resistivity"]["strong_span_log10"][1] + 1e-9:
        return "ood"
    return "weak" if span < weak else "moderate" if span < moderate else "high"


def _dip_class(c, dip):
    magnitude = abs(dip)
    if magnitude > c["dip"]["max_abs_deg"] + 1e-9:
        return "ood"
    return "gentle" if magnitude <= c["dip"]["gentle_abs_deg"] else "moderate"


def sample_model(config, rng, split="train"):
    """Return ((nz,ny) float32 log10rho, info) or raise RejectedSample.

    info holds the evaluation labels (tags, contrast_class, dip_class,
    span_log10) and every sampled geological parameter.
    """
    image, info, _, _ = _draw(config, rng, split)
    return image, info


def _draw(config, rng, split):
    """sample_model plus the internal scene and centred cell coordinates (for tests)."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    c, comp = config, config["components"]
    ye, ze, yy, zz = grid_coordinates(c)
    yc, zc = (ye[0] + ye[-1]) / 2, (ze[0] + ze[-1]) / 2
    y, z = yy - yc, zz - zc
    width, height = float(np.ptp(ye)), float(np.ptp(ze))
    window = ((-width / 2, width / 2), (-height / 2, height / 2))
    cell = max(width / c["grid"]["ny"], height / c["grid"]["nz"])
    minimum = c["quality"]["min_visible_cells"]
    slo, shi = c["resistivity"]["sampling_log10_bounds"]

    if split == "ood_dip":
        dip = _u(rng, c["ood"]["dip_abs_deg"]) * _sign(rng)
    else:
        limit_deg = c["dip"]["gentle_abs_deg"] if rng.random() < c["dip"]["gentle_probability"] else c["dip"]["max_abs_deg"]
        dip = float(rng.uniform(-limit_deg, limit_deg))
    frame = _Frame(dip, float(rng.uniform(-0.25, 0.25)) * width, float(rng.uniform(-0.25, 0.25)) * height)
    corner_u, corner_s = frame.to_us(np.array([window[0][0], window[0][1], window[0][0], window[0][1]]),
                                     np.array([window[1][0], window[1][0], window[1][1], window[1][1]]))
    u_range = (float(corner_u.min()), float(corner_u.max()))
    parameters = {"coordinates": "ft relative to the window centre; z positive down; u along bedding, s across bedding",
                  "dip_deg": dip, "dip_pivot_yz_ft": [frame.y0, frame.z0]}

    if split not in ("ood_contrast", "ood_dip") and rng.random() < c["stratigraphy"]["thick_bed_probability"]:
        # 厚层：窗口内没有层界，只保留轻微的沿层面法向趋势。
        trend_max = c["properties"]["trend_max_log10"]
        value = float(rng.uniform(slo + trend_max, shi - trend_max))
        trend = float(rng.uniform(0.1 * trend_max, trend_max)) * _sign(rng)
        _, s = frame.to_us(y, z)
        fraction = (s - corner_s.min()) / np.ptp(corner_s)
        image = value + trend * (fraction - 0.5)
        parameters.update(thick_bed_log10rho=value, trend_log10=trend)
        span = float(np.ptp(image))
        return image.astype(np.float32), {"tags": ["thick_bed"], "contrast_class": "none",
                                          "dip_class": _dip_class(c, dip), "span_log10": span,
                                          "parameters": parameters}, None, (y, z)

    if split == "ood_contrast":
        palette_span = _u(rng, c["ood"]["contrast_span_log10"])
        limit = c["ood"]["contrast_span_log10"][1]
    else:
        r = c["resistivity"]
        palette_span = _u(rng, r["weak_span_log10"] if rng.random() < r["weak_span_probability"] else r["strong_span_log10"])
        limit = r["strong_span_log10"][1]
    palette = _palette(rng, c, palette_span)
    median = _log_u(rng, c["stratigraphy"]["median_thickness_ft"])
    scene = _Scene(frame, _build_stack(rng, c, palette, median), palette)
    parameters.update(palette_log10rho=palette.tolist(), palette_span_log10=palette_span, median_thickness_ft=median)

    if rng.random() < comp["undulation"]["probability"]:
        scene.undulation = _undulation(rng, comp["undulation"], u_range)
    if rng.random() < comp["erosion"]["probability"]:
        scene.upper = _build_stack(rng, c, palette, median)
        _, s_centre = frame.to_us(np.array(0.0), np.array(0.0))
        if not _sample_erosion(rng, c, scene, y, z, window, float(s_centre), height, minimum):
            scene.upper = None
    if rng.random() < comp["pinchout"]["probability"]:
        _sample_pinchout(rng, c, scene, y, z, window, u_range, cell, minimum)
    if rng.random() < comp["fault"]["probability"]:
        _sample_faults(rng, c, scene, y, z, width, height, minimum)
    if rng.random() < comp["fluid_contact"]["probability"]:
        _sample_fluid(rng, c, scene, y, z, cell, limit, minimum)

    values, unit, blended = scene.evaluate(y, z)
    if len(np.unique(unit)) < 2:
        raise RejectedSample("no interface inside the window")
    span = float(np.ptp(values))
    if span > limit + 1e-9 or values.min() < slo - 1e-9 or values.max() > shi + 1e-9:
        raise RuntimeError("Generated values violate the configured contrast/sampling bounds")
    if split == "ood_contrast" and span < c["ood"]["contrast_span_log10"][0]:
        raise RejectedSample("OOD contrast is not visible in the window")

    stack_layers = sorted(set(unit[unit < _WATER].tolist()) | set((unit[unit >= _WATER] - _WATER).tolist()))
    tags = []
    if scene.undulation:
        tags.append("undulation")
    if scene.faults:
        tags.append("fault")
    if scene.pinchout is not None:
        tags.append("pinchout")
    if scene.erosion is not None:
        tags.append("erosion")
    if not tags:
        tags.append("planar")
    if scene.fluid is not None:
        tags.append("fluid_contact")
    if blended.any():
        tags.append("gradational")
    trends = [scene.lower.trend[k] if k < _UPPER else scene.upper.trend[k - _UPPER] for k in stack_layers]
    if any(abs(t) > 0 for t in trends):
        tags.append("trend")
    tags = [t for t in TAGS if t in tags]

    def stack_info(k):
        stack, local = (scene.lower, k) if k < _UPPER else (scene.upper, k - _UPPER)
        return {"stack": "lower" if k < _UPPER else "upper", "thickness_ft": float(stack.thickness[local]),
                "palette_index": int(stack.palette_index[local]), "trend_log10": float(stack.trend[local])}

    parameters.update(
        visible_layers=[stack_info(k) for k in stack_layers],
        undulation=scene.undulation or None,
        faults=scene.faults or None,
        erosion=scene.erosion,
        pinchout=scene.pinchout,
        fluid_contact=scene.fluid,
        gradational_cells=int(blended.sum()),
    )
    info = {"tags": tags, "contrast_class": _contrast_class(c, span, True), "dip_class": _dip_class(c, dip),
            "span_log10": span, "parameters": parameters}
    return values.astype(np.float32), info, scene, (y, z)


def replay_sample(config, record, split):
    """Re-draw an accepted sample from its manifest record (seed and attempt index)."""
    rng = np.random.default_rng(np.random.SeedSequence(record["seed_sequence"]))
    result = None
    for _ in range(record["accepted_attempt"] + 1):
        try:
            result = sample_model(config, rng, split)
        except RejectedSample:
            result = None
    if result is None:
        raise ValueError("Replay did not reproduce an accepted sample")
    return result


# ------------------------------------------------------------------------- output

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
    cell_y, cell_z = np.ptp(g["y_bounds_ft"]) / g["ny"], np.ptp(g["z_bounds_ft"]) / g["nz"]
    manifest = {"schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION, "config": c,
                "config_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "units": {"coordinates": "ft", "resistivity": "ohm m"},
                "array_order": "N,1,z,y", "inversion_flatten_order": "F",
                "parameterization": "log10(rho / (ohm m))",
                "normalization": "2*(log10rho-lo)/(hi-lo)-1; fixed bounds in config",
                "strike_invariant_axis": "x", "z_positive": "down",
                "coordinate_frame": COORDINATE_FRAME,
                "grid_contract": {"shape_nz_ny": [g["nz"], g["ny"]], "cell_ft_y_z": [cell_y, cell_z],
                                  "origin": "window centre", "flatten": "Fortran order, z fastest: index = iy*nz + iz"},
                "geometry_policy": {"image_rotation": False, "formation_dip_in_image": True,
                                    "tool_inclination_included": False,
                                    "fluid_contacts": "horizontal in the global frame",
                                    "isolated_bodies": False,
                                    "angular_unconformity": False},
                "anisotropy": False, "forward_responses_included": False,
                "split_policy": "Independent geometry per sample and per-split seed streams; OOD splits are excluded from fitting and tuning.",
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
            tag_counts, contrast_counts, dip_counts = Counter(), Counter(), Counter()
            minimum, maximum, square_sum, total_sum, rejected = np.inf, -np.inf, 0.0, 0.0, 0
            with (folder / "metadata.jsonl").open("w", encoding="utf8") as stream:
                for i in range(count):
                    seed = [c["seed"], split_id, i]
                    rng = np.random.default_rng(np.random.SeedSequence(seed))
                    for attempt in range(c["quality"]["max_attempts"]):
                        try:
                            model, info = sample_model(c, rng, split)
                        except RejectedSample:
                            rejected += 1
                            continue
                        digest = hashlib.sha256(model.astype("<f4").tobytes()).hexdigest()
                        if digest in hashes:
                            rejected += 1
                            continue
                        break
                    else:
                        raise RuntimeError(f"Could not generate a valid {split}/{i}; revise config")
                    lo, hi = c["resistivity"]["log10_bounds"]
                    if not np.isfinite(model).all() or model.min() < lo - 1e-6 or model.max() > hi + 1e-6:
                        raise RuntimeError("Generated values violate normalization contract")
                    hashes.add(digest)
                    data[i, 0] = model
                    tag_counts.update(info["tags"])
                    contrast_counts[info["contrast_class"]] += 1
                    dip_counts[info["dip_class"]] += 1
                    minimum, maximum = min(minimum, float(model.min())), max(maximum, float(model.max()))
                    total_sum += float(model.sum(dtype=np.float64))
                    square_sum += float(np.square(model.astype(np.float64)).sum())
                    record = {"index": i, "sample_id": f"{split}:{i:08d}", "seed_sequence": seed,
                              "accepted_attempt": attempt, "model_sha256": digest, **info}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if (i + 1) % 2000 == 0:
                        print(f"{split}: {i+1}/{count}", flush=True)
            data.flush()
            del data
            cells = count * g["nz"] * g["ny"]
            manifest["splits"][split] = {
                "count": count, "path": f"{split}/log10rho.npy", "metadata": f"{split}/metadata.jsonl",
                "tag_counts": {t: tag_counts[t] for t in TAGS if tag_counts[t]},
                "contrast_counts": {k: contrast_counts[k] for k in CONTRAST_CLASSES if contrast_counts[k]},
                "dip_counts": {k: dip_counts[k] for k in DIP_CLASSES if dip_counts[k]},
                "rejected_draws": rejected, "min_log10rho": minimum, "max_log10rho": maximum,
                "mean_log10rho": total_sum / cells,
                "std_log10rho": math.sqrt(max(0, square_sum / cells - (total_sum / cells) ** 2))}
            print(f"Completed {split}: {count} models; {rejected} rejected draws", flush=True)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest
