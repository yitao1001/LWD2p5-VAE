#!/usr/bin/env python3
"""Chinese HTML report on how datasets/lwd_v4 is generated, written for geological review.

Explains the generation method in physical units, lists every parameter with its source
(literature / engineering choice / grid), and shows typical (unfiltered) models and logs.
Reads datasets/lwd_v4 and calls the generator only for illustrations; no training, no EM solve.
  .venv/bin/python reports/lwd_v4_generation_method_20261112.py [--fragment PATH]
--fragment also writes the page without the document skeleton (for publishing as an Artifact).
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "reports")]
from report_common import CMAP, bar_cell, esc, plt, png, svg_histogram, svg_window  # noqa: E402
from lwd_vae.generation import RejectedSample, _draw, _erosion_surface  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

DATA = ROOT / "datasets/lwd_v4"
OUT = ROOT / "reports/lwd_v4_generation_method_20261112.html"
FT = 0.3048
RHO_TICKS = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
LOG_COLOR, MARK_COLOR = "#2a78d6", "#eb6834"
TITLE = "lwd_v4 地层模型生成方法"
STRUCT = ("undulation", "fault", "pinchout", "erosion", "fluid_contact")
CN = {"planar": "平面层序", "undulation": "褶曲起伏", "fault": "断层", "pinchout": "尖灭", "erosion": "侵蚀面",
      "fluid_contact": "流体界面", "gradational": "渐变界面", "trend": "层内趋势", "thick_bed": "厚层"}
MODE = {"top_descends": "顶面下降", "base_rises": "底面上升", "symmetric": "上下对称"}


# ------------------------------------------------------------------ formatting

def length(lo, hi=None):
    """'a–b ft（x–y m/cm）'."""
    values = [lo] if hi is None else [lo, hi]
    metres = [v * FT for v in values]
    si = ("–".join(f"{v * 100:.0f}" for v in metres) + " cm") if max(metres) < 1 else (
        "–".join(f"{v:.2f}" for v in metres) + " m")
    return "–".join(f"{v:g}" for v in values) + f" ft（{si}）"


def ratio(decades):
    r = 10 ** decades
    return f"{r:.2f}" if r < 1.95 else f"{r:.1f}" if r < 9.95 else f"{r:.0f}"


def pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------------ draws

def variant(config, components=(), gradational=0.0, trend=0.0, thick=0.0):
    """Config with only `components` switched on (probability 1) and the given detail probabilities."""
    c = copy.deepcopy(config)
    for name in STRUCT:
        c["components"][name]["probability"] = 1.0 if name in components else 0.0
    c["properties"].update(gradational_probability=gradational, trend_probability=trend)
    c["stratigraphy"]["thick_bed_probability"] = thick
    return c


# Scene attributes that remove one structure.
OFF = {"pinchout": {"pinchout": None}, "erosion": {"erosion": None}, "undulation": {"undulation": []},
       "fault": {"faults": []}, "fluid_contact": {"fluid": None}}


def changed(draw, component):
    """Cells whose value changes when `component` is removed from the drawn scene."""
    image, _, scene, (y, z) = draw
    step = copy.copy(scene)
    for key, value in OFF[component].items():
        setattr(step, key, value)
    other, _, _ = step.evaluate(y, z)
    return int(np.count_nonzero(np.abs(other - image) > 1e-5))


def redraw(config, record, split="train"):
    """Replay a stored sample with its scene (same stream and attempt count as generation)."""
    rng = np.random.default_rng(np.random.SeedSequence(record["seed_sequence"]))
    result = None
    for _ in range(record["accepted_attempt"] + 1):
        try:
            result = _draw(config, rng, split)
        except RejectedSample:
            result = None
    return result


def neighbour_contrast(draw):
    """Contrast between a pinchout host layer and the layer(s) that fill its vacated space."""
    scene, p = draw[2], draw[1]["parameters"]["pinchout"]
    k, base = p["host_layer"], scene.lower.base
    sides = {"top_descends": (k - 1,), "base_rises": (k + 1,), "symmetric": (k - 1, k + 1)}[p["mode"]]
    return min(abs(base[k] - base[j]) for j in sides)


def central_tip(draw):
    """Pinchout tip lies in the central part of the window (for clearer illustrations)."""
    scene, p = draw[2], draw[1]["parameters"]["pinchout"]
    k = p["host_layer"]
    top, bottom = scene.lower.edges[k], scene.lower.edges[k + 1]
    s = {"top_descends": bottom, "base_rises": top, "symmetric": (top + bottom) / 2}[p["mode"]]
    y, z = scene.present(np.array([p["tip_u_ft"]]), np.array([s]))
    return abs(float(y[0])) < 3.5 and abs(float(z[0])) < 3.5 and p["wedge_angle_deg"] >= 8


def pick(config, predicates, seed, limit=5000):
    """One accepted draw per predicate(draw) (in order), each from a fresh seeded stream."""
    chosen, i = [], 0
    for predicate in predicates:
        while i < limit:
            rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
            i += 1
            try:
                draw = _draw(config, rng, "train")
            except RejectedSample:
                continue
            if predicate(draw):
                chosen.append(draw)
                break
    return chosen


def erosion_line(draw):
    """Present-day trace of the erosion surface inside the window (isolated-erosion examples have no faults)."""
    scene = draw[2]
    u = np.linspace(-25, 25, 1001)
    y, z = scene.present(u, _erosion_surface(scene.erosion, u))
    keep = (np.abs(y) <= 16 / 3) & (np.abs(z) <= 16 / 3)
    return y[keep], z[keep]


# ------------------------------------------------------------------ figures

class Plot:
    def __init__(self, grid, slo, shi):
        self.extent = [*grid["y_bounds_ft"], grid["z_bounds_ft"][1], grid["z_bounds_ft"][0]]
        self.norm = Normalize(slo, shi)
        ye = np.linspace(*grid["y_bounds_ft"], grid["ny"] + 1)
        self.ze = np.linspace(*grid["z_bounds_ft"], grid["nz"] + 1)
        self.column = grid["ny"] // 2
        self.log_y = float((ye[self.column] + ye[self.column + 1]) / 2)

    def show(self, ax, image, title=None, logline=False, fontsize=8):
        im = ax.imshow(image, extent=self.extent, origin="upper", aspect="equal", cmap=CMAP, norm=self.norm,
                       interpolation="nearest")
        if logline:
            ax.axvline(self.log_y, color=MARK_COLOR, lw=1.0, ls="--")
        if title:
            ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=6)
        return im

    @staticmethod
    def colorbar(fig, im, axes=None, **kw):
        cb = fig.colorbar(im, ax=axes, **kw)
        cb.set_ticks(np.log10(RHO_TICKS))
        cb.set_ticklabels([f"{t:g}" for t in RHO_TICKS])
        cb.set_label("电阻率（Ω·m，对数色标）", fontsize=8)
        cb.ax.tick_params(labelsize=6)

    def log_track(self, ax, image):
        """Vertical resistivity log through the marked column, depth downward like a well log."""
        rho = 10 ** np.asarray(image[:, self.column], float)
        ax.plot(np.repeat(rho, 2), np.column_stack([self.ze[:-1], self.ze[1:]]).ravel(), color=LOG_COLOR, lw=1.3)
        ax.set_xscale("log")
        ax.set_xlim(0.15, 700)
        ax.set_ylim(self.ze[-1], self.ze[0])
        ax.set_xticks([0.2, 1, 10, 100])
        ax.set_xticklabels(["0.2", "1", "10", "100"])
        ax.minorticks_off()
        ax.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_xlabel("Ω·m", fontsize=7)
        ax.tick_params(labelsize=6, labelleft=False)
        ax.set_title("竖直测井", fontsize=7)

    def row(self, draws, titles, overlays=None):
        fig, axes = plt.subplots(1, len(draws), figsize=(2.75 * len(draws), 3.15), layout="constrained", squeeze=False)
        for n, (ax, draw, title) in enumerate(zip(axes[0], draws, titles)):
            im = self.show(ax, draw[0], title)
            if overlays:
                ax.plot(*overlays[n], color=MARK_COLOR, lw=1.4, ls="--")
                ax.set_xlim(self.extent[0], self.extent[1])
                ax.set_ylim(self.extent[2], self.extent[3])
        self.colorbar(fig, im, axes[0].tolist(), shrink=0.8, pad=0.01)
        return png(fig, dpi=100)

    def pairs(self, images, titles, cols=4):
        """Image + vertical log at the dashed line, `cols` pairs per row."""
        rows = math.ceil(len(images) / cols)
        fig, axes = plt.subplots(rows, 2 * cols, figsize=(12, 2.9 * rows + 0.6), layout="constrained", squeeze=False,
                                 gridspec_kw={"width_ratios": [3, 1.2] * cols})
        for i, (image, title) in enumerate(zip(images, titles)):
            r, c = divmod(i, cols)
            im = self.show(axes[r, 2 * c], image, title, logline=True, fontsize=7.5)
            self.log_track(axes[r, 2 * c + 1], image)
        self.colorbar(fig, im, axes[:, ::2].ravel().tolist(), location="bottom", shrink=0.35, aspect=35)
        # Freeze the layout, then give each log track exactly the vertical extent of its image.
        fig.draw_without_rendering()
        fig.set_layout_engine("none")
        for i in range(len(images)):
            r, c = divmod(i, cols)
            img, log = axes[r, 2 * c].get_position(), axes[r, 2 * c + 1].get_position()
            axes[r, 2 * c + 1].set_position([log.x0, img.y0, log.width, img.height])
        return png(fig, dpi=100)

    def gallery(self, images, titles, cols, size=2.0):
        rows = math.ceil(len(images) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(size * cols, (size + 0.25) * rows + 0.5), layout="constrained", squeeze=False)
        for ax, image, title in zip(axes.flat, images, titles):
            im = self.show(ax, image, title, fontsize=6.5)
        for ax in axes.flat[len(images):]:
            ax.axis("off")
        self.colorbar(fig, im, axes.ravel().tolist(), location="bottom", shrink=0.4, aspect=40)
        return png(fig, dpi=100)


def describe(component, draw):
    """Figure title for one illustration, in physical terms."""
    _, info, scene, _ = draw
    p = info["parameters"]
    if component == "undulation":
        kinds = Counter("高斯拱/凹" if t["kind"] == "gaussian" else "正弦" for t in p["undulation"])
        amp = max(abs(t["amplitude_ft"]) for t in p["undulation"])
        waves = [t.get("wavelength_ft", 4 * t.get("width_ft", 0)) for t in p["undulation"]]
        return f"{'+'.join(f'{k}×{v}' for k, v in kinds.items())}\n最大幅度 {amp:.2f} ft，波长 {min(waves):.0f}–{max(waves):.0f} ft"
    if component == "fault":
        f = p["faults"]
        sense = "/".join("正" if x["sense"] == "normal" else "逆" for x in f)
        dips = "/".join(f"{90 - abs(x['tilt_from_vertical_deg']):.0f}°" for x in f)
        throws = "/".join(f"{abs(x['throw_ft']):.2f}" for x in f)
        return f"{len(f)} 条{sense}断层，倾角 {dips}\n断距 {throws} ft"
    if component == "pinchout":
        q = p["pinchout"]
        return (f"{MODE[q['mode']]}，楔形长 {q['taper_length_ft']:.1f} ft\n"
                f"楔形角 {q['wedge_angle_deg']:.0f}°，尖灭点在窗{'内' if q['tip_in_window'] else '外'}")
    if component == "erosion":
        e = p["erosion"]
        return f"起伏幅度 {abs(e['amplitude_ft']):.2f} ft，波长 {e['wavelength_ft']:.0f} ft\n与层理夹角 {e['relative_tilt_deg']:.1f}°"
    if component == "fluid_contact":
        f = p["fluid_contact"]
        return f"界面以下降低 {ratio(f['drop_log10'])} 倍\n过渡带 {f['transition_ft']:.2f} ft，地层倾角 {p['dip_deg']:.1f}°"
    if component == "gradational":
        widths = [w for stack in (scene.lower, scene.upper) if stack is not None for w in stack.gradational if w > 0]
        return f"渐变宽度 {min(widths):.2f}–{max(widths):.2f} ft\n地层倾角 {p['dip_deg']:.1f}°"
    if component == "trend":
        largest = max(abs(l["trend_log10"]) for l in p["visible_layers"])
        return f"层内最大变化 {ratio(largest)} 倍\n最大对比 {ratio(info['span_log10'])} 倍"
    if component == "thick_bed":
        return (f"ρ = {10 ** p['thick_bed_log10rho']:.1f} Ω·m\n"
                f"窗口内变化 {ratio(abs(p['trend_log10']))} 倍，倾角 {p['dip_deg']:.1f}°")
    raise KeyError(component)


def pipeline(config, plot):
    """One sample with all five structures, rebuilt in geological order."""
    c = variant(config, STRUCT, gradational=0.15, trend=0.2)
    for i in range(3000):
        try:
            draw = _draw(c, np.random.default_rng(np.random.SeedSequence([20261109, i])), "train")
        except RejectedSample:
            continue
        if (all(t in draw[1]["tags"] for t in STRUCT) and len(draw[1]["parameters"]["faults"]) == 1
                and draw[1]["span_log10"] > 1.2 and len(draw[1]["parameters"]["visible_layers"]) >= 5):
            break
    _, info, scene, (y, z) = draw
    off = {"pinchout": None, "erosion": None, "undulation": [], "faults": [], "fluid": None}
    order = [("① 沉积层序（为对比已画上倾角）", ()),
             ("② + 尖灭", ("pinchout",)),
             ("③ + 侵蚀面与上覆层序", ("pinchout", "erosion")),
             ("④ + 褶曲起伏", ("pinchout", "erosion", "undulation")),
             ("⑤ + 断层", ("pinchout", "erosion", "undulation", "faults")),
             ("⑥ + 流体界面 = 最终模型", ("pinchout", "erosion", "undulation", "faults", "fluid"))]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4), layout="constrained")
    for ax, (title, keep) in zip(axes.flat, order):
        step = copy.copy(scene)
        for key, value in off.items():
            if key not in keep:
                setattr(step, key, value)
        values, _, _ = step.evaluate(y, z)
        im = plot.show(ax, values, title)
    p = info["parameters"]
    plot.colorbar(fig, im, axes.ravel().tolist(), shrink=0.75, pad=0.02)
    params = [f"地层倾角 {p['dip_deg']:.1f}°",
              f"层厚中位数 {p['median_thickness_ft']:.2f} ft（{p['median_thickness_ft'] * FT:.2f} m）",
              f"{len(p['palette_log10rho'])} 种电阻率 " + "、".join(f"{10 ** v:.3g}" for v in p["palette_log10rho"]) + " Ω·m",
              f"尖灭为{MODE[p['pinchout']['mode']]}，楔形角 {p['pinchout']['wedge_angle_deg']:.0f}°",
              f"侵蚀面起伏幅度 {abs(p['erosion']['amplitude_ft']):.2f} ft",
              f"褶曲起伏分量 {len(p['undulation'])} 个",
              f"断层倾角 {90 - abs(p['faults'][0]['tilt_from_vertical_deg']):.0f}°、断距 {abs(p['faults'][0]['throw_ft']):.2f} ft",
              f"流体界面以下降低 {ratio(p['fluid_contact']['drop_log10'])} 倍"]
    return png(fig, dpi=105), "；".join(params)


# Example selection: each illustration shows one structure clearly.
SPECS = {
    "pinchout": [lambda i, m=m, t=t: i["parameters"]["pinchout"]["mode"] == m and i["parameters"]["pinchout"]["tip_in_window"] == t
                 for m, t in (("top_descends", True), ("base_rises", True), ("symmetric", True), ("symmetric", False))],
    "erosion": [lambda i: abs(i["parameters"]["erosion"]["amplitude_ft"]) > 0.4 and abs(i["parameters"]["erosion"]["relative_tilt_deg"]) > 2] * 2
               + [lambda i: abs(i["parameters"]["erosion"]["relative_tilt_deg"]) > 3] * 2,
    "undulation": [lambda i: any(t["kind"] == "gaussian" for t in i["parameters"]["undulation"]),
                   lambda i: all(t["kind"] == "sine" for t in i["parameters"]["undulation"]),
                   lambda i: len(i["parameters"]["undulation"]) == 3, lambda i: True],
    "fault": [lambda i: len(i["parameters"]["faults"]) == 1 and i["parameters"]["faults"][0]["sense"] == "normal",
              lambda i: len(i["parameters"]["faults"]) == 1 and i["parameters"]["faults"][0]["sense"] == "reverse",
              lambda i: len(i["parameters"]["faults"]) == 1 and abs(i["parameters"]["faults"][0]["throw_ft"]) > 2,
              lambda i: len(i["parameters"]["faults"]) == 2],
    "fluid_contact": [lambda i, f=f: abs(i["parameters"]["fluid_contact"]["contact_z_ft"]) < 3.5 and f(i["parameters"])
                      for f in (lambda p: abs(p["dip_deg"]) > 5, lambda p: p["fluid_contact"]["transition_ft"] > 1.0,
                                lambda p: p["fluid_contact"]["transition_ft"] < 0.2, lambda p: True)],
}
SEED = {"undulation": 100, "fault": 101, "pinchout": 102, "erosion": 103, "fluid_contact": 104}
LEAST = {"undulation": 300, "fault": 1200, "pinchout": 800, "erosion": 1500, "fluid_contact": 800}


def component_figures(config, plot):
    figures = {}
    for component, predicates in SPECS.items():
        cfg = variant(config, (component,))
        wanted = [lambda d, p=p, comp=component: comp in d[1]["tags"] and d[1]["span_log10"] > 0.8 and p(d[1])
                  and changed(d, comp) >= LEAST[comp] and (comp != "pinchout" or neighbour_contrast(d) > 0.6)
                  and (comp != "pinchout" or not d[1]["parameters"]["pinchout"]["tip_in_window"] or central_tip(d))
                  and (comp not in ("undulation", "fault", "pinchout") or len(d[1]["parameters"]["visible_layers"]) >= 4)
                  for p in predicates]
        draws = pick(cfg, wanted, seed=SEED[component])
        overlays = None
        if component == "erosion":
            overlays = [erosion_line(d) for d in draws]
        elif component == "fluid_contact":
            overlays = [(plot.extent[:2], [d[1]["parameters"]["fluid_contact"]["contact_z_ft"]] * 2) for d in draws]
        figures[component] = plot.row(draws, [describe(component, d) for d in draws], overlays)
    grad = pick(variant(config, gradational=0.6), [lambda d: "gradational" in d[1]["tags"] and d[1]["span_log10"] > 0.8] * 4, seed=200)
    figures["gradational"] = plot.row(grad, [describe("gradational", d) for d in grad])
    trend = pick(variant(config, trend=1.0), [lambda d: max(abs(l["trend_log10"]) for l in d[1]["parameters"]["visible_layers"]) > 0.12] * 4, seed=201)
    figures["trend"] = plot.row(trend, [describe("trend", d) for d in trend])
    thick = pick(variant(config, thick=1.0), [lambda d: True] * 4, seed=202)
    figures["thick_bed"] = plot.row(thick, [describe("thick_bed", d) for d in thick])
    # Bed-thickness scale: base stacks only, one per median-thickness band, with vertical logs.
    bands = ((1.0, 1.4), (1.9, 2.6), (3.4, 4.6), (6.0, 8.0))
    scale = pick(variant(config), [lambda d, lo=lo, hi=hi: lo <= d[1]["parameters"]["median_thickness_ft"] < hi
                                   and abs(d[1]["parameters"]["dip_deg"]) < 4 and d[1]["span_log10"] > 0.8
                                   for lo, hi in bands], seed=300)
    titles = [f"层厚中位数 {d[1]['parameters']['median_thickness_ft']:.1f} ft（{d[1]['parameters']['median_thickness_ft'] * FT:.2f} m）\n"
              f"窗口内 {len(d[1]['parameters']['visible_layers'])} 层，{len(d[1]['parameters']['palette_log10rho'])} 种电阻率" for d in scale]
    figures["scale"] = plot.pairs([d[0] for d in scale], titles)
    return figures


def struct_label(tags):
    if "thick_bed" in tags:
        return "厚层"
    names = [CN[t] for t in STRUCT if t in tags]
    return "+".join(names) if names else "平面层序"


# ------------------------------------------------------------------ HTML pieces

SRC = {"lit": "文献", "eng": "工程设定", "geo": "网格/反演"}


def ptable(rows):
    body = []
    for name, value, kind, note in rows:
        note_html = f'<span class="note">{note}</span>' if note else ""
        body.append(f'<tr><td class="pname">{name}</td><td class="pval">{value}</td>'
                    f'<td><span class="src {kind}">{SRC[kind]}</span>{note_html}</td></tr>')
    return ('<div class="scroll"><table class="params"><thead><tr><th>参数</th><th>取值</th><th>来源与说明</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def figure(src, alt, caption):
    return (f'<figure class="fig"><div class="plate"><img src="{src}" alt="{esc(alt)}"></div>'
            f'<figcaption>{caption}</figcaption></figure>')


def chart(svg, caption, solo=False):
    return f'<div class="card chart-card{" solo" if solo else ""}">{svg}<p class="caption">{caption}</p></div>'


CSS_TOKENS_DARK = """--page:#0e1116; --surface-1:#161a21; --surface-2:#1c212a; --plate:#fcfcfb;
  --text-primary:#e8ebf0; --text-secondary:#b0b8c4; --muted:#8a93a0;
  --border:#2a313b; --grid:#262c35; --axis:#3b434e; --accent:#77aef0; --accent-soft:#1a2738;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --lit:#6fd3ad; --lit-bg:#123128; --eng:#eab95c; --eng-bg:#33270f; --geo:#c0c9d6; --geo-bg:#232a35;
  color-scheme:dark;"""

CSS = """
/* Layout: one reading column (prose ≤ 46em) inside a 1080px frame; figures, charts and tables use the full frame.
   Figures sit on a light "plate" in both themes because they are rendered images. */
:root {
  --page:#f5f6f8; --surface-1:#ffffff; --surface-2:#eef1f5; --plate:#fcfcfb;
  --text-primary:#151a21; --text-secondary:#465163; --muted:#7b8594;
  --border:#dde2e9; --grid:#e3e7ed; --axis:#c2c8d2; --accent:#1d5ca8; --accent-soft:#e4edf8;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --lit:#0b6a4c; --lit-bg:#dff2ea; --eng:#8a5800; --eng-bg:#fbefd6; --geo:#3a4658; --geo-bg:#e6eaf0;
  --font-body:"PingFang SC","Hiragino Sans GB","Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { DARK } }
:root[data-theme="dark"] { DARK }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--text-primary); font:15px/1.75 var(--font-body);
  padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); padding-block:0; }
main { max-width:1080px; margin:0 auto; padding-block:36px 72px; }
p, ul, ol { margin:0 0 12px; }
.prose > p, .prose > ul, p.prose, ul.prose { max-width:46em; }
h1 { font-size:27px; line-height:1.3; margin:6px 0 10px; text-wrap:balance; letter-spacing:.01em; }
h2 { font-size:21px; line-height:1.35; margin:48px 0 12px; padding-top:18px; border-top:1px solid var(--border); text-wrap:balance; }
h3 { font-size:16.5px; margin:30px 0 8px; text-wrap:balance; }
h2 .no, h3 .no { color:var(--accent); font-variant-numeric:tabular-nums; margin-right:8px; }
a { color:var(--accent); } a:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:2px; }
code { font:12.5px var(--font-mono); background:var(--surface-2); padding:1px 5px; border-radius:4px; }
.eyebrow { color:var(--muted); font-size:13px; letter-spacing:.06em; }
.lede { font-size:16px; color:var(--text-secondary); max-width:46em; }
dl.facts { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 180px), 1fr)); gap:10px; margin:20px 0; }
dl.facts > div { background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:10px 14px; min-width:0; }
dl.facts dt { color:var(--muted); font-size:12.5px; }
dl.facts dd { margin:0; font-size:19px; font-weight:650; font-variant-numeric:tabular-nums; line-height:1.4; }
dl.facts dd.sub { font-size:12.5px; font-weight:400; color:var(--text-secondary); }
.guide { background:var(--accent-soft); border-radius:10px; padding:12px 16px; margin:16px 0; }
.guide p { margin:0 0 6px; } .guide ul { margin:0; padding-left:0; list-style:none; display:grid; gap:4px; }
nav.toc ol { columns:2 240px; column-gap:32px; padding-left:22px; margin:8px 0 0; font-size:14px; }
nav.toc li { break-inside:avoid; }
.src { display:inline-block; font-size:12px; font-weight:600; line-height:1.6; padding:0 8px; border-radius:999px; margin-right:6px; white-space:nowrap; }
.src.lit { color:var(--lit); background:var(--lit-bg); } .src.eng { color:var(--eng); background:var(--eng-bg); }
.src.geo { color:var(--geo); background:var(--geo-bg); }
.scroll { overflow-x:auto; margin:10px 0 14px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); vertical-align:top; }
th { color:var(--text-secondary); font-weight:600; white-space:nowrap; font-size:13px; }
td.r, th.r { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
table.params td.pname { white-space:nowrap; font-weight:600; }
table.params td.pval { min-width:14em; font-variant-numeric:tabular-nums; }
table.params td:last-child { min-width:16em; }
table.params .note { display:block; color:var(--text-secondary); font-size:13px; margin-top:3px; }
td.barcell { min-width:150px; white-space:nowrap; }
.inbar { display:inline-block; width:90px; height:8px; background:var(--grid); border-radius:4px; vertical-align:middle; margin-right:8px; overflow:hidden; }
.inbar b { display:block; height:100%; background:var(--series-1); border-radius:4px; }
.num { font-variant-numeric:tabular-nums; }
.fig { margin:14px 0 18px; }
.plate { background:var(--plate); border:1px solid var(--border); border-radius:8px; padding:8px; overflow-x:auto; }
.plate img { display:block; max-width:100%; height:auto; margin:0 auto; }
figcaption, .caption { color:var(--text-secondary); font-size:13px; margin-top:6px; max-width:60em; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap:14px; margin:12px 0; }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px; min-width:0; }
.card .caption { margin-bottom:0; }
ol.timeline { list-style:none; padding:0; margin:14px 0; display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 150px), 1fr)); gap:8px; counter-reset:step; }
ol.timeline li { counter-increment:step; background:var(--surface-1); border:1px solid var(--border); border-radius:8px; padding:10px 12px; min-width:0; }
ol.timeline li::before { content:counter(step); display:inline-grid; place-items:center; width:20px; height:20px; border-radius:50%;
  background:var(--accent-soft); color:var(--accent); font-size:12px; font-weight:700; margin-right:6px; vertical-align:1px; }
ol.timeline b { font-weight:650; } ol.timeline span { display:block; color:var(--text-secondary); font-size:13px; margin-top:4px; line-height:1.55; }
ul.limits { padding-left:20px; max-width:50em; } ul.limits li { margin:6px 0; }
ol.refs { font-size:13.5px; color:var(--text-secondary); padding-left:22px; } ol.refs li { margin:4px 0; }
.meta { color:var(--muted); font-size:12.5px; margin-top:40px; }
.version { color:var(--text-secondary); font-size:13.5px; max-width:46em; border-left:3px solid var(--accent); padding-left:10px; }
svg.chart { width:100%; height:auto; display:block; }
svg .grid { stroke:var(--grid); stroke-width:1; } svg .axis { stroke:var(--axis); stroke-width:1; }
svg .tick { fill:var(--muted); font-size:11px; } svg .axis-label { fill:var(--text-secondary); font-size:12px; }
svg .mark:hover { opacity:.75; } svg .marker { stroke:var(--text-secondary); stroke-width:1.5; stroke-dasharray:4 3; }
svg .marker-label { fill:var(--text-primary); font-size:11.5px; paint-order:stroke; stroke:var(--surface-1); stroke-width:3px; stroke-linejoin:round; }
svg .step { fill:none; stroke-width:2; }
svg .window { fill:none; stroke:var(--text-primary); stroke-width:2; }
svg .rx { fill:var(--series-1); stroke:var(--surface-1); stroke-width:1; } svg .tx { fill:var(--series-3); }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:12.5px; color:var(--text-secondary); margin:2px 0 6px; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.legend .sw-window { border:2px solid var(--text-primary); background:none; }
.window-card { max-width:440px; }
.chart-card.solo { max-width:540px; margin:12px 0; }
table.kv td:first-child { white-space:nowrap; }
@media (max-width:560px) { h1 { font-size:22px; } h2 { font-size:19px; } table { font-size:13px; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior:auto; } }
""".replace("DARK", CSS_TOKENS_DARK)


# ------------------------------------------------------------------ report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, help="also write the page without <html>/<head>/<body> to this path")
    args = parser.parse_args()

    manifest = json.loads((DATA / "manifest.json").read_text())
    c = manifest["config"]
    grid, r, st, pr, dp, comp = c["grid"], c["resistivity"], c["stratigraphy"], c["properties"], c["dip"], c["components"]
    slo, shi = r["sampling_log10_bounds"]
    plot = Plot(grid, slo, shi)
    train = [json.loads(l) for l in (DATA / "train/metadata.jsonl").read_text().splitlines()]
    arrays = np.load(DATA / "train/log10rho.npy", mmap_mode="r")
    layered = [x for x in train if "thick_bed" not in x["tags"]]
    n_train, n_layered = len(train), len(layered)
    window_ft = float(np.ptp(grid["y_bounds_ft"]))
    cell_in = window_ft / grid["ny"] * 12

    # ---- statistics (train)
    thickness = np.array([l["thickness_ft"] for x in layered for l in x["parameters"]["visible_layers"]])
    n_layers = np.array([len(x["parameters"]["visible_layers"]) for x in layered])
    spans = np.array([x["span_log10"] for x in layered])
    dips = np.array([x["parameters"]["dip_deg"] for x in train])
    palette_sizes = Counter(len(x["parameters"]["palette_log10rho"]) for x in layered)
    rng = np.random.default_rng(20261111)
    sample_idx = np.sort(rng.choice(n_train, 2000, replace=False))
    pixels = np.asarray(arrays[sample_idx, 0], np.float32).ravel()
    edges_rho = np.linspace(slo, shi, 69)
    area = np.histogram(pixels, edges_rho)[0] / pixels.size * 100
    area_bands = [np.mean(pixels < 0), np.mean((pixels >= 0) & (pixels < 1)), np.mean((pixels >= 1) & (pixels < 2)), np.mean(pixels >= 2)]

    edges_t = np.linspace(math.log10(0.2), math.log10(12), 46)
    thick_svg = svg_histogram(edges_t, [("与窗口相交的层", "--series-1", np.histogram(np.log10(thickness), edges_t)[0])],
                              "单层厚度（ft，对数刻度）", "层数", markers=[(math.log10(0.5), "15 cm"), (math.log10(1 / FT), "1 m")],
                              aria="单层厚度分布", xfmt=lambda v: f"{10 ** v:.2g}",
                              xticks=[math.log10(v) for v in (0.2, 0.5, 1, 2, 5, 10)], vfmt=lambda n: f"{int(n):,} 层")
    edges_n = np.arange(0.5, 21.5, 1)
    count_svg = svg_histogram(edges_n, [("模型", "--series-1", np.histogram(np.clip(n_layers, 1, 20), edges_n)[0])],
                              "窗口内可见层数（20 及以上并入 20）", "模型数", aria="每个模型窗口内的层数",
                              xticks=[1, 5, 10, 15, 20], vfmt=lambda n: f"{int(n):,} 个模型")
    rho_svg = svg_histogram(edges_rho, [("电阻率", "--series-1", area)], "电阻率（Ω·m，对数刻度）", "面积占比（%）",
                            aria="全部单元的电阻率分布", xfmt=lambda v: f"{10 ** v:.3g}",
                            xticks=[math.log10(v) for v in RHO_TICKS], vfmt=lambda n: f"占面积 {n:.2f}%")
    edges_s = np.arange(0, 2.55 + 1e-9, 0.05)
    span_svg = svg_histogram(edges_s, [("模型", "--series-1", np.histogram(spans, edges_s)[0])],
                             "窗口内最大与最小电阻率之比（倍，对数刻度）", "模型数",
                             markers=[(math.log10(2), "2 倍"), (math.log10(50), "50 倍")], aria="每个模型的最大对比度",
                             xfmt=lambda v: f"{10 ** v:.0f}", xticks=[math.log10(v) for v in (1, 2, 5, 10, 20, 50, 100, 316)],
                             vfmt=lambda n: f"{int(n):,} 个模型")
    edges_d = np.arange(-16, 16.01, 0.5)
    dip_svg = svg_histogram(edges_d, [("模型", "--series-1", np.histogram(dips, edges_d)[0])],
                            "地层视倾角（°），正值向右加深", "模型数", markers=[(-5, "−5°"), (5, "5°")], aria="地层倾角分布",
                            xticks=[-15, -10, -5, 0, 5, 10, 15], xfmt=lambda v: f"{v:g}°", vfmt=lambda n: f"{int(n):,} 个模型")

    pins = [x["parameters"]["pinchout"] for x in train if x["parameters"].get("pinchout")]
    faults = [f for x in train for f in (x["parameters"].get("faults") or [])]
    fluids = [x["parameters"]["fluid_contact"] for x in train if x["parameters"].get("fluid_contact")]
    erosions = [x["parameters"]["erosion"] for x in train if x["parameters"].get("erosion")]
    undulations = [x["parameters"]["undulation"] for x in train if x["parameters"].get("undulation")]

    # ---- combinations (train)
    combo = Counter(tuple(t for t in STRUCT if t in x["tags"]) for x in layered)
    n_struct = Counter(len(k) for k in combo.elements())
    with_count = {t: sum(t in x["tags"] for x in layered) for t in STRUCT}
    marg = {t: with_count[t] / n_layered for t in STRUCT}
    expect = lambda key: np.prod([marg[t] if t in key else 1 - marg[t] for t in STRUCT])
    order_cn = ("pinchout", "erosion", "undulation", "fault", "fluid_contact")
    prob_rows = "".join(
        f'<tr><td>{CN[t]}</td><td class="r">{comp[t]["probability"]:.0%}</td>'
        f'<td class="barcell">{bar_cell(with_count[t], n_layered)}</td><td class="r">{marg[t]:.1%}</td>'
        f'<td class="r">{combo.get((t,), 0) / with_count[t]:.0%}</td></tr>' for t in order_cn)
    count_rows = "".join(f'<tr><td class="r">{k}</td><td class="barcell">{bar_cell(n_struct.get(k, 0), n_layered)}</td>'
                         f'<td class="r">{n_struct.get(k, 0) / n_layered:.1%}</td></tr>' for k in range(max(n_struct) + 1))
    combo_rows = "".join(
        f'<tr><td>{" + ".join(CN[t] for t in key) or "无结构（平面层序）"}</td><td class="r">{n:,}</td>'
        f'<td class="r">{n / n_layered:.1%}</td><td class="r">{expect(key):.1%}</td></tr>'
        for key, n in combo.most_common(12))
    multi = sum(v for k, v in n_struct.items() if k >= 2) / n_layered

    # ---- figures
    print("figures: pipeline", flush=True)
    pipeline_png, pipeline_params = pipeline(c, plot)
    print("figures: components", flush=True)
    figs = component_figures(c, plot)
    print("figures: combinations", flush=True)
    show = [key for key, n in combo.most_common() if len(key) >= 2][:7]
    show += [key for key, n in combo.most_common() if len(key) >= 3 and key not in show][:1]
    visible = {"undulation": 300, "fault": 300, "pinchout": 300, "erosion": 300, "fluid_contact": 300}
    combo_images, combo_titles = [], []
    for key in show:
        for x in [x for x in layered if tuple(t for t in STRUCT if t in x["tags"]) == key and x["span_log10"] > 0.8][:400]:
            draw = redraw(c, x)
            if all(changed(draw, t) >= visible[t] for t in key):
                combo_images.append(draw[0])
                combo_titles.append(f"{' + '.join(CN[t] for t in key)}\n倾角 {x['parameters']['dip_deg']:.1f}°，"
                                    f"最大对比 {ratio(x['span_log10'])} 倍")
                break
    combo_png = plot.gallery(combo_images, combo_titles, cols=4, size=2.6)
    print("figures: random gallery", flush=True)
    gallery_idx = np.sort(np.random.default_rng(7).choice(n_train, 24, replace=False))
    gallery_titles = [f"{struct_label(train[i]['tags'])}\n倾角 {train[i]['parameters']['dip_deg']:.1f}°，"
                      f"对比 {ratio(train[i]['span_log10'])} 倍" for i in gallery_idx]
    gallery_png = plot.gallery([arrays[i, 0] for i in gallery_idx], gallery_titles, cols=6)
    log_idx = [i for i in gallery_idx if "thick_bed" not in train[i]["tags"]][:8]
    logs_png = plot.pairs([arrays[i, 0] for i in log_idx],
                          [f"{struct_label(train[i]['tags'])}\n中位层厚 {train[i]['parameters']['median_thickness_ft']:.1f} ft，"
                           f"倾角 {train[i]['parameters']['dip_deg']:.1f}°" for i in log_idx])

    # ---- parameter tables
    m_lo, m_hi = st["median_thickness_ft"]
    t_lo, t_hi = st["thickness_clip_ft"]
    sigma = st["thickness_log_sigma"]
    weak, strong = r["weak_span_log10"], r["strong_span_log10"]
    sizes = r["palette_size_probabilities"]
    u, f, p, e, fl = (comp[k] for k in ("undulation", "fault", "pinchout", "erosion", "fluid_contact"))

    thickness_table = ptable([
        ("层厚中位数", f"每个模型抽一个：{length(m_lo, m_hi)}，对数均匀（1–2、2–4、4–8 ft 三段概率相同）", "eng",
         "上一版取 0.4–6 ft，窗口内中位 8 层、四成模型超过 10 层，我们认为过密，改为现值"),
        ("单层厚度", f"以中位数为中心的对数正态分布，对数标准差 {sigma:g}（约 2/3 的层在中位数的 0.6–1.6 倍之间）", "lit",
         "Talling（2001）：浊积岩层厚近似对数正态（混合）分布"),
        ("厚度截断", length(t_lo, t_hi), "geo",
         f"下限约为 2 个网格单元（{cell_in:.0f} inch 单元）；上限超过窗口高度"),
    ])
    rho_table = ptable([
        ("电阻率范围", f"{10 ** slo:.1f}–{10 ** shi:.0f} Ω·m", "eng",
         "参考公开算例确定的覆盖范围：Shahriari 等（2020）1–1000 Ω·m；Hashemian 等（2021）1–100 Ω·m，复杂模型 0.5/3/100 Ω·m"),
        ("每个模型的电阻率级别数", " / ".join(f"{k} 种 {v:.0%}" for k, v in sorted(sizes.items())), "eng",
         "这里的“岩性”只是电阻率级别，不对应具体岩石名称"),
        ("最大对比度（最高/最低）",
         f"{r['weak_span_probability']:.0%} 的模型为弱对比：{ratio(weak[0])}–{ratio(weak[1])} 倍；"
         f"其余 {ratio(strong[0])}–{ratio(strong[1])} 倍（对数均匀）", "lit",
         "上限覆盖 Chemali（2009）的典型条件：50 Ω·m 储层对 1 Ω·m 水层或泥岩盖层（50 倍）；弱对比所占比例为工程设定"),
        ("各级别的位置", "最低值在允许范围内对数均匀抽取；中间级别在最低与最高之间对数均匀抽取", "eng", ""),
        ("层的电阻率", "从本模型的级别中随机取，且与上一层不同；同一级别可在层序中重复出现", "eng",
         "层的先后顺序没有旋回或相序规律"),
        ("层间扰动", f"每层 ±{pr['layer_jitter_log10']:g} 个数量级（±{(10 ** pr['layer_jitter_log10'] - 1) * 100:.0f}%）", "eng",
         "同一级别的不同层电阻率略有差别"),
    ])
    dip_table = ptable([
        ("地层视倾角", f"{dp['gentle_probability']:.0%} 的模型在 ±{dp['gentle_abs_deg']:g}° 内均匀取值，"
                  f"其余在 ±{dp['max_abs_deg']:g}° 内均匀取值", "lit",
         "Shahriari 等（2020）的合成数据集取地层倾角 ±10°；两档的比例为工程设定"),
        ("倾斜中心", "窗口中部随机一点（横向、纵向各 ±25% 窗口范围）", "eng", ""),
    ])
    pinchout_table = ptable([
        ("出现概率", f"{p['probability']:.0%} 的模型", "eng", ""),
        ("宿主层", f"窗口内可见、厚度 ≥ {length(p['min_host_thickness_ft'])} 的任意一层", "eng", ""),
        ("收敛方式", "顶面下降（上覆层填入）/ 底面上升（下伏层填入）/ 上下对称收敛，各 1/3", "eng", ""),
        ("楔形长度", f"{length(*p['taper_length_ft'])}，对数均匀；同时保证楔形角 ≤ {p['max_wedge_angle_deg']:g}°", "eng",
         "Zhang 等（2025）用楔形地层做正演算例；本数据的尺寸为工程设定"),
        ("尖灭点位置", "楔形段与窗口有重叠即可，尖灭点可以在窗口外（窗口里只看到减薄的一段）", "eng", ""),
    ])
    erosion_table = ptable([
        ("出现概率", f"{e['probability']:.0%} 的模型", "eng", ""),
        ("侵蚀面位置", "窗口中心对应的层位上下 ±30% 窗口高度内", "eng", ""),
        ("侵蚀面形状", f"正弦起伏：波长 {length(*e['wavelength_ft'])}，幅度 {length(*e['amplitude_ft'])}，坡度 ≤ {e['max_slope']:g}", "eng", ""),
        ("与层理的夹角", f"±{e['max_relative_tilt_deg']:g}° 内均匀", "eng", "不生成角度不整合"),
        ("上覆层序", "用同一套电阻率级别和同一层厚中位数重新抽取，与下伏层序平行", "eng", ""),
    ])
    undulation_table = ptable([
        ("出现概率", f"{u['probability']:.0%} 的模型", "eng", ""),
        ("分量", f"{u['count'][0]}–{u['count'][1]} 个叠加；每个一半概率为高斯形拱或凹，一半为正弦波", "eng", ""),
        ("波长", f"{length(*u['wavelength_ft'])}，均匀；高斯拱的宽度（标准差）为波长的 1/4", "eng", ""),
        ("幅度", f"{length(*u['amplitude_ft'])}，多个分量时除以 √分量数；层面坡度 ≤ {u['max_slope']:g}"
               f"（约 {math.degrees(math.atan(u['max_slope'])):.0f}°）", "eng", ""),
        ("位置", "分量中心可以在窗口外，窗口里可能只看到拱的一翼", "eng", ""),
    ])
    fault_table = ptable([
        ("出现概率", f"{f['probability']:.0%} 的模型，其中 {f['second_fault_probability']:.0%} 有两条", "eng", ""),
        ("断层倾角", f"{90 - f['tilt_from_vertical_deg'][1]:g}–{90 - f['tilt_from_vertical_deg'][0]:g}°，均匀；向左或向右倾随机", "lit",
         "按 Anderson 断层理论正断层倾角约 60°；实测正断层倾角多为 30–60°"),
        ("垂直断距", f"{length(*f['throw_ft'])}，均匀", "eng", ""),
        ("正断层/逆断层", "各约一半", "eng", ""),
        ("位置", "断层面经过窗口中部（横向 ±35%、纵向 ±30% 窗口范围内的随机一点）", "eng", ""),
        ("形态", "平面断层面，一侧整体沿断层面滑移；无牵引、无破碎带", "eng", ""),
    ])
    fluid_table = ptable([
        ("出现概率", f"{fl['probability']:.0%} 的模型", "eng", ""),
        ("储层", f"电阻率不低于本模型各级别中位数、窗口内有厚度 ≥ {length(fl['min_reservoir_thickness_ft'])} 的层的级别中随机一种", "eng",
         "只有这一种级别受影响，其他层不变"),
        ("界面", "现今坐标中水平；深度在储层单元深度的 10%–90% 分位之间均匀", "eng", "断层两盘、侵蚀面上下使用同一界面深度"),
        ("界面以下电阻率降低", f"{fl['drop_log10'][0]:g}–{fl['drop_log10'][1]:g} 个数量级"
                        f"（{ratio(fl['drop_log10'][0])}–{ratio(fl['drop_log10'][1])} 倍），均匀", "lit",
         "Chemali（2009）：50 Ω·m 储层对 1 Ω·m 水层（50 倍）"),
        ("过渡带", f"界面以上 {length(*fl['transition_ft'])}，均匀；带内电阻率按对数线性变化", "eng",
         "毛管过渡带实际可厚达 20 m 以上；在 3.25 m 窗口里，更厚的过渡带近似为层内趋势（5.2 节）"),
    ])
    gradational_table = ptable([
        ("出现概率", f"每条层界 {pr['gradational_probability']:.0%}", "eng", ""),
        ("渐变宽度", f"{length(*pr['gradational_width_ft'])}，均匀；不超过相邻两层厚度的一半", "eng", "宽度内电阻率按对数线性过渡"),
    ])
    trend_table = ptable([
        ("出现概率", f"每层 {pr['trend_probability']:.0%}", "eng", ""),
        ("变化量", f"层顶到层底不超过 {pr['trend_max_log10']:g} 个数量级（{ratio(pr['trend_max_log10'])} 倍），"
                f"且不超过本模型最大对比度的一半；沿垂直层面方向线性", "eng", "代表孔隙度、泥质含量或含油饱和度的渐变"),
    ])
    thick_table = ptable([
        ("出现概率", f"{st['thick_bed_probability']:.0%} 的模型", "eng", "反演从均匀初始模型出发，先验需要能表示这种情况"),
        ("电阻率", f"{10 ** (slo + pr['trend_max_log10']):.2g}–{10 ** (shi - pr['trend_max_log10']):.0f} Ω·m，对数均匀", "eng", ""),
        ("趋势", f"整个窗口内变化 {ratio(0.1 * pr['trend_max_log10'])}–{ratio(pr['trend_max_log10'])} 倍，沿垂直层面方向", "eng", ""),
    ])

    splits = manifest["splits"]
    split_rows = "".join(f'<tr><td>{name}</td><td class="r">{splits[key]["count"]:,}</td><td>{use}</td></tr>' for key, name, use in (
        ("train", "训练", "先验从中学习"),
        ("val", "验证", "训练过程中选择模型"),
        ("test", "测试", "与训练同分布，独立评估"),
        ("ood_contrast", "外推：高对比", f"最大对比 {ratio(c['ood']['contrast_span_log10'][0])}–{ratio(c['ood']['contrast_span_log10'][1])} 倍，超出训练范围"),
        ("ood_dip", "外推：大倾角", f"地层倾角 {c['ood']['dip_abs_deg'][0]:g}–{c['ood']['dip_abs_deg'][1]:g}°，超出训练范围")))

    t50 = np.median(thickness)
    n10, n50, n90 = np.percentile(n_layers, [10, 50, 90])
    dip_within = np.mean(np.abs(dips) <= dp["gentle_abs_deg"])

    content = f"""
<main>
<header>
<div class="eyebrow">数据集 lwd_v4 · 说明日期 2026-11-12</div>
<h1>随钻电阻率反演用地层模型：生成方法说明</h1>
<p class="lede">这些二维电阻率模型是随钻电磁波电阻率反演的训练数据，反演所用的地质先验从中学习“合理的地层剖面是什么样的”。本文说明模型的生成方法、每个参数的取值和来源，并展示生成结果，供从地质角度审查这些设定是否合理。</p>
<p class="version">本版（lwd_v4）与 2026-11-11 的说明（lwd_v3）相比只去掉了钙质结核，其余设定不变。</p>
<dl class="facts">
<div><dt>模型窗口</dt><dd>{window_ft * FT:.2f} × {window_ft * FT:.2f} m</dd><dd class="sub">{window_ft:.2f} × {window_ft:.2f} ft，过井竖直剖面</dd></div>
<div><dt>网格</dt><dd>{grid['ny']} × {grid['nz']}</dd><dd class="sub">单元 {cell_in * 2.54:.2f} cm（{cell_in:.0f} inch）</dd></div>
<div><dt>电阻率</dt><dd>{10 ** slo:.1f}–{10 ** shi:.0f} Ω·m</dd><dd class="sub">单个模型内最大对比 {ratio(strong[1])} 倍</dd></div>
<div><dt>层厚中位数</dt><dd>{m_lo * FT:.2f}–{m_hi * FT:.2f} m</dd><dd class="sub">每个模型抽一个（{m_lo:g}–{m_hi:g} ft）</dd></div>
<div><dt>训练模型</dt><dd>{n_train:,} 个</dd><dd class="sub">每个模型参数独立随机</dd></div>
</dl>
<div class="guide">
<p>每节的参数表都在最后一列注明取值的来源：</p>
<ul>
<li><span class="src lit">文献</span>取值参考了公开文献中的地质观测或典型算例（见文末参考文献）。</li>
<li><span class="src eng">工程设定</span>根据经验确定，没有文献或区块资料直接支持，最需要地质上的判断。</li>
<li><span class="src geo">网格/反演</span>由反演网格的分辨率或窗口大小决定。</li>
</ul>
</div>
<nav class="toc" aria-label="目录"><ol>
<li><a href="#s1">这些模型的用途和基本设定</a></li>
<li><a href="#s2">生成思路</a></li>
<li><a href="#s3">基础层序：层厚、电阻率、倾角</a></li>
<li><a href="#s4">五种地质结构</a></li>
<li><a href="#s5">层界与层内的细节</a></li>
<li><a href="#s6">结构的组合方式</a></li>
<li><a href="#s7">典型结果</a></li>
<li><a href="#s8">简化与未包含的内容</a></li>
<li><a href="#s9">数据规模</a></li>
<li><a href="#refs">参考文献</a></li>
</ol></nav>
</header>

<h2 id="s1"><span class="no">1</span>这些模型的用途和基本设定</h2>
<div class="prose">
<p>我们用随钻电磁波电阻率测量反演井周地层的二维电阻率剖面。剖面有 {grid['ny']}×{grid['nz']} 个未知单元，仅靠测量数据无法唯一确定，需要先验约束：先让一个生成模型（变分自编码器）从大量人工生成的地层模型中学习“合理的剖面是什么样的”，反演时只在它能表示的剖面中寻找与测量吻合的解。</p>
<p>因此这个数据集决定了反演“认为什么样的地层是可能的”。数据集里没有的地质情况，反演很难重建出来；数据集里与实际不符的情况，也可能出现在反演结果中。</p>
</div>
<div class="grid2">
<div class="card"><div class="scroll" style="margin:0"><table class="kv">
<tr><th>项目</th><th>设定</th></tr>
<tr><td>剖面</td><td>过井的竖直剖面：y 水平、z 竖直向下。沿垂直于剖面的方向（走向）地层不变，即二维模型</td></tr>
<tr><td>窗口</td><td>{window_ft:.2f} × {window_ft:.2f} ft（{window_ft * FT:.2f} × {window_ft * FT:.2f} m），是反演区域，工具位于其中</td></tr>
<tr><td>网格</td><td>{grid['ny']} × {grid['nz']} 个正方形单元，每个 {cell_in:.0f} inch（{cell_in * 2.54:.2f} cm）</td></tr>
<tr><td>物理量</td><td>各向同性电阻率 ρ（Ω·m），每个单元一个值</td></tr>
<tr><td>坐标</td><td>全局坐标：网格始终水平、竖直，不随井斜旋转。地层倾角画在模型里，井斜只在正演中出现</td></tr>
<tr><td>不包含</td><td>井眼、泥浆侵入（属于测量环境，不放在地质模型里）</td></tr>
</table></div></div>
<div class="card window-card">{svg_window(grid)}<p class="caption">窗口与工具的相对尺度（示意）：12 个接收器和末端发射器，井斜 90° 和 75° 两种情况。坐标单位 ft，原点在窗口中心。</p></div>
</div>

<h2 id="s2"><span class="no">2</span>生成思路</h2>
<div class="prose">
<p>每个模型先生成一套“基础层序”（每个模型都有），再对五种地质结构分别按概率独立决定是否加入。抽中的结构按地质先后顺序作用到同一个模型上，所以一个模型可以同时含几种结构，例如被断层错开的褶曲地层中有油水界面。没有被抽中任何结构的模型就是倾斜的平面层序。</p>
</div>
<ol class="timeline">
<li><b>沉积下伏层序</b><span>每个模型都有。其中一层可能尖灭（{p['probability']:.0%}）</span></li>
<li><b>侵蚀</b><span>{e['probability']:.0%}：侵蚀面以上沉积一套新层序</span></li>
<li><b>褶曲起伏</b><span>{u['probability']:.0%}：上下两套层序一起弯曲</span></li>
<li><b>区域倾斜</b><span>每个模型都有：整体倾斜到地层倾角</span></li>
<li><b>断层</b><span>{f['probability']:.0%}：错开前面所有结构</span></li>
<li><b>流体界面</b><span>{fl['probability']:.0%}：按现今深度水平分布</span></li>
</ol>
<div class="prose">
<p>生成时对每个网格单元按相反顺序倒推：先撤销断层位移，再撤销倾斜和起伏，找到这个点沉积时的位置和所属的层，再判断它是否落在尖灭让出的空间或侵蚀面以上，最后按现今深度套上流体界面。这样后发生的结构总是作用在先发生的结构上，例如断层会把已经褶曲、含尖灭的地层整体错开，而流体界面不随地层倾斜和断层变形。</p>
</div>
{figure(pipeline_png, "同一个模型按地质顺序逐步加入各结构", f"同一个模型按地质顺序逐步加入各结构，每一步使用同一套随机参数。为便于逐步对比，第 ① 步就画上了最终的倾角。横轴 y、纵轴 z（向下），单位 ft，原点在窗口中心。本例参数：{pipeline_params}。演示时五种结构全部打开，实际数据中每种结构按概率独立出现。全文所有模型图使用同一色标。")}
<div class="prose">
<p><b>全局约束</b>：所有单元的电阻率都在 {10 ** slo:.1f}–{10 ** shi:.0f} Ω·m 内，且单个模型窗口内最高与最低之比不超过 {ratio(strong[1])} 倍（流体界面的降低幅度按剩余空间抽取以满足这一条）；每种结构都必须在窗口内看得见（改变至少 {c['quality']['min_visible_cells']} 个单元，尖灭为 36 个），否则重新抽位置，多次失败则放弃这种结构；除厚层模型外，窗口内至少要有一条层界。</p>
</div>

<h2 id="s3"><span class="no">3</span>基础层序：层厚、电阻率、倾角</h2>
<h3><span class="no">3.1</span>层厚</h3>
<div class="prose">
<p>每个模型先抽一个层厚中位数，模型里各层的厚度再围绕这个中位数按对数正态分布抽取。这样不同模型之间有的是薄互层，有的是块状厚层；同一模型内各层厚度也有变化。层从窗口上方一直铺到下方，起点随机错开。</p>
</div>
{thickness_table}
{figure(figs['scale'], "不同层厚中位数的层序和竖直测井", "不同层厚中位数的基础层序（未加任何结构），右侧为沿橙色虚线的竖直电阻率曲线，相当于一口直井在该处测得的电阻率。")}
<div class="grid2">
{chart(thick_svg, f"train 中与窗口相交的所有层的厚度（垂直层面量），共 {thickness.size:,} 层。中位数 {t50:.1f} ft（{t50 * FT:.2f} m）；薄于 0.5 ft（15 cm）的占 {pct(np.mean(thickness < 0.5))}，薄于 1 ft 的占 {pct(np.mean(thickness < 1))}，厚于 1 m 的占 {pct(np.mean(thickness > 1 / FT))}。最右一列是被截断到 {t_hi:g} ft 的层；较厚的层更容易与窗口相交，所以这里厚层所占比例高于层序本身。")}
{chart(count_svg, f"每个模型窗口内能看到的层数：中位数 {n50:.0f} 层，10%–90% 分位 {n10:.0f}–{n90:.0f} 层。层数包括只露出一部分的层。")}
</div>

<h3><span class="no">3.2</span>电阻率</h3>
<div class="prose">
<p>每个模型先抽 2–4 个电阻率级别（下文称“岩性”），每一层从中取一个，并且与上一层不同；同一级别可以在层序中重复出现。这里的“岩性”只是电阻率级别，生成器不区分泥岩、砂岩或碳酸盐岩，层的先后顺序也是随机的。</p>
<p>最高与最低级别之比就是这个模型的最大对比度。实际数据中 2、3、4 种级别的模型分别占 {palette_sizes[2] / n_layered:.0%}、{palette_sizes[3] / n_layered:.0%}、{palette_sizes[4] / n_layered:.0%}（弱对比时级别间距太小，部分模型只能取 2 种）。</p>
</div>
{rho_table}
<div class="grid2">
{chart(rho_svg, f"train 中随机 2,000 个模型全部单元的电阻率分布（按面积）。低于 1 Ω·m 的占 {pct(area_bands[0])}，1–10 Ω·m 占 {pct(area_bands[1])}，10–100 Ω·m 占 {pct(area_bands[2])}，100 Ω·m 以上占 {pct(area_bands[3])}。")}
{chart(span_svg, f"每个模型窗口内最高与最低电阻率之比（厚层模型除外）。中位数 {ratio(np.median(spans))} 倍；低于 2 倍的占 {pct(np.mean(spans < math.log10(2)))}，超过 50 倍的占 {pct(np.mean(spans > math.log10(50)))}。")}
</div>

<h3><span class="no">3.3</span>地层倾角</h3>
<div class="prose">
<p>地层倾角是地层在过井竖直剖面上的视倾角，整个层序共用一个值，绕窗口内一个随机点倾斜。流体界面不随地层倾斜，始终水平。</p>
</div>
{dip_table}
{chart(dip_svg, f"train 中各模型的地层视倾角。{pct(dip_within)} 的模型在 ±{dp['gentle_abs_deg']:g}° 以内，最大 ±{dp['max_abs_deg']:g}°。", solo=True)}

<h2 id="s4"><span class="no">4</span>五种地质结构</h2>
<div class="prose">
<p>以下按地质先后顺序介绍。每组示例只打开对应的一种结构（其余结构、渐变层界和层内趋势都关闭），以便看清它本身的形态；示例标题是该模型抽到的主要参数。为了看得清，示例挑选了对比度较高、结构改变面积较大的模型；第 7 节是不经挑选的随机样本。</p>
</div>

<h3><span class="no">4.1</span>尖灭</h3>
<div class="prose"><p>某一层沿层面方向逐渐减薄直到消失，让出的空间由相邻层填充。收敛方式随机为三种之一。尖灭点可以在窗口内，也可以在窗口外，这时窗口里只看到减薄的一段。</p></div>
{pinchout_table}
{figure(figs['pinchout'], "尖灭示例", f"尖灭示例。train 中共 {len(pins):,} 个尖灭，尖灭点在窗口内的占 {pct(np.mean([q['tip_in_window'] for q in pins]))}；楔形角中位数 {np.median([q['wedge_angle_deg'] for q in pins]):.0f}°。")}

<h3><span class="no">4.2</span>侵蚀面</h3>
<div class="prose"><p>一条平缓起伏的侵蚀面截断下伏层序（连同其中的尖灭），面以上是一套重新抽取的上覆层序。上覆层序与下伏层序层理平行，侵蚀面本身与层理有小的夹角并带起伏，在图上表现为下伏层被一条缓起伏的面斜切。图中橙色虚线为侵蚀面。</p></div>
{erosion_table}
{figure(figs['erosion'], "侵蚀面示例", f"侵蚀面示例。train 中共 {len(erosions):,} 个侵蚀面，起伏幅度中位数 {np.median([abs(x['amplitude_ft']) for x in erosions]):.2f} ft，与层理夹角绝对值中位数 {np.median([abs(x['relative_tilt_deg']) for x in erosions]):.1f}°。")}

<h3><span class="no">4.3</span>褶曲起伏</h3>
<div class="prose"><p>整套层序（包括侵蚀面上下两套）一起上拱或下凹，代表小尺度褶曲或沉积起伏。{window_ft * FT:.2f} m 的窗口只能看到大尺度背斜或向斜的一小段，在窗口里表现为平缓弯曲的层界，所以不单独区分背斜、向斜。</p></div>
{undulation_table}
{figure(figs['undulation'], "褶曲起伏示例", f"褶曲起伏示例。train 中共 {len(undulations):,} 个模型有起伏，每个模型的最大幅度中位数 {np.median([max(abs(t['amplitude_ft']) for t in x) for x in undulations]):.2f} ft。")}

<h3><span class="no">4.4</span>断层</h3>
<div class="prose"><p>断层面一侧的地层沿断层面整体滑移，正断层和逆断层都有。有两条断层时，第二条再错开第一条之后的结果。</p></div>
{fault_table}
{figure(figs['fault'], "断层示例", f"断层示例。train 中共 {len(faults):,} 条断层，正断层占 {pct(np.mean([x['sense'] == 'normal' for x in faults]))}；倾角中位数 {90 - np.median([abs(x['tilt_from_vertical_deg']) for x in faults]):.0f}°，垂直断距中位数 {np.median([abs(x['throw_ft']) for x in faults]):.2f} ft（{np.median([abs(x['throw_ft']) for x in faults]) * FT * 100:.0f} cm）。")}

<h3><span class="no">4.5</span>流体界面</h3>
<div class="prose"><p>在一种较高阻的储层级别中，界面以下电阻率降低，代表油水界面。界面在现今坐标中水平，所以会斜切倾斜地层。界面以上可以有一段过渡带。图中橙色虚线为界面深度，过渡带在虚线之上。</p></div>
{fluid_table}
{figure(figs['fluid_contact'], "流体界面示例", f"流体界面示例。train 中共 {len(fluids):,} 个流体界面，界面以下降低倍数中位数 {ratio(np.median([x['drop_log10'] for x in fluids]))} 倍，过渡带厚度中位数 {np.median([x['transition_ft'] for x in fluids]):.2f} ft。")}

<h2 id="s5"><span class="no">5</span>层界与层内的细节</h2>
<h3><span class="no">5.1</span>渐变层界</h3>
<div class="prose"><p>相邻两层之间在一段宽度内逐渐过渡，而不是突变。下图为演示把概率提高到 60%。</p></div>
{gradational_table}
{figure(figs['gradational'], "渐变层界示例", "渐变层界示例。")}
<h3><span class="no">5.2</span>层内趋势</h3>
<div class="prose"><p>层内电阻率沿垂直层面方向线性变化。下图为演示把概率提高到 100%。</p></div>
{trend_table}
{figure(figs['trend'], "层内趋势示例", "层内趋势示例。")}
<h3><span class="no">5.3</span>厚层模型</h3>
<div class="prose"><p>窗口完全位于一个厚层中：没有层界，只有轻微的趋势。</p></div>
{thick_table}
{figure(figs['thick_bed'], "厚层模型示例", "厚层模型示例。")}

<h2 id="s6"><span class="no">6</span>结构的组合方式</h2>
<div class="prose">
<p>五种结构各自独立地按概率抽取，生成器没有设定结构之间的相关性，例如“断层 + 褶曲”同时出现的概率约等于两者概率之积。有的结构找不到合适的宿主层或在窗口内看不到时会被放弃，所以实际占比略低于设定概率。下表统计 train 中的 {n_layered:,} 个层状模型（不含 {n_train - n_layered:,} 个厚层模型）。</p>
</div>
<div class="grid2">
<div class="card"><div class="scroll" style="margin:0"><table>
<tr><th>结构</th><th class="r">设定概率</th><th>含该结构的模型</th><th class="r">实际占比</th><th class="r">其中只含这一种</th></tr>
{prob_rows}</table></div>
<p class="caption">“只含这一种”指五种结构中只有它（可以带倾角、渐变层界和层内趋势）。</p></div>
<div class="card"><div class="scroll" style="margin:0"><table>
<tr><th class="r">每个模型的结构数</th><th>模型数</th><th class="r">占比</th></tr>{count_rows}</table></div>
<p class="caption">{pct(n_struct.get(0, 0) / n_layered)} 的模型是没有结构的平面层序，{pct(multi)} 含两种及以上结构。</p></div>
</div>
<h3>最常见的组合</h3>
<div class="scroll"><table><tr><th>结构组合</th><th class="r">模型数</th><th class="r">实际占比</th><th class="r">按独立抽取的期望</th></tr>{combo_rows}</table></div>
<p class="caption">“按独立抽取的期望”用各结构的实际占比相乘得到，与实际占比一致，说明各结构之间没有相关性。</p>
{figure(combo_png, "常见组合示例", "常见组合的实际样本（取自 train，每种组合一例，最后一例含三种结构）。挑选条件：每种结构都明显改变了图像，且最大对比度大于 6 倍。")}

<h2 id="s7"><span class="no">7</span>典型结果</h2>
<div class="prose"><p>以下样本从 {n_train:,} 个训练模型中随机抽取，没有经过挑选，代表数据集的真实面貌。标题为模型含有的结构、地层倾角和窗口内最大对比度。</p></div>
{figure(gallery_png, "随机抽取的 24 个训练模型", "随机抽取的 24 个训练模型。")}
{figure(logs_png, "随机模型的竖直测井", "上图中前 8 个层状模型及其沿橙色虚线的竖直电阻率曲线。")}

<h2 id="s8"><span class="no">8</span>简化与未包含的内容</h2>
<ul class="limits">
<li><b>二维模型</b>：沿走向不变，对应 2.5 维正演。</li>
<li><b>电阻率各向同性</b>：没有水平和垂直电阻率之差。泥岩和薄互层的各向异性在实测数据中会表现为模型误差。</li>
<li><b>“岩性”只是电阻率级别</b>：层的先后顺序随机，只要求相邻层不同，没有沉积旋回、粒序或相序规律；同一模型中同一级别的各层电阻率只差 ±{(10 ** pr['layer_jitter_log10'] - 1) * 100:.0f}%。</li>
<li><b>层在横向上均匀</b>：除尖灭外，一个层沿层面方向电阻率不变，没有横向相变。</li>
<li><b>没有孤立地质体</b>：没有结核、透镜体、河道砂体等横向有限的地质体，所有高阻或低阻单元都是连续的层（或被尖灭、侵蚀、断层截断的层）。</li>
<li><b>层界和层内的变化形式简单</b>：层界只有尖锐和线性渐变两种；层内只有线性趋势，没有小尺度非均质，也没有噪声。</li>
<li><b>断层</b>：平面断层面，没有牵引构造和破碎带；断层两盘的流体界面深度相同，不考虑封闭断层造成的界面差异。</li>
<li><b>流体</b>：每个模型最多一个界面，只作用于一种电阻率级别；没有气油界面。</li>
<li><b>不整合</b>：上覆层序与下伏层序平行，不生成角度不整合。</li>
<li><b>窗口尺度</b>：窗口只有 {window_ft * FT:.2f} m，背斜、向斜等大尺度构造只能以平缓弯曲层界的形式出现。</li>
<li><b>结构之间相互独立</b>：没有地质上的相关性，例如断层附近更常见褶曲；各结构的出现概率和多数尺寸参数是工程设定。</li>
</ul>

<h2 id="s9"><span class="no">9</span>数据规模</h2>
<div class="prose"><p>各部分按同一方法独立生成，模型互不重复。两个外推测试集只用于检验先验在训练范围以外的表现，不参与训练。</p></div>
<div class="scroll"><table><tr><th>部分</th><th class="r">模型数</th><th>用途</th></tr>{split_rows}</table></div>

<h2 id="refs">参考文献</h2>
<ol class="refs">
<li>Chemali, R. (2009). Advanced geosteering with azimuthal deep resistivity helps in optimal well placement. AADE-NTCE-09-03.</li>
<li>Talling, P. J. (2001). On the frequency distribution of turbidite thickness. <i>Sedimentology</i>, 48, 1297–1329.</li>
<li>Shahriari, M., Pardo, D., Picón, A., Galdrán, A., Del Ser, J., Torres-Verdín, C. (2020). A deep learning approach to the inversion of borehole resistivity measurements. <i>Computational Geosciences</i>, 24(3), 971–994.</li>
<li>Hashemian, A., Garcia, D., Rivera, J. A., Pardo, D. (2021). Database generation for deep learning inversion of 2.5D borehole electromagnetic measurements using refined isogeometric analysis. <i>Computers &amp; Geosciences</i>.</li>
<li>Zhang, P., Deng, S., Yuan, X., Liu, F., Xie, W. (2025). Fast forward modeling and response analysis of extra-deep azimuthal resistivity measurements in complex model. <i>Frontiers in Earth Science</i>, 13, 1506238.</li>
<li>Anderson 断层理论（正断层倾角约 60°）；活动正断层实测倾角多为 30–60° 见低角度正断层与地震活动的综述。</li>
</ol>
<p class="meta">由 <code>reports/lwd_v4_generation_method_20261112.py</code> 根据 <code>datasets/lwd_v4</code>（生成器 {esc(manifest['generator_version'])}，配置 <code>configs/lwd_default.json</code>）生成。</p>
</main>
"""
    head = f'<title>{TITLE}</title>\n'
    page = (f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n{head}'
            f'<style>{CSS}</style></head><body>{content}</body></html>\n')
    OUT.write_text(page, encoding="utf-8")
    print(f"Report: {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    if args.fragment:
        args.fragment.write_text(f'{head}<meta name="claude:viewport-fit-sides" content="cover">\n<style>{CSS}</style>\n{content}',
                                 encoding="utf-8")
        print(f"Fragment: {args.fragment}")


if __name__ == "__main__":
    main()
