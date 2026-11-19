"""Shared helpers for the self-contained HTML dataset reports (charts, CSS, figure encoding)."""
from __future__ import annotations

import base64
import html
import io
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE = ["#fde6dc", "#f9c6b0", "#f4a483", "#ef8458", "#eb6834", "#c9521f", "#a34116", "#7d300f", "#5a210a"]
CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE)
CMAP_STD = LinearSegmentedColormap.from_list("orange_seq", ORANGE)
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_families = ["DejaVu Sans"]
try:
    font_manager.fontManager.addfont(_CJK)
    _families.insert(0, font_manager.FontProperties(fname=_CJK).get_name())
except (OSError, RuntimeError):
    pass  # figure titles fall back to DejaVu Sans
plt.rcParams.update({"font.family": _families, "font.size": 8, "axes.unicode_minus": False,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#898781",
                     "ytick.color": "#898781", "axes.titlecolor": "#0b0b0b",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def png(fig, dpi=150):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def esc(value):
    return html.escape(str(value))


def nice_ticks(lo, hi, count=5):
    span = hi - lo
    raw = span / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=raw)
    start = math.ceil(lo / step - 1e-9) * step
    ticks, value = [], start
    while value <= hi + 1e-9 * span:
        ticks.append(round(value, 10))
        value += step
    return ticks


def fmt(value):
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"


def svg_histogram(edges, series, xlabel, ylabel, markers=(), width=520, height=250, aria="", xfmt=fmt, stacked=True,
                  xticks=None, vfmt=lambda n: f"{int(n):,} 个"):
    """Histogram; series = [(label, css_var, counts)]. Stacked by default; stacked=False overlays outlines.
    xticks: tick positions in edge units (default: nice ticks over the edge range); vfmt formats bar values in tooltips."""
    left, right, top, bottom = 56, 18, 30, 42
    pw, ph = width - left - right, height - top - bottom
    edges = np.asarray(edges, float)
    arrays = [np.asarray(c, float) for _, _, c in series]
    ymax = max(float((np.sum(arrays, axis=0) if stacked else np.max(arrays, axis=0)).max()), 1.0)
    yticks = nice_ticks(0, ymax * 1.05, 4)
    ytop = max(yticks[-1], ymax)
    x = lambda v: left + (v - edges[0]) / (edges[-1] - edges[0]) * pw
    y = lambda v: top + ph - v / ytop * ph
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}" class="chart">']
    for t in yticks:
        out.append(f'<line x1="{left}" x2="{left+pw}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
                   f'<text x="{left-6}" y="{y(t)+3:.1f}" class="tick" text-anchor="end">{fmt(t)}</text>')
    for t in (xticks if xticks is not None else nice_ticks(edges[0], edges[-1], 6)):
        out.append(f'<text x="{x(t):.1f}" y="{top+ph+14}" class="tick" text-anchor="middle">{xfmt(t)}</text>')
    base = np.zeros(len(edges) - 1)
    for (label, var, _), counts in zip(series, arrays):
        if not stacked:
            points = [f"{x(edges[0]):.1f},{y(0):.1f}"]
            for i, n in enumerate(counts):
                points += [f"{x(edges[i]):.1f},{y(n):.1f}", f"{x(edges[i+1]):.1f},{y(n):.1f}"]
            points.append(f"{x(edges[-1]):.1f},{y(0):.1f}")
            out.append(f'<polyline points="{" ".join(points)}" style="stroke:var({var})" class="step"><title>{esc(label)}</title></polyline>')
            continue
        for i, n in enumerate(counts):
            if n <= 0:
                continue
            x0, x1 = x(edges[i]) + 1, x(edges[i + 1]) - 1
            y0, y1 = y(base[i]), y(base[i] + n)
            w, h = max(x1 - x0, 0.8), y0 - y1
            r = min(3.0, w / 2, h)
            path = (f"M{x0:.1f},{y0:.1f} V{y1 + r:.1f} Q{x0:.1f},{y1:.1f} {x0 + r:.1f},{y1:.1f} "
                    f"H{x0 + w - r:.1f} Q{x0 + w:.1f},{y1:.1f} {x0 + w:.1f},{y1 + r:.1f} V{y0:.1f} Z")
            out.append(f'<path d="{path}" style="fill:var({var})" class="mark"><title>{esc(label)}: '
                       f'{xfmt(edges[i])}–{xfmt(edges[i+1])}，{vfmt(n)}</title></path>')
        base += counts
    out.append(f'<line x1="{left}" x2="{left+pw}" y1="{top+ph}" y2="{top+ph}" class="axis"/>')
    for value, label in markers:
        anchor, dx = ("end", -4) if x(value) > left + 0.7 * pw else ("start", 4)
        out.append(f'<line x1="{x(value):.1f}" x2="{x(value):.1f}" y1="{top-4}" y2="{top+ph}" class="marker"/>'
                   f'<text x="{x(value)+dx:.1f}" y="{top-10}" class="marker-label" text-anchor="{anchor}">{esc(label)}</text>')
    out.append(f'<text x="{left+pw/2}" y="{height-6}" class="axis-label" text-anchor="middle">{esc(xlabel)}</text>'
               f'<text transform="translate(14,{top+ph/2}) rotate(-90)" class="axis-label" text-anchor="middle">{esc(ylabel)}</text>')
    out.append("</svg>")
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background:var({var})"></i>{esc(label)}</span>' for label, var, _ in series) + "</div>"
    return legend + "".join(out)


def svg_window(grid):
    """Square window with two illustrative station layouts (12 RX + TX up to 32 inch)."""
    y0, y1 = grid["y_bounds_ft"]
    z0, z1 = grid["z_bounds_ft"]
    s, left, top = 30.0, 46, 16
    width, height = left + (y1 - y0) * s + 16, top + (z1 - z0) * s + 40
    X = lambda v: left + (v - y0) * s
    Z = lambda v: top + (v - z0) * s
    out = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="128×128 正方形窗口与示意测站" class="chart">']
    for v in range(math.ceil(y0), math.floor(y1) + 1):
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{Z(z0):.1f}" y2="{Z(z1):.1f}" class="grid"/>'
                   f'<text x="{X(v):.1f}" y="{Z(z1)+14:.1f}" class="tick" text-anchor="middle">{v}</text>')
    for v in range(math.ceil(z0), math.floor(z1) + 1):
        out.append(f'<line x1="{X(y0):.1f}" x2="{X(y1):.1f}" y1="{Z(v):.1f}" y2="{Z(v):.1f}" class="grid"/>'
                   f'<text x="{X(y0)-6:.1f}" y="{Z(v)+3:.1f}" class="tick" text-anchor="end">{v}</text>')
    out.append(f'<rect x="{X(y0):.1f}" y="{Z(z0):.1f}" width="{(y1-y0)*s:.1f}" height="{(z1-z0)*s:.1f}" class="window">'
               f'<title>VAE 图像 = 正方形反演窗口：128×128 个 1 inch 单元</title></rect>')
    for inclination, zc in ((90, -2.0), (75, 2.0)):
        a = math.radians(90 - inclination)
        start = -4.1
        for k in range(12):
            d = start + 0.5 * k
            out.append(f'<circle cx="{X(d*math.cos(a)):.1f}" cy="{Z(zc+d*math.sin(a)):.1f}" r="2.6" class="rx">'
                       f'<title>井斜 {inclination}°：RX {k+1}</title></circle>')
        for off in (16, 24, 32):
            d = start + 5.5 + off / 12
            out.append(f'<path d="M{X(d*math.cos(a)):.1f},{Z(zc+d*math.sin(a))-4:.1f} l4,7 h-8 Z" class="tx">'
                       f'<title>井斜 {inclination}°：TX{off}（末站）</title></path>')
        out.append(f'<text x="{X(start*math.cos(a))-4:.1f}" y="{Z(zc+start*math.sin(a))-8:.1f}" class="marker-label">井斜 {inclination}°（示意）</text>')
    out.append(f'<text x="{X(0):.1f}" y="{height-4:.0f}" class="axis-label" text-anchor="middle">y（ft，相对窗口中心）</text>'
               f'<text transform="translate(12,{Z(0):.1f}) rotate(-90)" class="axis-label" text-anchor="middle">z（ft，向下）</text></svg>')
    legend = ('<div class="legend"><span><i class="sw-window"></i>VAE 窗口 = 正方形反演窗口</span>'
              '<span><i style="background:var(--series-1)"></i>RX ×12（示意）</span>'
              '<span><i style="background:var(--series-3)"></i>末站 TX 16/24/32 inch</span></div>')
    return legend + "".join(out)


def bar_cell(value, maximum):
    return (f'<div class="inbar"><b style="width:{100*value/maximum:.1f}%"></b></div>'
            f'<span class="num">{value:,}</span>')


CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --good:#0ca30c; --critical:#d03b3b; --good-text:#006300; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --page:#0d0d0d; --surface-1:#1a1a19;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --good-text:#0ca30c; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--text-primary); font:15px/1.65 system-ui,-apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:28px; line-height:1.25; margin:0 0 6px; }
h2 { font-size:20px; margin:40px 0 10px; padding-top:8px; border-top:1px solid var(--border); }
h3 { font-size:16px; margin:24px 0 6px; }
.lede { color:var(--text-secondary); margin:0 0 20px; } .meta { color:var(--muted); font-size:13px; }
code { font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace; background:var(--surface-1); border:1px solid var(--border); padding:0 4px; border-radius:4px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:18px 0 8px; }
.kpi { background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
.kpi .v { font-size:26px; font-weight:650; line-height:1.2; } .kpi .l { color:var(--text-secondary); font-size:13px; }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin:12px 0; }
.plate { background:#fcfcfb; border-radius:8px; padding:6px; overflow-x:auto; } .plate img { display:block; max-width:100%; height:auto; margin:0 auto; }
.caption { color:var(--text-secondary); font-size:13px; margin-top:6px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--border); vertical-align:top; }
th { color:var(--text-secondary); font-weight:600; white-space:nowrap; }
td.r, th.r { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
td.barcell { min-width:150px; white-space:nowrap; }
td.param { white-space:nowrap; }
.inbar { display:inline-block; width:90px; height:8px; background:var(--grid); border-radius:4px; vertical-align:middle; margin-right:8px; overflow:hidden; }
.inbar b { display:block; height:100%; background:var(--series-1); border-radius:4px; }
.num { font-variant-numeric:tabular-nums; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }
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
ul.checks { list-style:none; padding:0; margin:0; } ul.checks li { padding:8px 0; border-bottom:1px solid var(--border); }
.badge { display:inline-block; min-width:64px; font-size:12px; font-weight:600; border-radius:999px; padding:1px 8px; margin-right:10px; text-align:center; }
.badge.ok { color:var(--good-text); border:1px solid var(--good); } .badge.bad { color:var(--critical); border:1px solid var(--critical); }
.sub { display:block; color:var(--text-secondary); font-size:13px; margin-left:84px; }
details summary { cursor:pointer; color:var(--text-secondary); font-size:13px; }
.window-svg { max-width:420px; }
ol.steps li { margin:4px 0; }
@media (max-width:520px) { .sub { margin-left:0; } .grid2 { grid-template-columns:1fr; } }
"""
