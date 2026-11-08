"""One example per geological family, shared physical axes and color limits."""
import json
import os
from pathlib import Path

import numpy as np


def make_preview(root, split="train"):
    root = Path(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((root / "manifest.json").read_text())
    info = manifest["splits"][split]
    data = np.load(root / info["path"], mmap_mode="r")
    examples = {}
    for line in (root / info["metadata"]).read_text().splitlines():
        record = json.loads(line)
        examples.setdefault(record["family"], record)
    # 使用数据本身的家族列表，使历史数据的预览也能完整显示。
    families = [f for f in manifest["config"]["family_weights"] if f in examples]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), layout="constrained")
    g = manifest["config"]["grid"]
    color_bounds = manifest["config"]["resistivity"]["sampling_log10_bounds"]
    extent = [*g["y_bounds_ft"], *g["z_bounds_ft"][::-1]]
    for ax, family in zip(axes.flat, families):
        record = examples[family]
        im = ax.imshow(data[record["index"], 0], extent=extent, origin="upper", aspect="equal",
                       vmin=color_bounds[0], vmax=color_bounds[1], cmap="viridis", interpolation="nearest")
        ax.set_title(f"{family.replace('_', ' ').title()} | {record['parameters']['contrast_class']}", fontsize=11)
        ax.set_xlabel("Global-aligned y offset (ft)")
        ax.set_ylabel("Global-aligned z offset (ft), down")
    for ax in list(axes.flat)[len(families):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$\log_{10}(\rho\,/\,\Omega\,m)$")
    policy = "horizontal background, no overall rotation" if manifest.get("schema_version") == 2 else "legacy geological dip prior"
    fig.suptitle("LWD 2.5-D | " + policy + " | " + split, fontsize=14)
    path = root / f"preview_{split}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Preview: {path}")
    return path
