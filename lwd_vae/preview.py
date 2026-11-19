"""First example of each geological tag, shared physical axes and color limits."""
import json
import os
from pathlib import Path

import numpy as np

from .generation import TAGS


def make_preview(root, split="train"):
    root = Path(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((root / "manifest.json").read_text())
    info = manifest["splits"][split]
    data = np.load(root / info["path"], mmap_mode="r")
    records = [json.loads(line) for line in (root / info["metadata"]).read_text().splitlines()]
    examples, used = {}, set()
    for tag in TAGS:  # a different sample per tag where possible
        candidates = [r for r in records if tag in r["tags"]]
        fresh = [r for r in candidates if r["index"] not in used]
        if candidates:
            examples[tag] = (fresh or candidates)[0]
            used.add(examples[tag]["index"])
    tags = [t for t in TAGS if t in examples]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.6), layout="constrained")
    g = manifest["config"]["grid"]
    color_bounds = manifest["config"]["resistivity"]["sampling_log10_bounds"]
    extent = [*g["y_bounds_ft"], *g["z_bounds_ft"][::-1]]
    for ax, tag in zip(axes.flat, tags):
        record = examples[tag]
        im = ax.imshow(data[record["index"], 0], extent=extent, origin="upper", aspect="equal",
                       vmin=color_bounds[0], vmax=color_bounds[1], cmap="viridis", interpolation="nearest")
        others = [t for t in record["tags"] if t != tag and t not in ("trend", "gradational")]
        ax.set_title(f"{tag}" + (f" (+{', '.join(others)})" if others else "")
                     + f"\n{record['contrast_class']} | dip {record['parameters']['dip_deg']:.1f}°", fontsize=10)
        ax.set_xlabel("y from window centre (ft)")
        ax.set_ylabel("z from window centre (ft), down")
    for ax in list(axes.flat)[len(tags):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$\log_{10}(\rho\,/\,\Omega\,m)$")
    fig.suptitle(f"LWD 2.5-D | generator {manifest['generator_version']} | {g['nz']}x{g['ny']} cells | {split}", fontsize=13)
    path = root / f"preview_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Preview: {path}")
    return path
