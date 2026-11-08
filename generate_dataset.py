#!/usr/bin/env python3
"""Generate isotropic LWD model images; no EM simulations or VAE training."""
from pathlib import Path
import argparse

from lwd_vae.generation import generate_dataset, load_config


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/lwd_default.json")
    parser.add_argument("--output", type=Path, help="New dataset directory; default depends on --smoke")
    parser.add_argument("--smoke", action="store_true", help="90 train + 18 val + 18 test + 18 ood_contrast")
    parser.add_argument("--preview", action="store_true", help="Also render a gallery (requires matplotlib)")
    args = parser.parse_args()
    if args.output is None:
        args.output = root / ("datasets/smoke_global_v2_2" if args.smoke else "datasets/lwd_global_v2_2")
    c = load_config(args.config)
    if args.smoke:
        c["splits"] = {s: 90 if s == "train" else 18 for s in c["splits"]}
    manifest = generate_dataset(c, args.output)
    print(f"Saved {sum(s['count'] for s in manifest['splits'].values())} models to {args.output.resolve()}")
    if args.preview:
        from lwd_vae.preview import make_preview
        make_preview(args.output)


if __name__ == "__main__":
    main()
