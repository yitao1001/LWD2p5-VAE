#!/usr/bin/env python3
import argparse
from lwd_vae.preview import make_preview

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot one model per geological family")
    parser.add_argument("dataset")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    make_preview(args.dataset, args.split)
