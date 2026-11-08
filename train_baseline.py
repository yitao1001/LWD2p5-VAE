#!/usr/bin/env python3
"""历史 LWD 基线入口；新的论文参考微调请使用 train_vae.py。"""
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "datasets/lwd_global_v2_2")
    parser.add_argument("--output", type=Path, required=True, help="New run directory; never overwritten")
    parser.add_argument("--config", type=Path, help="Default: configs/train_default.json; omit for --resume")
    parser.add_argument("--device", default="auto", help="auto/cpu/mps/cuda/cuda:N")
    parser.add_argument("--epochs", type=int, help="Total epoch target, also when resuming")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--pretrained", type=Path, help="Local compatible kl-f8 state dict/ckpt/safetensors")
    initialization.add_argument("--resume", type=Path, help="Our last.pt (keep sibling best.pt); use a new --output directory")
    args = parser.parse_args()
    from lwd_vae.training import load_training_config, train
    if args.resume and args.config:
        parser.error("--resume restores the saved config; omit --config")
    config = None if args.resume else load_training_config(args.config or root / "configs/train_default.json")
    result = train(args.dataset, args.output, config, device_name=args.device, pretrained=args.pretrained,
                   resume=args.resume, epochs=args.epochs)
    print(f"Best checkpoint: {args.output.resolve() / result['best_checkpoint']}")


if __name__ == "__main__":
    main()
