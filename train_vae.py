#!/usr/bin/env python3
"""LWD 主训练入口：原始 CompVis kl-f8 + 两种独立的参考微调配置。"""
from lwd_vae.finetuning.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
