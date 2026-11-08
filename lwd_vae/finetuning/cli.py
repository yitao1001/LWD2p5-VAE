"""命令行入口：先读配置和检查路径，再按需导入训练依赖。"""
from __future__ import annotations

import argparse
import json

from .config import DEFAULT_CONFIG, load_config, missing_inputs, resolve_paths, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="LWD VAE：CSEM 论文路线，预训练 KL-f8 微调")
    parser.add_argument("--config", help="项目根目录相对路径或绝对路径；默认 CSEM 参考配置")
    parser.add_argument("--dataset", help="覆盖数据集目录")
    parser.add_argument("--output", help="覆盖输出目录；必须为新目录")
    parser.add_argument("--pretrained", help="覆盖本地 CompVis VAE 权重路径")
    parser.add_argument("--alexnet-weights", help="覆盖本地 torchvision AlexNet 权重路径")
    parser.add_argument("--resume", help="续训 checkpoints/last.pt；必须指定新的 --output")
    parser.add_argument("--device", default="auto", help="auto/cpu/mps/cuda/cuda:N")
    parser.add_argument("--epochs", type=int, help="总训练轮数，续训时也表示总数")
    parser.add_argument("--allow-smoke-test", action="store_true", help="明确允许缩小网络/随机初始化的流程测试；不用于正式训练")
    inspect = parser.add_mutually_exclusive_group()
    inspect.add_argument("--print-config", action="store_true", help="仅显示配置及绝对路径，无需 PyTorch")
    inspect.add_argument("--check-only", action="store_true", help="严格加载权重、校验数据并执行前向，不训练")
    args = parser.parse_args(argv)
    try:
        if args.resume:
            if args.config or args.pretrained or args.alexnet_weights or args.print_config or args.check_only:
                parser.error("--resume 从 checkpoint 恢复完整配置，不能同时覆盖配置或初始权重")
            from .trainer import train
            result = train(resume=args.resume, output=args.output, dataset=args.dataset,
                           epochs=args.epochs, device_name=args.device)
            output = args.output
        else:
            config = load_config(args.config or DEFAULT_CONFIG)
            for argument, key in ((args.dataset, "dataset"), (args.output, "output"),
                                  (args.pretrained, "vae_weights"), (args.alexnet_weights, "alexnet_weights")):
                if argument is not None:
                    config["paths"][key] = argument
            if args.pretrained:
                config["initialization"]["mode"] = "pretrained"
            if args.epochs is not None:
                config["training"]["epochs"] = args.epochs
            config = validate_config(config)
            paths = resolve_paths(config)
            if args.print_config:
                print(json.dumps({"config": config, "resolved_paths": {
                    k: str(v) if v else None for k, v in paths.items()}}, indent=2, ensure_ascii=False))
                return 0
            if config["purpose"] == "smoke_test" and not args.allow_smoke_test:
                parser.error("测试配置必须显式传 --allow-smoke-test；正式微调请使用默认配置")
            missing = missing_inputs(config)
            if missing:
                raise FileNotFoundError("缺少必需输入：\n  " + "\n  ".join(missing) + "\n请参阅 TRAINING.md 和 weights/README.md。")
            if args.check_only:
                from .preflight import check_setup
                print(json.dumps(check_setup(config, args.device), indent=2, ensure_ascii=False))
                return 0
            from .trainer import train
            result = train(config, device_name=args.device)
            output = config["paths"]["output"]
        from .config import project_path
        print(f"Best checkpoint: {project_path(output) / result['best_checkpoint']}")
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        parser.exit(2, f"{error}\n")
