# LWD 2.5D：KL-f8 VAE 微调

本项目用地质模型图像训练 VAE，供 LWD 2.5D 反演使用。思路迁移自 CSEM + VAE 反演（参考目录 `/home/wytao/projects/2026年3月4日_25DCSEM_代码发送`）：VAE 只学习模型空间的地质先验，不需要每张图的电磁响应；VAE 图像就是反演网格。

正式训练只有一条路线：**通用预训练 KL-f8 → LWD 单通道适配 → CSEM 论文形式的损失 → 全网络微调**。对齐依据为 [CSEM 论文 §II-B](https://arxiv.org/html/2601.02145v1#S2.SS2)。论文没有公开全部训练细节，本项目对齐方法主线；特征层、损失归约、优化器和数值参数的选择明确记录在配置中，不宣称逐项复现作者私有运行。

## 从哪里开始读

1. [TRAINING.md](TRAINING.md)：训练流程、每个文件的作用、关键参数与运行命令。
2. [configs/vae_finetune_csem.json](configs/vae_finetune_csem.json)：唯一正式训练配置。
3. [docs/DATASET_V4_CN.md](docs/DATASET_V4_CN.md)：数据集 lwd_v4 的网格约定、组合式地质生成器与各组件参数（面向地质审查的完整图文报告：`reports/lwd_v4_generation_method_20261112.html`）。
4. [docs/CSEM_ALIGNMENT_CN.md](docs/CSEM_ALIGNMENT_CN.md)：论文、CSEM 目录中脚本与本实现的对应关系。
5. [docs/DATASET_GUIDE_CN.md](docs/DATASET_GUIDE_CN.md)：数据数组、坐标及展平顺序。

## 目录

```text
train_vae.py                    正式微调、预检查和续训入口
prepare_pretrained.py           从官方来源准备 KL-f8 / AlexNet 权重
test_vae.py                     独立 test / OOD 评估
generate_dataset.py             地质模型生成
preview_dataset.py              数据集预览
configs/
  lwd_default.json             地质分布、网格与数据划分
  vae_finetune_csem.json        正式训练参数和路径
  pretrained_sources.json      官方权重地址、校验值
  testing/                     小网络流程测试、完整权重短流程检查
lwd_vae/
  generation.py / dataset.py  数据生成 / 固定归一化读取
  finetuning/                 训练逻辑
  weights.py                  完整权重检查和 RGB→单通道
  runtime.py                  保存、续训、随机状态、数据身份
  inference.py / mapping.py   反演接口 / 展平顺序
  evaluation.py              重构和分组误差
  vendor/compvis/             固定版本的原版网络计算代码
weights/                     大权重，仅本机文件，不进 Git
datasets/                    生成的数据，不进 Git
runs/                        每次训练独立目录，不进 Git
evaluations/                 独立评价结果，不进 Git
tests/                       自带临时数据的自动测试
references/                  CompVis 原文件、CSEM 与逆散射脚本快照（来源审计用）
docs/                        中文设计文档和验证记录
```

相对训练路径统一以项目根目录为基准。每个实验必须使用新输出目录。

## 快速开始

```bash
# 服务器在项目根目录使用 .venv/bin/python
python prepare_pretrained.py --check-only
python generate_dataset.py --preview            # → datasets/lwd_v4
python train_vae.py --check-only --device cuda:0
python train_vae.py --device cuda:0 --output runs/csem_kl_f8_lwd_v4

# 独立测试，不参与训练和模型选择
python test_vae.py --dataset datasets/lwd_v4 \
  --checkpoint runs/csem_kl_f8_lwd_v4/checkpoints/best.pt \
  --device cuda:0 --output evaluations/csem_kl_f8_lwd_v4 --save-reconstructions
```

只想检查流程、暂不使用大权重时：

```bash
python generate_dataset.py --smoke --preview    # → datasets/lwd_v4_smoke
python train_vae.py --config configs/testing/vae_smoke.json --allow-smoke-test \
  --device cpu --output runs/my_smoke
python -m unittest discover -s tests -v
```

## 数据约定

当前数据集为 `lwd_v4`（`datasets/lwd_v4`，共 46,000 张；层厚中位数 1–8 ft（lwd_v3 起），去掉了钙质结核（lwd_v4））：train 40,000、val 2,000、test 2,000，以及对比度外推 ood_contrast、倾角外推 ood_dip 各 1,000。每张保存 float32 `log10(ρ)`，单通道 `128×128`，1 inch 单元，10.67×10.67 ft 正方形窗口，全局 y-z 坐标（z 向下），网格不随工具倾角旋转。图像按 Fortran 顺序（z 最快）展平得到 16,384 维参数向量，新版正方形反演按此约定实现即可直接使用。地层倾角是图像中的地质参数，工具倾角只属于正演。固定归一化范围为 `log10ρ∈[-1,3] → [-1,1]`。

lwd_v1（96×256、0.5 inch，对应 ver8 网格）的设计与评估记录保留在 `docs/DATASET_V1_CN.md` 与 `docs/validation/`。

VAE 重构测试与实际 LWD 电磁响应/反演验证是不同阶段；反演接入尚未实现。
