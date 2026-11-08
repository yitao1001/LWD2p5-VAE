# LWD 2.5D：KL-f8 VAE 微调

本项目用地质模型图像训练 VAE，供后续 LWD 反演使用。当前正式入口只有一条：**通用预训练 KL-f8 → LWD 单通道适配 → CSEM 论文形式的损失 → 全网络微调**。

对齐依据为 [CSEM 论文 §II-B](https://arxiv.org/html/2601.02145v1#S2.SS2)。论文没有公开全部训练细节，因此本项目对齐方法主线；特征层、损失归约、优化器和数值参数的选择明确记录在配置中，不宣称逐项复现作者私有运行。

## 从哪里开始读

1. [TRAINING.md](TRAINING.md)：中文训练流程、每个文件的作用、关键参数与运行命令。
2. [configs/vae_finetune_csem.json](configs/vae_finetune_csem.json)：唯一正式训练配置。
3. [docs/CSEM_ALIGNMENT_CN.md](docs/CSEM_ALIGNMENT_CN.md)：论文、目录中脚本与本实现的对应关系。
4. [docs/DATASET_GLOBAL_V2_2_CN.md](docs/DATASET_GLOBAL_V2_2_CN.md)：九类地质模型与比例。
5. [docs/DATASET_GUIDE_CN.md](docs/DATASET_GUIDE_CN.md)：数据数组、坐标及映射细节。

## 目录

```text
train_vae.py                    正式微调、预检查和续训入口
prepare_pretrained.py           从官方来源准备 KL-f8 / AlexNet 权重
test_vae.py                     独立 test / OOD 评估
generate_dataset.py             地质模型生成
preview_dataset.py              数据集预览
configs/
  lwd_default.json             地质分布、物理网格与数据划分
  vae_finetune_csem.json        正式训练参数和路径
  pretrained_sources.json      官方权重地址、校验值
  testing/vae_smoke.json       小网络流程测试，不能作为正式模型
lwd_vae/
  generation.py / dataset.py  数据生成 / 固定归一化读取
  finetuning/                 当前唯一正式训练逻辑
  weights.py                  完整权重检查和 RGB→单通道
  runtime.py                  保存、续训、随机状态、数据身份
  inference.py / mapping.py   反演接口 / 网格与展平顺序
  evaluation.py              重构和分组误差
  vendor/compvis/             固定版本的原版网络计算代码
  legacy/                    历史 AI 基线兼容模块
weights/                     大权重，仅本机文件，不进 Git
datasets/                    原始模型与 manifest，不进 Git
runs/                        每次训练独立目录，不进 Git
evaluations/                 独立评价结果，不进 Git
reports/                     HTML 报告，不进 Git
tests/                       自带临时数据的自动测试
references/                  原始材料和历史版本，不作为新训练入口
docs/                        中文设计文档、历史说明和验证记录
```

相对训练路径统一以项目根目录为基准。数据、权重、实验结果分别存放，不混进源代码目录；每个实验必须使用新输出目录。

## 快速开始

```bash
# 在项目根目录，先使用适合本机 CUDA / CPU 的 Python 环境
python -m pip install -r requirements-training.txt
python prepare_pretrained.py
python train_vae.py --print-config

# 服务器已有正式数据 datasets/lwd_global_v2_2，可直接检查
python train_vae.py --check-only --device cuda:0
python train_vae.py --device cuda:0 --output runs/csem_kl_f8_lwd_v1

# 独立测试，不参与训练和模型选择
python test_vae.py --dataset datasets/lwd_global_v2_2 \
  --checkpoint runs/csem_kl_f8_lwd_v1/checkpoints/best.pt \
  --device cuda:0 --output evaluations/csem_kl_f8_lwd_v1 --save-reconstructions
```

只想理解流程、暂不下载大权重时：

```bash
python generate_dataset.py --smoke --output datasets/my_smoke
python train_vae.py --config configs/testing/vae_smoke.json --allow-smoke-test \
  --dataset datasets/my_smoke --device cpu --output runs/my_smoke
python -m unittest discover -s tests -v
```

## 数据约定

当前正式数据为 `datasets/lwd_global_v2_2`，总计 25,000 张：train 20,000、val 2,000、test 2,000、OOD 对比度测试 1,000。每张保存 float32 `log10(ρ)`，单通道 `48×120`，覆盖竖向 4 ft、横向 10 ft，每个单元 1 inch。固定归一化范围为 `log10ρ∈[-1,3] → [-1,1]`。保持全局 y-z 坐标，不旋转地质模型来表示已知工具倾角。

**本次不改变地质生成逻辑、九类比例或正式数据文件。** VAE 重构测试与实际 LWD 电磁响应/反演验证是不同阶段；本项目当前尚未完成后者。
