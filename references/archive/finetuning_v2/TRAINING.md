# LWD VAE 训练与代码阅读指南

数据默认使用 [全局对齐的 v2.2](docs/DATASET_GLOBAL_V2_2_CN.md)：水平参考层序、局部背斜及斜断层错移、无随机整体旋转、无矩形或透镜状孤立异常，工具倾角留给正演。

当前主入口是 `train_vae.py`：使用固定版本的 CompVis 原始编码器、解码器和高斯后验，微调完整 kl-f8。它提供两条独立的参考路线。CSEM 论文未交代的细节，以及逆散射脚本与启动参数的矛盾，均明确列在 [方法对照与迁移假设](docs/METHODS_AND_ASSUMPTIONS_CN.md) 中。当前实现是迁移起点，尚不能称为论文的完整复现或已经验证有效的 LWD 方案。

## 1. 先看目录，再读代码

```text
VAE_LWD2p5Inversion/
├── train_vae.py                 # 新主入口，只负责调用 CLI
├── train_baseline.py            # 历史 LWD 基线入口
├── test_vae.py                  # 新旧 checkpoint 共用的独立评估入口
├── generate_dataset.py         # 只生成地质模型，不生成电磁响应
├── configs/
│   ├── lwd_default.json         # 数据分布、物理网格、训练/验证/测试划分
│   └── finetuning/
│       ├── csem_reference.json
│       ├── inverse_scattering_reference.json
│       └── smoke_original_core.json
├── lwd_vae/
│   ├── generation.py / dataset.py / mapping.py
│   ├── finetuning/
│   │   ├── config.py            # 严格校验参数；统一解析项目相对路径
│   │   ├── cli.py               # 命令行覆盖、预检查
│   │   ├── model.py             # 原网络的薄适配层，支持矩形图像
│   │   ├── features.py          # 冻结 AlexNet：特征 L2 / 官方 LPIPS
│   │   ├── losses.py            # 三项损失、归约方式、系数调度
│   │   └── trainer.py           # 初始化 → 训练 → 验证 → 保存 → 续训
│   ├── weights.py              # 预训练权重完整校验与 RGB→单通道转换
│   ├── runtime.py / inference.py # checkpoint、随机状态、反演接口
│   ├── vendor/compvis/          # 原始计算模块、固定 commit、哈希和 MIT 许可
│   └── model.py / training.py  # 历史基线实现，仅由旧入口调用
├── references/                 # 原始参考脚本，供核对，不作为训练入口
├── weights/vae/                # 放 kl_f8.ckpt
├── weights/perceptual/          # 放 alexnet_imagenet.pth
├── datasets/                   # 每个数据集独立目录与 manifest
├── runs/                       # 每次实验一个新目录
├── evaluations/                # 留出集评价，独立于训练输出
├── tests/                      # 行为、数值和端到端检查
└── docs/archive/               # 重构前说明，作为历史记录
```

建议阅读顺序：**配置 → dataset.py → model.py → features.py / losses.py → trainer.py → inference.py**。先看一张 `[1,48,120]` 图怎么进入网络，再看一次训练循环；不必一开始阅读所有第三方源码。`trainer.py` 按四个阶段写了中文注释。

新训练入口中的相对路径统一以**本项目根目录**为基准，包括配置文件、数据、初始权重、输出及续训文件。绝对路径可直接用于服务器。把项目复制到服务器后通常只需调整权重与数据路径。旧数据生成、测试、基线入口仍沿用工作目录相对路径，因此以下命令统一在项目根目录执行。

## 2. 建立环境，先运行短流程

Python >=3.10。建议独立环境；CUDA 的 PyTorch/torchvision 安装组合应按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择。其余依赖在 `requirements-training.txt`。项目不依赖旧 Lightning、W&B 或外部 ldm 安装。

```bash
cd /Users/yitao/Documents/2026LWD_Aramco/VAE_LWD2p5Inversion
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt

# 此命令只读配置，尚未安装 PyTorch 或准备权重也能运行。
python train_vae.py --print-config

# 附带 datasets/smoke_global_v2_2 时可直接使用；若需自己生成，使用新目录：
# python generate_dataset.py --smoke --output datasets/my_smoke --preview
python train_vae.py --config configs/finetuning/smoke_original_core.json --check-only
python train_vae.py --config configs/finetuning/smoke_original_core.json \
  --device cpu --output runs/finetuning_smoke_example

python test_vae.py --dataset datasets/smoke_global_v2_2 \
  --checkpoint runs/finetuning_smoke_example/checkpoints/best.pt \
  --device cpu --batch-size 2 --output evaluations/finetuning_smoke_example \
  --save-reconstructions

python -m unittest discover -s tests -v
```

smoke 配置显式从头初始化、32 初始通道、每级 1 个残差块，不使用图像特征损失，仅用 8 张训练图、4 张验证图训练 2 轮。这是在原始计算模块上缩小规模以检查流程，不能代表完整 kl-f8 的性能。正式配置均为 128 通道、每级 2 个残差块，强制加载预训练权重。

## 3. 准备正式微调

先按 [权重说明](weights/README.md) 准备两个**真实预训练**文件。没有权重时启动会一次列出缺失项，并停止，不会悄悄改成从头训练。`--check-only` 检查路径与数据；正式训练开始时还会严格加载全部权重，并用一张真实输入检查网络和损失。

```bash
# 正式规模共 25,000 张；不会覆盖已有目录。
python generate_dataset.py --output datasets/lwd_global_v2_2 --preview

python train_vae.py --config configs/finetuning/csem_reference.json --check-only
python train_vae.py --config configs/finetuning/csem_reference.json \
  --device cuda --output runs/finetuning_csem_s20261104_v1

python train_vae.py --config configs/finetuning/inverse_scattering_reference.json \
  --device cuda --output runs/finetuning_inverse_s20261104_v1
```

默认 batch=4、float32、10 epochs 是保守的**试验起点**，不是充分收敛的保证。CSEM 公开文本称微调在 10 steps 收敛，但这里不将其直接等同于明确的 10 epochs；本配置的轮数取自用户给出的逆散射启动参数。CUDA 显存允许时在复制的 JSON 中调 batch；`training.precision` 可设 `float16` 或 `bfloat16`。CPU/MPS 使用 float32。当前没有多卡 DDP。不能仅凭 10 轮结束就判断训练成功，应检查曲线、分组重构与物理响应。

运行名建议 `finetuning_方案_s随机种子_v版本`。每次改变参数复制一个配置文件并使用新的输出目录；不要反复覆盖同一个实验。配置拒绝未知字段，防止拼错参数无声失效。每次运行还会保存实际配置、权重 SHA256 和数据指纹。

若服务器文件位置不同，可直接覆盖路径：

```bash
python train_vae.py --config configs/finetuning/csem_reference.json \
  --dataset /absolute/data/lwd_global_v2_2 --pretrained /absolute/weights/kl_f8.ckpt \
  --alexnet-weights /absolute/weights/alexnet_imagenet.pth \
  --output /absolute/runs/csem_v1 --device cuda
```

## 4. 看训练输出，理解如何选择模型

```text
runs/某次实验/
├── run_config.json        # 实际参数、绝对路径、网络来源、权重/数据指纹
├── summary.json           # 完成轮数、最佳轮次、最佳验证指标
├── checkpoints/
│   ├── best.pt            # 验证集原始 log10ρ RMSE 最低的完整状态
│   └── last.pt            # 最近完整轮次，用于续训
└── logs/history.csv       # 每轮总损失、各原始/加权损失、系数、验证指标
```

CompVis 原模块初始化时会打印按其名义 resolution=128 计算的 z 形状；那不是本数据的实际张量。真实潜变量形状记录在 `run_config.json` 的 `posterior_shape`，本项目默认是 `[4,6,15]`。

训练时从后验采样；验证/测试用后验均值 `decode(encode(u).mode())`。验证指标在原始 `log10ρ` 单位计算，不裁切预测，并报告越界比例。仅使用 train 和 val，test/OOD 不用于选模型。`best.pt` 选择标准是本项目的明确工程约定。

关注日志的 `train_pixel`、`train_feature`、`train_kl` 及对应 `train_weighted_*`，确认三项对总损失的实际贡献。`train_kl_per_latent` 便于对比旧基线；正式参考路线的 `train_kl` 则是每张图潜变量求和后取 batch 均值。CSV 的 epoch 从 1 开始；checkpoint 内部从 0 开始。

逆散射配置在每个 epoch 末使用原始 CompVis `LitEma` 更新并回写参数，验证随后执行。这是上传代码的行为，**不是只在评估时临时换 EMA 权重**；优化器动量不随回写重置。原 `LitEma` 还按更新次数逐渐提高有效衰减率，因此配置的 0.99 是上限，不能理解成第一轮就固定 0.99。CSEM 配置默认禁用 EMA，因为尚未核实论文使用它。

## 5. 断点续训与留出集评价

```bash
python train_vae.py --resume runs/finetuning_csem_s20261104_v1/checkpoints/last.pt \
  --epochs 30 --device cuda --output runs/finetuning_csem_s20261104_v1_continue

python test_vae.py --dataset datasets/lwd_global_v2_2 \
  --checkpoint runs/finetuning_csem_s20261104_v1_continue/checkpoints/best.pt \
  --device cuda --output evaluations/finetuning_csem_s20261104_v1 \
  --save-reconstructions
```

`--epochs 30` 表示总共训练到第 30 轮。续训从 checkpoint 恢复完整配置、VAE、冻结特征网络、优化器、AMP、EMA 和随机状态，因此无需原始外部权重。不能同时指定 `--config` 或改变损失。保留同目录的 `best.pt`；若 last 并非历史最优，续训需要继承它。数据搬家可传 `--dataset`，但 manifest、训练/验证数组和元数据哈希必须一致。逐位一致仅针对相同软件、设备与数值环境。

测试入口按 manifest 选择已有留出集：v2 为 `test`、`ood_contrast`；旧 v1 还包含 `ood_dip`。它输出 `metrics.json`、每张样本指标、按家族/对比度分组的误差及可选重构图。不要用测试结果反复调参后仍将它称为独立测试。

## 6. 怎样选择迁移方案

先用相同数据划分、相同初始 kl-f8 权重和随机种子跑两条参考路线；它们整体设置不同，这一轮比较回答“哪套方案更适合 LWD”。若要判断改进究竟来自哪一项，第二轮只改像素损失或特征损失，并固定优化器、EMA、训练预算和模型选择标准。保留 `train_baseline.py` 作为历史对照。

优先检查薄层是否消失、层界偏移、弱对比是否被抹平、均匀场是否凭空出现结构，然后比较整体与分组 logρ 误差。选定候选后，必须用原 LWD 正演程序比较重构前后 Hyy/Hzz 响应，再决定是否进入反演。当前只提供可微 `LWDVAE.encode/decode/decode_inversion_vector`，尚未把 VAE 接入 GN 主循环，也没有证明反演精度得到改善。

历史基线说明和原实验记录位于 `docs/archive/`、`TRAINING_VALIDATION.json`；微调框架初版验证记录为 `FINETUNING_VALIDATION.json`；v2.0/v2.1 的记录为 `DATASET_GLOBAL_V2_VALIDATION.json` 和 `DATASET_GLOBAL_V2_1_VALIDATION.json`；当前加入背斜并移除透镜后的记录为 `DATASET_GLOBAL_V2_2_VALIDATION.json`。测试使用的随机合成权重只是为了检查代码行为，不能替代 ImageNet 或 kl-f8 预训练权重。
