# 本地、GitHub 与服务器同步

本地和服务器使用同一个仓库：<https://github.com/yitao1001/LWD2p5-VAE>，主分支为 `main`。

| 位置 | 项目目录 | 用途 |
|---|---|---|
| 本地 Mac | `/Users/yitao/Documents/2026LWD_Aramco/VAE_LWD2p5Inversion` | 修改代码和配置，提交后 push |
| 服务器 `wytao@101.6.69.149` | `/home/wytao/projects/VAE_LWD2p5Inversion` | pull 代码，生成数据，执行训练 |

Git 只同步源码、配置、测试与说明。`datasets/`、`runs/`、`evaluations/`、`reports/`、模型权重和本机环境不进入仓库。服务器首次同步前的旧源码单独备份；服务器独有文件与已有数据保留。

## 1. 本地修改后上传

在本地项目目录中检查变更，按实际修改的文件提交：

```bash
git status
git diff
git add configs/lwd_default.json lwd_vae/generation.py
git commit -m "Update LWD dataset configuration"
git push origin main
```

本地 `origin` 使用 SSH 地址 `git@github.com:yitao1001/LWD2p5-VAE.git`。服务器可使用 HTTPS 地址拉取公开仓库。

## 2. 服务器更新代码

```bash
ssh wytao@101.6.69.149
cd /home/wytao/projects/VAE_LWD2p5Inversion
git status
git pull --ff-only origin main
```

如果服务器也修改了源码，先检查并保存这些修改；不要用强制重置覆盖。`--ff-only` 会在分支分叉时停止，便于先处理差异。

## 3. 生成 lwd_v4 正式数据

```bash
cd /home/wytao/projects/VAE_LWD2p5Inversion
.venv/bin/python generate_dataset.py --preview
```

默认读取 `configs/lwd_default.json`，写入 `datasets/lwd_v4/`：40,000 张训练、2,000 张验证、2,000 张测试，以及对比度外推、倾角外推各 1,000 张。网格为 1 inch、128×128、10.67×10.67 ft 正方形全局窗口，数组约 3 GB，单进程生成约 8 分钟。输出目录已存在时，生成器会拒绝覆盖；另一批数据请用新目录并同步修改训练配置中的数据路径。数据生成只需要 CPU。

各组件参数和外推集规则见 [数据集 lwd_v4 说明](DATASET_V4_CN.md)。每批数据的 `manifest.json` 保存各标签的实际数量、生成器版本和源码校验值。

## 4. 关于测试数据

测试用的检查数据不随 Git 传输。自动测试在临时目录里自己生成所需数据；需要手工检查时执行 `python generate_dataset.py --smoke --preview`，写入 `datasets/lwd_v4_smoke/`。

## 5. 训练

入口为 `train_vae.py`，唯一正式配置为 `configs/vae_finetune_csem.json`。服务器独立 Python 环境为项目内 `.venv`（PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124），独立于原有 `lwd2p5` conda 环境。

```bash
cd /home/wytao/projects/VAE_LWD2p5Inversion
.venv/bin/python prepare_pretrained.py --check-only
.venv/bin/python train_vae.py --check-only --device cuda:0
.venv/bin/python train_vae.py --device cuda:0 --output runs/csem_kl_f8_lwd_v4
```

模型权重不走 Git。首次可通过 `prepare_pretrained.py` 下载，或从本地复制已校验文件至 `weights/vae/` 和 `weights/perceptual/`，再执行校验。
