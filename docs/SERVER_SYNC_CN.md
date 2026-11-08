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

## 3. 生成 v2.2 正式数据

服务器已具备 NumPy 和 Matplotlib 的 Python 环境：

```bash
/home/wytao/miniconda3/envs/lwd2p5/bin/python generate_dataset.py --preview
```

默认读取 `configs/lwd_default.json`，写入 `datasets/lwd_global_v2_2/`：20,000 张训练、2,000 张验证、2,000 张测试和 1,000 张高物性对比度外推样本。分辨率为 48×120，物理窗口为 4×10 ft，单元为 1 inch。

输出目录已存在时，生成器会拒绝覆盖。生成另一批数据应使用新的目录，例如 `--output datasets/lwd_global_v2_2_run02`，并同步修改对应训练配置中的数据路径。数据生成只需要 CPU，不会启动 VAE 训练。

九类配比、几何参数和外推集规则见 [数据集 v2.2 说明](DATASET_GLOBAL_V2_2_CN.md)。每批数据的 `manifest.json` 保存实际类别数量、生成器版本和源码校验值，可用于核对代码与数据是否对应。

## 4. 关于测试数据

测试用的检查数据同样不随 Git 传输。新检出目录可执行 `python generate_dataset.py --smoke --preview` 生成当前版本检查数据。完整历史兼容性测试还使用旧版检查数据，需保留原有数据或使用 `references/dataset_v1/` 与 `references/dataset_global_v2/` 中的归档生成器重建；仅检出源码后直接运行全部测试，并不代表这些历史数据已经存在。
