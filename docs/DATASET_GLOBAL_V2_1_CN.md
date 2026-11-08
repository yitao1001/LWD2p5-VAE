> 历史数据版本说明。文中旧训练命令已归档，当前训练请看 [TRAINING.md](../TRAINING.md)。

# 数据集 v2.1：移除独立矩形小块体

> 本文为历史记录。当前默认已升级为 [v2.2](DATASET_GLOBAL_V2_2_CN.md)，进一步移除透镜状异常，并加入背斜和斜断层错移背斜，使用新配比与新目录。

按 2026-11-06 的数据设计调整，当前生成器删除 `block` 家族及矩形掩膜分支。原先 5% 的块体配额分配给水平层状模型：`layered` 从 20% 调为 25%。其余家族为均匀介质、弯曲层、断层、尖灭、不整合、透镜和流体接触面，共 8 个家族，比例总和为 100%。

这里删除的是独立矩形异常体；`lens` 仍表示边缘连续尖灭的扁长透镜。该调整没有引入母模型裁剪方案，也没有证明剩余家族已覆盖真实储层分布。

## 版本与配置

- 生成代码：`lwd_vae/generation.py`，`GENERATOR_VERSION = "2.1.0"`。
- 默认配置：`configs/lwd_default.json`，是家族比例的唯一默认来源。
- 数据格式继续使用 `schema_version=2`；全局 y-z 坐标、水平背景、1 inch 网格、48×120 图像、log10ρ 参数化和固定归一化保持相同约定。
- manifest 记录 `generator_version=2.1.0` 和 `geometry_policy.isolated_block_bodies=false`，用于辨认新先验。
- 旧配置中的 `block` 键会触发迁移提示；应删除该键并重新分配权重，不能只把其数值改为 0。

## 数据与运行目录

| 目录 | 用途 |
|---|---|
| `datasets/smoke_global_v2_1` | 新版流程检查集，90 train + 18 val + 18 test + 18 ood_contrast，共 144 张 |
| `datasets/lwd_global_v2_1` | 新版正式数据的默认输出，20,000 + 2,000 + 2,000 + 1,000，共 25,000 张；需执行生成命令 |
| `runs/finetuning_smoke_global_v2_1` | 新版短流程默认输出 |
| `runs/finetuning_csem_global_v2_1` | CSEM 参考配置默认输出 |
| `runs/finetuning_inverse_scattering_global_v2_1` | 逆散射参考配置默认输出 |

```bash
# 输出目录已存在时拒绝覆盖；自行重生成时用 --output 指定新目录。
python generate_dataset.py --smoke --preview

# 生成正式规模数据，再检查训练配置。
python generate_dataset.py --preview
python train_vae.py --config configs/finetuning/csem_reference.json --check-only
```

默认生成路径、三份微调配置、历史基线入口的默认路径以及训练指南均已同步。旧数据、旧 checkpoint 与旧验证记录保留原名；它们对应旧先验。若改变训练数据，应从选定的初始权重重新开始，使用新的运行目录。

## 代码阅读与验证

先读 `FAMILIES` 和默认 `family_weights`，再读 `sample_model()` 的各家族分支，最后看 `_schedule()` 的整数配额分配和 `generate_dataset()` 的写盘过程。透镜分支现在只计算椭圆掩膜，不再共享矩形块体分支。

生成测试检查旧 `block` 配置和直接采样请求被拒绝，所有新 split 的元数据及统计均不包含该家族，同时检查数据范围、坐标方向、无重复、逐样本重放和失败清理。验证记录见 `docs/validation/20261106_no_blocks_tests.txt` 与根目录 `DATASET_GLOBAL_V2_1_VALIDATION.json`。
