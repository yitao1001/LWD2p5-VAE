# 数据集 lwd_v1：与 ver8 反演网格一致的地质模型

> 2026-11-09 起已被 [lwd_v2](DATASET_V4_CN.md) 取代。本文件只作为 lwd_v1 训练与评估记录的说明；文中的 `check_ver8_grid.py` 和九类家族生成器已从代码中移除。

数据集命名从本版重新开始。`lwd_v1` 的图像网格就是 ver8 反演
`Test_GNinv_model01_ver8_batched_TRSmoothed_gpu_RealTool_DiscreteAdjoint.py`（默认配置
`configs/RealTool/config_dual_frequency_ver8_joint_wl2.yaml`）的反演网格：同样的单元、同样的顺序，不插值、不缩放。这与 CSEM 参考代码的做法一致——`VAE_generate_dataset.py` 在反演单元中心取值，`GN_2p5d_mCSEM_v5_VAE.py` 把 decoder 输出直接当作反演网格上的参考模型。

## 1. 网格

| 项 | 值 |
|---|---|
| 单元 | 0.5 inch 正方形，与 ver8 `min_step_inch=0.5`、`forward_step_multiplier=1` 一致 |
| 图像 | `[N,1,96,256]` float32，原始 `log10(ρ/Ωm)`，轴 0 为 z（向下），轴 1 为 y |
| 窗口（相对 ver8 地质原点 (255, 1780) ft） | y ∈ [-5.0625, 5.6041667]，z ∈ [-1.9791667, 2.0208333] ft，即 10.667 × 4 ft |
| 绝对坐标 | y ∈ [249.9375, 260.6041667]，z ∈ [1778.0208333, 1782.0208333] ft |
| 展平 | 图像按 Fortran 顺序（z 最快）展平得到 24,576 维向量，等于 ver8 的参数顺序 `iy*nz+iz` |
| KL-f8 潜变量 | 4 × 12 × 32 = 1,536 个 |

窗口两端各差 0.25 inch 的偏移来自 ver8 的 `coordinate_anchor: first_rx`：它把网格平移，使第一个接收器位于单元中心。所以窗口取决于 ver8 的测站位置、反演窗口和网格步长。改动这些设置后，必须重新运行：

```bash
/home/wytao/miniconda3/envs/lwd2p5/bin/python check_ver8_grid.py
```

它只调用 ver8 自身的 `load_config`、`build_base_config`、`build_inversion_window`，比较单元边界和每个参数的单元中心；不求解电磁场、不写文件。`compatible=false` 时不能用这批数据训练出的 VAE 直接对接该反演配置。

## 2. 九类地质与比例

地质家族和比例沿用上一版设计，只把网格换成 ver8 网格，并把层厚上限由 1.5 ft 放宽到 2.5 ft。

| 家族 | 比例 | 20,000 张训练集中的数量 | 结构 |
|---|---:|---:|---|
| homogeneous | 5% | 1,000 | 均匀介质 |
| layered | 25% | 5,000 | 水平层状 |
| curved | 10% | 2,000 | 连续缓弯层序 |
| anticline | 15% | 3,000 | 两翼宽度可不同的上拱层序 |
| faulted | 15% | 3,000 | 水平层序被倾斜断层错开 |
| faulted_anticline | 10% | 2,000 | 上拱层序沿斜断层滑移 |
| pinchout | 10% | 2,000 | 楔形尖灭：顶面水平、底面线性减薄到零 |
| unconformity | 5% | 1,000 | 侵蚀面截断较老层序 |
| fluid_contact | 5% | 1,000 | 储层内部流体接触面 |

train/val/test 采用同一比例；ood_contrast 排除均匀介质并重新归一化其余权重。弱/强物性对比独立于家族抽样。

## 3. 尖灭（楔形）家族

`sample_model()` 的 pinchout 分支：

```text
local_thickness(y) = T * clip(direction * (tip - y) / taper_length, 0, 1)
目标层：z_top <= z < z_top + local_thickness(y)，z_top = offset - T/2（顶面水平）
```

厚侧一直延伸到窗口边界，减薄段长 `taper_length`，在 `tip` 处厚度降为零。这与 ver8 合成真模型的楔形是同一种几何：ver8 为 T=2 ft、顶面 z=-1 ft、减薄段 3.75 ft、尖端 y=1.5 ft（相对原点）。

| 参数 | 抽样范围 |
|---|---|
| 厚部厚度 T（`bed_thickness_ft`，与 fluid_contact 储层厚度共用） | 0.25–2.5 ft |
| 尖端位置 tip | 窗口中心 −20% 至 +30% 窗宽 |
| 减薄长度 | 25%–70% 窗宽（约 2.7–7.5 ft） |
| 减薄方向 | ±y 各半 |
| 层中心深度 offset | 窗口中心 ±20% 窗高 |

ver8 合成模型的对比度（Δlog10ρ=0.1）偏低，不作为基准；实际训练对比度见下一节。

## 4. 电阻率

| 项 | 值 |
|---|---|
| 抽样范围 | 0.5–200 Ωm（log10 −0.301 至 2.301） |
| 固定归一化 | log10ρ ∈ [−1, 3] → [−1, 1]，全数据集共用，不逐张拉伸 |
| 弱对比（35% 的非均匀样本） | Δlog10ρ 0.05–0.25 |
| 强对比 | Δlog10ρ 0.3–1.5 |
| ood_contrast | Δlog10ρ 1.8–2.4，只用于独立评价 |

## 5. 背斜与斜断层

背斜：整组水平层界共用一个向上位移 `U(y) = A·exp(-0.5((y-c)/w_side)^2)`，左右翼宽度独立抽样；A 为 0.25–0.75 ft，单翼特征宽度 1.5–3.5 ft，轴部位于窗口中心 ±15% 窗宽。

斜断层错移背斜：断层线 `y = y_fault + (z - z_reference)·tan(alpha)`，|alpha| 为 20–45°，倾向随机；断层一侧沿断层面滑移，垂向断距 |throw| 为 0.25–1 ft，横向分量 `throw·tan(alpha)`。普通 `faulted` 家族为水平层序被 ±20° 断层垂向错断。

全程使用固定全局 y-z 坐标，不旋转图片。工具倾角是正演中的已知条件，不进入地质图像。

## 6. 规模、文件与运行

| split | 数量 |
|---|---:|
| train | 20,000 |
| val | 2,000 |
| test | 2,000 |
| ood_contrast | 1,000 |

```bash
# 流程检查数据：90 train + 各 18 张 val/test/ood_contrast → datasets/lwd_v1_smoke
.venv/bin/python generate_dataset.py --smoke --preview

# 正式数据 → datasets/lwd_v1（约 2.5 GB；已有目录会拒绝覆盖）
.venv/bin/python generate_dataset.py --preview
```

manifest 记录 `generator_version=1.0.0`、配置与生成器 SHA256、各 split 的家族和对比度计数；metadata.jsonl 记录每张图的几何参数与随机种子序列，可逐张重放。

- `configs/lwd_default.json`：唯一默认分布和网格来源。
- `lwd_vae/generation.py`：`validate_config()`、`sample_model()`、`generate_dataset()`。
- `check_ver8_grid.py`：网格与 ver8 反演网格的逐单元核对。

## 7. 验证范围

`tests/test_generation.py` 检查：网格为 0.5 inch、96×256 且边界等于 ver8 窗口；尖灭为顶面水平、厚度朝尖端单调减小的楔形，厚度上限覆盖 2 ft；背斜层界上拱、断层只移动一侧；精确配额、无重复、种子重放和原子写盘。`check_ver8_grid.py` 用 ver8 代码本身核对网格与参数顺序。这些是数据约定检查，VAE 重构质量和 LWD 响应误差需要另行评价。
