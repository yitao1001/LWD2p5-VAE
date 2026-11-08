# 数据集 v2：固定全局方向，去掉随机整体倾斜

> 本文记录 generator 2.0.0 的历史设计。当前默认已升级为 [v2.2](DATASET_GLOBAL_V2_2_CN.md)，删除矩形及透镜状孤立异常，加入背斜及斜断层错移，并使用新配比与新目录；下文仅用于解释历史样本。

本次按用户选择，将 VAE 数据统一描述为全局地质坐标的局部窗口：横轴 y，竖轴 z 向下，x 方向地质不变。局部窗口可沿全局坐标平移，但坐标轴不随工具旋转。工具井斜是正演阶段已知的测量条件，数据生成器不再采样它或借整体地层旋转来模拟它。

要区分两个独立选择：固定全局坐标本身允许真实倾斜地层；本版本进一步按当前任务去掉随机整体倾角，以水平背景层序作为训练先验。若以后确实需要区域性倾斜地层，应将它作为独立的地质参数重新引入，不能和工具井斜混用。

## 1. 哪些生成规则改变了

| 部分 | v1 | v2 |
|---|---|---|
| 层序整体倾角 | 默认随机 ±0–15° | 去掉，水平背景 |
| 透镜、块体整体方向 | 跟随随机倾角 | 透镜长轴沿 y；块体边缘沿 y/z |
| 含流体界面的储层 | 整体倾斜，流体面水平 | 储层和流体面均水平 |
| 不整合 | 倾斜老层被侵蚀面截断 | 水平老层被起伏侵蚀面截断，属于侵蚀不整合 |
| 弯曲层、断层、尖灭 | 随机整体倾斜叠加局部结构 | 仅保留显式的局部地质结构 |
| 倾角外推集 | ood_dip | 取消 |
| 对比度外推集 | ood_contrast | 保留，仍检查实际保存图像的对比度 |

断层面的倾斜是断层本身的几何参数，因此 `fault_tilt_from_vertical_deg` 保留。弯曲界面、尖灭边缘和透镜边缘自然存在局部斜率，这不表示整幅模型被随机旋转。所有家族都直接在单元中心采样，没有图片旋转、缩放或旋转后填充背景。

## 2. 配置、数组与命名

唯一默认生成配置仍为 `configs/lwd_default.json`，显式声明：

```json
{
  "schema_version": 2,
  "coordinate_frame": "global_aligned_yz",
  "geometry": {"background_bedding": "horizontal"}
}
```

以上是字段节选，不是完整可运行配置。旧 `dip_abs_deg`、`ood.dip_abs_deg` 和 `ood_dip` 字段已从生成器删除；带旧字段的配置会报错，防止静默忽略参数。

网格仍为 `[N,1,48,120]` float32 NPY，窗口 10×4 ft，步长 1 inch。存储值为原始 log10ρ，加载时按固定 [-1,3] 范围映射到 [-1,1]。物性分布、家族比例和单位保持原有约定。

新 manifest 的 schema 为 2，包含坐标方向与几何策略。每张图的元数据明确记录 `overall_rotation_deg=0` 和 `background_bedding=horizontal`，并保留地质形态参数、随机种子、重放次数与哈希。数据加载器校验坐标约定；训练 checkpoint 记录坐标系以及数据指纹。

| 数据目录 | 数量 | 用途 |
|---|---:|---|
| datasets/smoke_global_v2 | 90 train + 18 val + 18 test + 18 ood_contrast = 144 | 已生成的流程检查数据 |
| datasets/lwd_global_v2 | 20,000 + 2,000 + 2,000 + 1,000 = 25,000 | 正式生成命令的默认目标；需运行生成命令 |
| datasets/smoke_v1 | 162 | 保留的历史数据，含随机整体倾角，不是新默认训练输入 |

去掉原来的 1,000 张倾角外推后，正式规模由 26,000 变为 25,000，训练集 20,000 张不变。新版本不表示已经覆盖 0–180° 工具姿态所需的全部空间窗口；不同井斜的覆盖范围仍须在正演/反演端单独检查。

## 3. 如何运行与学习

在项目根目录激活 `.venv` 后：

```bash
# 已附带 smoke_global_v2；需要自己重生成时使用新目录，避免覆盖。
python generate_dataset.py --smoke --output datasets/my_global_smoke --preview

# 正式数据：默认写 datasets/lwd_global_v2。
python generate_dataset.py --preview

# 三份新微调配置已分别指向 v2 正式/小样本目录。
python train_vae.py --config configs/finetuning/smoke_original_core.json \
  --device cpu --output runs/finetuning_global_learning

python test_vae.py --dataset datasets/smoke_global_v2 \
  --checkpoint runs/finetuning_global_learning/checkpoints/best.pt \
  --device cpu --batch-size 2 --output evaluations/finetuning_global_learning
```

阅读 `lwd_vae/generation.py` 时，先看 `sample_model()`：`strat=z-offset` 建立水平参考层序，各家族分支再添加自己的结构；lens/block 的局部变量只减去中心，不使用旋转矩阵。随后看 `generate_dataset()` 的独立随机流、质量检查和 manifest 保存。

旧版生成器与原配置的未修改副本存于 `references/dataset_v1/`，便于解释旧样本。旧数据和旧 checkpoint 不被覆盖；数据改变后不能把原 checkpoint 当作同一数据上的精确续训。若要比较两种数据先验，应从相同的真实预训练权重重新开始，并使用新的运行目录。

## 4. 验证范围

本次检查水平层界、轴对齐块体、局部地质形态保留、平移不改变方向、逐样本重放、无重复样本、坐标标记校验、旧数据兼容，以及新数据的训练和独立评估入口。结果记录在根目录 `DATASET_GLOBAL_V2_VALIDATION.json`。

这些是代码与数据约定的验证，不证明水平背景先验最适合所有实际地层，也不代表完整 LWD 电磁反演效果已经验证。
