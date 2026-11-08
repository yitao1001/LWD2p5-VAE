# LWD 2.5D VAE：数据生成与参考方案微调

这里提供 VAE 的**模型空间数据生成、训练与测试**。数据生成器只生成各向同性电阻率图像，不运行 LWD 正演，也不生成 Hyy/Hzz 标签。借用 CSEM 论文“先学习地质模型分布，再在物理反演中使用生成先验”的思路，将几何尺度、参数排列和模型家族适配到当前 LWD 项目。

**学习和运行请先读 [TRAINING.md](TRAINING.md)**。主入口现采用原始 CompVis kl-f8 的预训练微调，分别提供 CSEM 论文参考配置与逆散射代码参考配置。方法差异、未核实细节和选择建议见 [三套方案对照](docs/METHODS_AND_ASSUMPTIONS_CN.md)。旧 LWD 实现保留为 `train_baseline.py` 对照，不再作为默认路线。

用户上传的逆散射 `sdvae_model_finetune.py`、`kl_f8.yaml` 内容不变，已归档到 `references/inverse_scattering/`；新数据使用 `lwd_vae.dataset.LWDVAEDataset`。**不要直接把新数据交给旧 `FinetuneFaceData`**：它的 `np.resize` 会重复/截断元素，且归一化下界固定为 1，均不符合这里的数据约定。

当前数据生成采用 **v2.2：全局坐标对齐、水平参考层序、无随机整体旋转，加入背斜与斜断层错移背斜，移除矩形和透镜状孤立异常**。工具井斜由正演处理。变更、比例与几何公式见 [数据集 v2.2 说明](docs/DATASET_GLOBAL_V2_2_CN.md)。

## 快速运行

Python >=3.10；生成只需要 NumPy，预览需要 Matplotlib。当前机器已有可用环境 `/opt/miniconda3/envs/lwd_np/bin/python`。

```bash
cd /Users/yitao/Documents/2026LWD_Aramco/VAE_LWD2p5Inversion

# 独立环境安装时使用；当前 lwd_np 已具备这些依赖。
python -m pip install -r requirements-dataset.txt

# 小规模检查：90 train + 18 val + 18 test + 18 ood_contrast。
python generate_dataset.py --smoke --output datasets/my_smoke --preview

# 正式规模：20,000 train + 2,000 val + 2,000 test + 1,000 ood_contrast。
python generate_dataset.py --output datasets/lwd_global_v2_2 --preview

# 对已生成数据重新作图 / 检查。
python preview_dataset.py datasets/lwd_global_v2_2 --split ood_contrast
# 完整测试还需 requirements-training.txt 中的训练依赖
python -m unittest discover -s tests -v
```

正式规模共 25,000 张，原始 float32 数组约 549 MiB，另有元数据。通过 NPY memmap 顺序写入，不在内存同时建立全部图像。输出目录若已存在则拒绝覆盖；失败时清理本次临时目录，只在全部成功后发布数据集。示例集 `datasets/smoke_global_v2_2` 已生成并检查；它用于检查接口，不足以训练 VAE。

## 与当前反演的约定

以 `LWD2p5_Inversion-Hyy-Hzz/archive/Test_GNinv_model01_ver5_batched_TRSmoothed/config.yaml` 的手动默认窗口为依据（2026-10-15 只读检查）：

| 项目 | 约定 |
|---|---|
| 地质平面 | 全局对齐 y–z；窗口可平移，坐标轴不随工具旋转；沿 x 不变，z 向下 |
| 物理窗口 | 局部 y∈[-5,5] ft，z∈[-2,2] ft，即 3.048 m × 1.2192 m |
| 分辨率 | ny=120，nz=48；两方向均为 1 inch/cell |
| 文件形状 | `[N,1,nz,ny]`，即 `[N,1,48,120]`，float32 |
| 原始数值 | `p = log10(rho / (ohm m))`；不是线性ρ、σ或像素颜色 |
| 导出反演向量 | 单张 `(nz,ny)` 图像用 `ravel(order='F')`，z 最快 |
| 归一化 | `u = 2*(p-lo)/(hi-lo)-1`，全数据集固定 lo=-1、hi=3 |
| 物性 | 各向同性、单通道；不生成随后会被丢弃的ρv |

当前默认绝对坐标原点为 (y,z)=(255,1780) ft。与绝对网格映射时，先给源局部坐标加该原点，或给目标坐标减去原点，**必须统一参考系后再插值**。生成器不读取、修改或自动跟随原项目；若采用 `survey_adaptive` 或改变反演窗口，应相应修改本目录 JSON。不得把不同物理尺度的图像仅通过 resize 当作同一问题。

48 和 120 均能被 kl-f8 的 8 倍下采样整除，潜变量为 `[B,4,6,15]`。主模型直接使用固定版本的 CompVis 原始计算模块，并保留矩形网格；原包装器中强制 square latent 的分支不适用。

## 初版地质先验

以下比例和范围是**针对当前 LWD 窗口提出的工程配置**，不是文献中的统一标准。文献依据与已核实参数见 [LITERATURE_AND_DESIGN.md](LITERATURE_AND_DESIGN.md)。

| 家族 | 比例 | 实现 |
|---|---:|---|
| homogeneous | 5% | 均匀介质，避免网络默认每张图必须有异常 |
| layered | 25% | 1–4 个水平有序层界；两种物性端元交替 |
| curved | 10% | 对完整层序施加同一个正弦位移，层界不相交 |
| anticline | 15% | 整组连续层界上拱；左右翼宽度独立抽样，允许偏斜 |
| faulted | 15% | 倾斜断层面一侧的完整层序发生垂向错移 |
| faulted_anticline | 10% | 连续背斜被斜断层切开；沿断层面同时发生水平和垂向错移 |
| pinchout | 10% | 层厚随 y 连续减小到零；左右尖灭均可 |
| unconformity | 5% | 起伏侵蚀面截断较老水平层序，上覆第三种物性 |
| fluid_contact | 5% | 水平储层内水平流体界面，上高阻、下低阻 |

核心模型取值范围为 **0.5–200 Ωm**，物性端元在 log 域抽样；归一化边界为 0.1–1000 Ωm，留有余量。35% 非均匀模型抽取弱端元对比 `Δlog10ρ∈[0.05,0.25]`，其余为 `[0.3,1.5]`；弱对比范围包含现有 10 vs 12.59 Ωm 的差值 0.1。这不是每个界面都固定同一对比：第三种物性、裁切会改变可见对比；元数据同时保存端元差和图像实际差值。

参考层序水平，不采样随机整体倾角。普通断层面相对垂直方向 ±20°；断层背斜的断层面取 ±20–45°，两者垂向断距均为 0.25–1 ft。背斜上拱幅度 0.25–0.75 ft，左右翼特征宽度各为 1.5–3.5 ft，轴部在窗口中心左右 1.5 ft 内抽样；这些是当前 10×4 ft 窗口的工程参数。目标层厚度参数为 0.25–1.5 ft；一般层状模型通过独立的有序层界控制间距。尖灭尖端允许变薄至消失，不能把其最小局部厚度解释成 0.25 ft。所有图像取单元中心值，阶梯边缘是网格离散化，不对界面做图像模糊。

v2.2 共 9 个家族，总比例为 100%；背斜与断层背斜合计 25%。`block` 和 `lens` 及对应生成分支均已删除，旧配置含这两个键时会报迁移提示，即使权重为 0。旧样本保留为历史数据；当前训练配置指向新版目录。

当前模型族不是全面地质模拟：大多数家族是两种物性端元，无层内随机纹理、双断层、各向异性、井眼或侵入带。先用这组可解释先验验证重构与物理响应，再根据真实研究模型扩大；更复杂不自动等于更合适。

## 划分、质量检查与外推测试

- 每个样本从独立随机流生成完整地质实现；没有先造母模型再把相邻裁块分到不同集合。固定 seed、split id、sample index 和重试次数可以精确重放。
- 家族比例按整数配额分配，集合内随机打乱。train/val/test 来自相同先验；额外 OOD 集不参与训练或调参。
- v2 取消 `ood_dip`；旧数据中的该划分仅作为历史倾角先验的测试。全局固定坐标仍允许真实地质结构存在局部斜率。
- `ood_contrast`：端元差为 1.8–2.4，且**实际保存图像的** `max(p)-min(p) >= 1.8`，不满足则重抽。该 OOD 集排除没有物性对比度的均匀模型。
- 检查 shape、dtype、有限值、范围和非均匀图像的少数物性面积（至少 1%）；全数据集 SHA256 去除完全相同图像。随机几何参数独立不等于离散几何掩膜严格互斥，不能将这组测试宣称为“未见过任何相似构型”。
- `manifest.json` 保存完整配置、代码/配置 SHA256、坐标单位、每个 split 的数量、家族/对比统计与均值范围；JSONL 保存每个样本的参数和 hash。

## 数据加载与后续反演接口

```python
from lwd_vae.dataset import LWDVAEDataset, denormalize_log10rho
from lwd_vae.mapping import to_inversion_vector

ds = LWDVAEDataset('datasets/smoke_global_v2_2', split='train')
ds.validate_all()                 # 全量检查；普通读取按样本检查
u = ds[0]                        # [1,48,120]，归一化 float32
p = denormalize_log10rho(u, ds.log10_bounds)[0]
inversion_vector = to_inversion_vector(p)  # [5760]，Fortran顺序
ds.close()
```

PyTorch 使用 `LWDVAEDataset(..., as_torch=True)`，再传给 `torch.utils.data.DataLoader`。训练集 shuffle，验证集不 shuffle；使用已有固定 split，不让旧 DataModule 再随机划分。归一化反变换函数默认严格拒绝 [-1,1] 以外的输入；实际无界 VAE decoder 可能超界，训练/反演阶段必须明确约束策略，不能静默裁切。

`map_to_inversion_grid` 显式接收源/目标 y、z 边坐标，以单元中心双线性插值映射 logρ；目标域不能超出源域，源域外半单元内使用最近中心值。它不是守恒平均，也不是将粗模型复制到细正演单元的既有 `expand_inversion_parameters`。**相同默认网格直接 F 展平即可，不需要插值**；后续仍由原反演的扩展算子填充正演网格，域外背景另行指定。

这些 NumPy 映射用于导出/检查，不用于自动微分链。将 decoder 引入 GN 或 latent 优化时，在 PyTorch 中保持 `p=lo+(u+1)*(hi-lo)/2` 和单张二维图像的 `p.transpose(-2,-1).reshape(-1)`；不同网格要建立可微插值算子。数据阶段不把 Tx/Rx 邻域抹成背景，也不加入 PML：它们属于数值计算设置，非地质先验。

VAE 训练完成后建议先看：logρ 重构误差、边界位置/薄层保留、均匀场是否产生伪结构，以及按家族和对比度分组的误差。新增测试入口输出这些分组重构指标；之后仍应抽少量模型比较重构前后 Hyy/Hzz 的复数响应差。像素指标好不等于测量响应误差小，本项目的新训练/测试流程尚不包含 EM 正演评价，不宣称已改善反演。

## 文件

- `generate_dataset.py`：生成入口；`configs/lwd_default.json`：唯一默认参数来源。
- `lwd_vae/generation.py`：地质生成、验证、划分和写盘。
- `lwd_vae/dataset.py`：固定归一化、memmap 与可选 PyTorch 加载。
- `lwd_vae/mapping.py`：物理坐标映射与 F 顺序转换。
- `preview_dataset.py`：数据预览；`tests/`：数据、模型、指标及训练流程检查。
- `LITERATURE_AND_DESIGN.md`：论文、源码差异和设计依据。
- `docs/DATASET_GLOBAL_V2_2_CN.md`：背斜、斜断层错移、比例和路径的当前设计。
- `docs/DATASET_GLOBAL_V2_1_CN.md`：只移除矩形小块体时的历史说明。
- `docs/DATASET_GLOBAL_V2_CN.md`：v2.0 去掉随机整体倾斜时的历史说明。
- `train_vae.py` / `test_vae.py`：训练与独立测试；`TRAINING.md`：完整说明。
- `lwd_vae/finetuning/`：新训练框架；`lwd_vae/vendor/compvis/`：固定版本原始网络。
- `lwd_vae/weights.py` / `inference.py`：权重适配与可微反演接口。
- `weights/README.md`：外部权重位置；`references/`：原始材料；`docs/archive/`：历史说明。
