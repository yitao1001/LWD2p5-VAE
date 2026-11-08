# 数据与映射阅读指南

以下保留 v2.2 数据设计细节；训练入口以项目根目录 TRAINING.md 为准。

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
- `docs/LITERATURE_AND_DESIGN.md`：论文、源码差异和设计依据。
- `docs/DATASET_GLOBAL_V2_2_CN.md`：背斜、斜断层错移、比例和路径的当前设计。
- `docs/DATASET_GLOBAL_V2_1_CN.md`：只移除矩形小块体时的历史说明。
- `docs/DATASET_GLOBAL_V2_CN.md`：v2.0 去掉随机整体倾斜时的历史说明。
- `train_vae.py` / `test_vae.py`：训练与独立测试；`TRAINING.md`：完整说明。
- `lwd_vae/finetuning/`：新训练框架；`lwd_vae/vendor/compvis/`：固定版本原始网络。
- `lwd_vae/weights.py` / `inference.py`：权重适配与可微反演接口。
- `weights/README.md`：外部权重位置；`references/`：原始材料；`docs/archive/`：历史说明。
