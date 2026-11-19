# 数据与展平顺序阅读指南

网格约定、地质组件和比例见 [数据集 lwd_v4](DATASET_V4_CN.md)；训练入口以根目录 TRAINING.md 为准。

## 数据加载与反演接口

```python
from lwd_vae.dataset import LWDVAEDataset, denormalize_log10rho
from lwd_vae.mapping import to_inversion_vector

ds = LWDVAEDataset('datasets/lwd_v4_smoke', split='train')
ds.validate_all()                 # 全量检查；普通读取按样本检查
u = ds[0]                        # [1,128,128]，归一化 float32
p = denormalize_log10rho(u, ds.log10_bounds)[0]
m = to_inversion_vector(p)        # [16384]，Fortran 顺序（z 最快）= 反演参数向量
ds.close()
```

PyTorch 使用 `LWDVAEDataset(..., as_torch=True)`，再传给 `torch.utils.data.DataLoader`。训练集 shuffle，验证集不 shuffle；使用 manifest 中固定的 split，不再随机划分。归一化反变换函数默认拒绝 [-1,1] 以外的输入；VAE decoder 输出无界，可能超界，训练/反演阶段必须明确约束策略，不能静默裁切。

数据网格就是计划中的正方形反演网格（1 inch、128×128、全局坐标、以窗口中心为原点），所以**直接按 F 顺序展平即可，不需要插值或单元映射**。反演程序再把参数复制到自己的正演网格并保留域外背景。

把 decoder 接入反演时，在 PyTorch 中保持 `p = lo + (u+1)*(hi-lo)/2` 和单张二维图像的 `p.transpose(-2,-1).reshape(-1)`，梯度即可从反演参数传回潜变量；`lwd_vae/inference.py` 的 `LWDVAE.decode_inversion_vector()` 已按此实现。数据阶段不把 Tx/Rx 邻域抹成背景，也不加入 PML：它们属于数值计算设置，不是地质先验。

`lwd_vae/mapping.py` 另有 `map_to_inversion_grid()`（单元中心双线性插值），只用于网格不同的导出/检查，不保留梯度；网格一致时不需要它。

## VAE 训练后先看什么

log10ρ 重构误差、界面位置/薄层与尖灭端保留、均匀场是否出现伪结构，以及按地质标签、对比度和倾角分组的误差（`test_vae.py` 输出）。之后抽少量模型比较原模型与重构模型的 LWD 正演响应。像素指标好不等于响应误差小。

## 文件

- `generate_dataset.py`：生成入口；`configs/lwd_default.json`：唯一默认参数来源。
- `lwd_vae/generation.py`：地质生成、验证、划分和写盘。
- `lwd_vae/dataset.py`：固定归一化、memmap 与可选 PyTorch 加载。
- `lwd_vae/mapping.py`：F 顺序转换。
- `preview_dataset.py`：数据预览；`tests/`：数据、模型、指标及训练流程检查。
- `docs/LITERATURE_AND_DESIGN.md`：文献与设计依据。
- `train_vae.py` / `test_vae.py`：训练与独立测试；`TRAINING.md`：完整说明。
- `lwd_vae/finetuning/`：训练框架；`lwd_vae/vendor/compvis/`：固定版本原始网络。
- `lwd_vae/weights.py` / `inference.py`：权重适配与可微反演接口。
- `weights/README.md`：外部权重位置；`references/`：来源审计用的原始材料。
