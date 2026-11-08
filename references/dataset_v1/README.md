# 历史数据生成器 v1

`generation_original.py` 和 `lwd_default_original.json` 是切换到全局对齐水平背景先验前的未修改副本，SHA256 见 `provenance.json`。它们用于解释、重放已有 `datasets/smoke_v1` 等旧样本；新入口不导入此目录。

v1 在全局 y/z 坐标中加入随机地质整体倾角，并生成 ood_dip。v2 按当前任务删除该整体倾角，保留地质局部形态。固定坐标与水平背景是两个不同约定；详见 `docs/DATASET_GLOBAL_V2_CN.md`。
