# 全局坐标数据生成器 v2.0 历史副本

这里保存 2026-11-06 删除小块体之前的生成代码与完整配置，便于核对和重放 `datasets/smoke_global_v2` 中的历史样本。两个文件均按原字节复制。

- `generation_original.py`：generator 2.0.0，包含旧 `block` 家族。
- `lwd_default_original.json`：旧家族比例。

当前入口使用 `lwd_vae/generation.py` 和 `configs/lwd_default.json`，版本为 2.1.0，已删除 `block`。本目录不作为当前训练的数据生成入口。
