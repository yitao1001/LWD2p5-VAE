# CSEM 对齐范围与来源审计（2026-11-08）

## 采用哪套逻辑

本次依据 [Zhou 等的 CSEM 论文 §II-B，式 (12)–(13)](https://arxiv.org/html/2601.02145v1#S2.SS2) 统一正式训练路线。用户已确认：**从通用 KL-f8 预训练权重开始，用 LWD 数据微调**。

现有文件无法证明作者最终实验的全部实现细节。不能把“同目录有一个脚本”当成“它就是发表版本”，也不能把逆散射方案重新命名为 CSEM。

| 方面 | 可核实的 CSEM 方法 | 当前 LWD 实现与限制 |
|---|---|---|
| VAE | 预训练 KL-f8，约 86M 可训练参数 | 使用固定版本 CompVis 原始 Encoder/Decoder，完整加载官方原版 KL-f8；实际可训练参数 83,649,253 |
| 单通道适配 | 对 RGB 输入/输出权重沿通道取均值 | 同样处理首层、末层及末层偏置 |
| 微调范围 | 微调 VAE | Encoder、Decoder、quant_conv、post_quant_conv 全部训练 |
| 像素项 | L2 重构 | 逐元素 MSE；论文未说明求和/均值的代码实现 |
| 特征项 | 冻结预训练图像网络，例如 AlexNet；特征 L2 | ImageNet AlexNet 五层 ReLU MSE，等权平均；具体层与标准化为显式选择 |
| KL | 后验相对标准正态的正则 | 每样本 latent 元素求和，再对 batch 平均 |
| 优化器、系数、调度 | 现有论文材料不足以恢复完整设置 | SGD、lr=1e-3 余弦衰减到 1e-5、γ=1、β 见配置、float16、10 epochs；lr 与 γ 由 lwd_v1 小规模筛选选定（`docs/validation/20261108_lwd_v1_pilot.json`），β 由 lwd_v2 筛选复核 |
| EMA / LPIPS / TV | 论文损失公式不能支持直接加入这些操作 | 正式路线不含这些附加项 |
| 训练数据 | CSEM 地质模型先验，图像即反演网格（`VAE_generate_dataset.py` 在反演单元中心取值） | LWD `lwd_v4`：图像即计划中的正方形反演网格，1 inch、128×128，F 顺序展平即反演参数顺序 |
| 验证与选模 | 作者实现细节未完全公开 | 固定 val，后验均值重构，按原始 log10ρ RMSE 选 best |
| 冻结参数统计 | 论文另报约 17M 非训练参数 | 本实现仅保留所选 AlexNet 卷积特征，冻结参数 2,469,696；不实例化原 checkpoint 的无用损失/判别器子网，不能宣称与论文参数统计逐项一致 |

官方原始 KL-f8 与后来发布的 `sd-vae-ft-ema` / `sd-vae-ft-mse` 是不同 checkpoint。后两者在其他图像集继续微调过 Decoder，见 [Stability AI 模型说明](https://huggingface.co/stabilityai/sd-vae-ft-mse-original)。本项目默认选 [原始 KL-f8](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip)，不把三个名称当作同一组权重，也不声称已获作者私有 checkpoint 的哈希。

## CSEM 文件夹中实际存在的其他代码

参考目录：`/Users/yitao/Documents/2026LWD_Aramco/2026年3月4日_25DCSEM_代码发送`。

- `VAE_mCSEM_train_v1/v2/v3.py` 是 TensorFlow 历史网络，包含从头初始化；相关 `GN_..._VAE.py` 导入 `Neural_Model_v2`。这不等于论文所述 KL-f8 微调。
- 该目录的 `sdvae_model_finetune.py` 与用户另传的逆散射脚本在训练主体上高度相近。它实际使用 L1 + LPIPS + KL；不能据此将它认定为 CSEM 论文式 (13) 的最终实现。
- 该脚本第 430 行 `vae_finetune=False` 会跳过权重加载，第 435 行把 `lpips_loss_weight` 接到 `args.kl_weight`，且 `np.resize` 与固定数值下界不适合 LWD。当前实现不复制这些行为。

目录脚本原样保存在 [references/csem_directory_snapshot](../references/csem_directory_snapshot/README.md)，只用于审计，不作为当前训练入口。用户另传的逆散射材料仍在 `references/inverse_scattering/`，来源保持分开。

## 当前整理（2026-11-08）

- 正式配置收敛到 `configs/vae_finetune_csem.json`，schema_version=3；只产生、读取和续训 `lwd_vae_csem_v3` checkpoint。
- 旧 AI 基线、双路线微调实现、旧数据生成器副本和历史验证记录已删除，可在 Git 历史中找回；未曾提交的 ver9 映射草案及旧数据集、旧训练输出也已删除，不能从 Git 恢复。
- 数据集命名从 `lwd_v1` 重新开始；`lwd_v2` 改为 1 inch、128×128 正方形全局窗口和组合式地质生成器（地层倾角、每层独立岩性、断层、尖灭、结核、流体界面等），见 [DATASET_V4_CN.md](DATASET_V4_CN.md)；`lwd_v3` 只把层厚中位数改为 1–8 ft；`lwd_v4` 去掉钙质结核。

## 验证边界

自动测试检查原版网络一致性、权重完整性/通道转换、三项损失数值、冻结特征的输入梯度、数据和坐标约定、确定性续训、checkpoint 评价和 latent 梯度。测试中用随机合成权重的部分仅验证逻辑。

本版测试与官方权重前向检查记录在 `docs/validation/`。短流程成功不代表正式数据训练已收敛，也不代表 LWD 电磁响应或反演已经验证有效。
