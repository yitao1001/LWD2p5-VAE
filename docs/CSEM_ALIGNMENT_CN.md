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
| 优化器、系数、调度 | 现有论文材料不足以恢复完整设置 | SGD、lr=1e-4、γ=1、β=1e-5、10 epochs 为 LWD 初始配置 |
| EMA / LPIPS / TV | 论文损失公式不能支持直接加入这些操作 | 正式路线不含这些附加项 |
| 训练数据 | CSEM 地质模型先验 | 用当前 LWD v2.2 的 48×120 全局网格，保留 1 inch 单元 |
| 验证与选模 | 作者实现细节未完全公开 | 固定 val，后验均值重构，按原始 log10ρ RMSE 选 best |
| 冻结参数统计 | 论文另报约 17M 非训练参数 | 本实现仅保留所选 AlexNet 卷积特征，冻结参数 2,469,696；不实例化原 checkpoint 的无用损失/判别器子网，不能宣称与论文参数统计逐项一致 |

官方原始 KL-f8 与后来发布的 `sd-vae-ft-ema` / `sd-vae-ft-mse` 是不同 checkpoint。后两者在其他图像集继续微调过 Decoder，见 [Stability AI 模型说明](https://huggingface.co/stabilityai/sd-vae-ft-mse-original)。本项目默认选 [原始 KL-f8](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip)，不把三个名称当作同一组权重，也不声称已获作者私有 checkpoint 的哈希。

## CSEM 文件夹中实际存在的其他代码

参考目录：`/Users/yitao/Documents/2026LWD_Aramco/2026年3月4日_25DCSEM_代码发送`。

- `VAE_mCSEM_train_v1/v2/v3.py` 是 TensorFlow 历史网络，包含从头初始化；相关 `GN_..._VAE.py` 导入 `Neural_Model_v2`。这不等于论文所述 KL-f8 微调。
- 该目录的 `sdvae_model_finetune.py` 与用户另传的逆散射脚本在训练主体上高度相近。它实际使用 L1 + LPIPS + KL；不能据此将它认定为 CSEM 论文式 (13) 的最终实现。
- 该脚本第 430 行 `vae_finetune=False` 会跳过权重加载，第 435 行把 `lpips_loss_weight` 接到 `args.kl_weight`，且 `np.resize` 与固定数值下界不适合 LWD。当前实现不复制这些行为。

目录脚本原样保存在 [references/csem_directory_snapshot](../references/csem_directory_snapshot/README.md)，只用于审计，不作为当前训练入口。用户另传的逆散射材料仍在 `references/inverse_scattering/`，来源保持分开。

## 本次整理

- 正式配置收敛到 `configs/vae_finetune_csem.json`，schema_version=3。
- 旧两路线配置、训练实现和文档快照归档到 `references/archive/finetuning_v2/`。
- 旧 AI 基线模型与训练器在 `lwd_vae/legacy/`；原入口、配置及旧测试在 `references/archive/baseline/`。
- 过去根目录散落的验证 JSON 移到 `docs/validation/archive/`，历史结果保留。
- 当前入口只产生 `lwd_vae_csem_v3` checkpoint；独立评估继续读取旧 v1/v2 checkpoint，续训只接受 v3。
- 地质生成代码、配置、正式数据和报告没有因训练重构而改变。

## 验证边界

自动测试检查原版网络一致性、权重完整性/通道转换、三项损失数值、冻结特征的输入梯度、数据和坐标约定、确定性续训、checkpoint 评价和 latent 梯度。测试中用随机合成权重的部分仅验证逻辑。

官方权重的加载与短流程微调记录独立写入 `docs/validation/CSEM_REFACTOR_VALIDATION.json`。短流程成功不代表正式数据训练已收敛，也不代表 LWD 电磁响应或反演已经验证有效。
