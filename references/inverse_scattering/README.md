# 用户提供的逆散射原始材料

`sdvae_model_finetune.py` 和 `kl_f8.yaml` 从项目根目录移到此处，内容保持不变。它们属于用户指出的逆散射方案，不是已确认的 CSEM 最终代码；论文标题及发表实验对应版本仍未知。

只用于逐项核对来源，不作为新 LWD 训练入口。新入口为项目根目录 `train_vae.py`，对应配置为 `configs/finetuning/inverse_scattering_reference.json`。

原快照的微调开关为 false，LPIPS 权重误接 KL 参数，并使用不适合 LWD 的 resize/归一化。新实现的修正和保留项见 `docs/METHODS_AND_ASSUMPTIONS_CN.md`。原作者代码的使用与再分发权限由其原始授权决定；移入 references 不改变授权归属。
