# CompVis AutoencoderKL 审计副本

`autoencoder_original.py` 是 CompVis/latent-diffusion commit `a506df5756472e2ebaf9078affdde2c4f1502cd4` 的未修改原文件，仅用于审计与测试，不在训练时导入其 Lightning/taming 生命周期。

运行时的原始 Encoder、Decoder、Gaussian 和 EMA 位于 `lwd_vae/vendor/compvis/`，文件只调整了本地导入前缀。`provenance.json` 记录上游和本地 SHA256，MIT 许可位于该目录的 `LICENSE`。本项目 `finetuning/model.py` 组装原始计算模块，`tests/test_finetuning.py` 对照此原文件的构造、编码、解码和 forward 方法检查数值一致性。
