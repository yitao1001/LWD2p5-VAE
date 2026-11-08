# 通用预训练权重

当前默认明确选择 **CompVis 原版 KL-f8**。它不是 CSEM 已微调权重，也不是后来发布的 SD `ft-MSE` / `ft-EMA` 版本。

```text
weights/
  vae/kl_f8_original.ckpt         官方 zip 中的 model.ckpt
  perceptual/alexnet_imagenet.pth 官方 ImageNet AlexNet
  downloads/kl-f8.zip             可重用的原始下载包
  provenance.json                本机文件、来源与 SHA256
```

在项目根目录执行 `python prepare_pretrained.py` 自动准备；再次运行会校验并复用已存在文件。`python prepare_pretrained.py --check-only` 只检查本地文件，不联网。下载中断可以再次运行继续 `.part`；大小和 SHA256 一致后才发布目标文件。完整性不符合时停止，不覆盖已有最终文件。

来源和已核实 SHA256 固定在 `configs/pretrained_sources.json`。原始 KL-f8 zip 约 1.01 GB，解压 checkpoint 约 1.10 GB（包括原训练附带状态），AlexNet 约 244 MB；加上训练 checkpoint 请预留数 GB 空间。

- [CompVis 原项目](https://github.com/CompVis/latent-diffusion)
- [官方原始 KL-f8 包](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip)
- [torchvision AlexNet 官方权重](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth)

训练只取完整 VAE state_dict，丢弃原损失网络/判别器，沿 RGB 取均值适配单通道。这里不对官方 checkpoint 做不透明修改。大权重、下载包和本机来源记录均不提交 Git；可将已校验的两个文件直接复制到服务器相同相对路径，再运行 `--check-only`。

也可通过 `--pretrained` / `--alexnet-weights` 指向其他已有权重，但训练记录的 SHA256 和源文件名会随之改变；只有结构兼容并不意味着权重来源相同。不要给随机权重起一个官方文件名。

官方 checkpoint 含旧 Lightning `ModelCheckpoint` 元数据标记；读取器以无行为的空类兼容该标记，仍保持 `weights_only=True`，不安装旧 Lightning、不启用任意 pickle 加载。该兼容路径已用真实官方文件验证。
