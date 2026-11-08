# 外部预训练权重放在哪里

两条正式参考配置都需要：

| 文件 | 内容 | 用途 |
|---|---|---|
| `weights/vae/kl_f8.ckpt` | 与原始 CompVis kl-f8 相容的完整 VAE 权重 | 初始化可训练 encoder/decoder |
| `weights/perceptual/alexnet_imagenet.pth` | torchvision AlexNet ImageNet 完整 state_dict | 冻结图像特征网络 |

也可通过 `--pretrained`、`--alexnet-weights` 指向已有文件。程序不会训练时联网下载，不会用随机特征代替真实 ImageNet 权重。仅靠文件名无法确认训练来源，请保留权重来源与说明；每次运行保存实际 SHA256。

VAE 支持原始 state_dict、包含 `state_dict` 的 ckpt/pt，或相同 CompVis 键名的 safetensors。配置为 ch=128、ch_mult=[1,2,4,4]、每级2个残差块、4通道latent。RGB 首尾权重自动取平均；中间结构必须完全匹配，缺失或未知模型权重会拒绝加载。Diffusers 命名或整套 Stable Diffusion checkpoint 不能直接当作独立 VAE。若旧 ckpt 含不被安全读取器接受的对象，应在可信来源环境导出纯 state_dict，而不是关闭安全读取。

AlexNet 可从 [torchvision 官方权重定义](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html) 获取 `IMAGENET1K_V1`。下列操作是**显式下载**，仅在需要时手动运行：

```python
from pathlib import Path
import torch
from torchvision.models import AlexNet_Weights

path = Path('weights/perceptual/alexnet_imagenet.pth')
if path.exists():
    raise FileExistsError(path)
state = AlexNet_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
path.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, path)
```

当前没有确认用户逆散射/CSEM 实验实际使用的那一份 VAE checkpoint，所以项目没有自动代选并下载别的 VAE。请以所持真实权重为准。随机合成测试权重不在这里发布，也不能作为正式初始化。
