"""冻结图像网络：CSEM 风格特征 L2 与逆散射 LPIPS 是两种不同损失。

所有 ImageNet 权重必须显式从本地读取，不在训练开始时自动下载。
特征网络参数被冻结，但预测图像仍须能通过特征网络反向传播。
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..runtime import sha256_file


def _alexnet_features(weights_path: Path | None, restoring=False):
    from torchvision.models import alexnet

    model = alexnet(weights=None)
    if not restoring:
        if weights_path is None or not weights_path.is_file():
            raise FileNotFoundError("Provide local ImageNet AlexNet weights with --alexnet-weights")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("AlexNet weights must be a torchvision AlexNet state_dict")
        # 完整载入官方分类网络，检查键名与形状后，只保留卷积特征层。
        model.load_state_dict(state, strict=True)
    return model.features


class FrozenFeatures(nn.Module):
    def freeze(self):
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # 外层训练器即使调用 train()，也不能启用特征网络中的 dropout。
        return super().train(False)


class AlexNetFeatureL2(FrozenFeatures):
    """工程化解释 CSEM 公式(13)：所选 ReLU 特征的逐元素 MSE，再对层平均。

所用层、ImageNet 标准化、各层等权平均是显式迁移假设；论文未公开这些
全部细节，因此此实现称为 reference adaptation，不称逐项精确复现。
"""
    def __init__(self, weights_path: Path | None, layers: list[int], *, restoring=False):
        super().__init__()
        features = _alexnet_features(weights_path, restoring)
        self.layers = tuple(layers)
        # 截到最后一个所选 ReLU，跳过最后的分类池化；支持 48x120。
        self.features = features[: max(layers) + 1]
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.freeze()

    def activations(self, image):
        # 数据集的固定 [-1,1] 数值映射；无逐张归一化、裁切或空间缩放。
        value = ((image + 1) / 2).repeat(1, 3, 1, 1)
        value = (value - self.image_mean) / self.image_std
        outputs = []
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in self.layers:
                outputs.append(value)
        return outputs

    def forward(self, prediction, target):
        predicted = self.activations(prediction)
        with torch.no_grad():
            expected = self.activations(target)
        return torch.stack([F.mse_loss(a, b) for a, b in zip(predicted, expected)]).mean()


class AlexNetLPIPS(FrozenFeatures):
    """官方 LPIPS v0.1 校准头 + 本地 ImageNet AlexNet；输入仍为 [-1,1]。"""
    def __init__(self, weights_path: Path | None, *, restoring=False):
        super().__init__()
        import lpips

        # pnet_rand=True 仅禁止库隐式联网；下一步用本地权重替换全部 trunk。
        # 续训时由 checkpoint 中的完整 loss_state 恢复 trunk 和校准头。
        self.metric = lpips.LPIPS(net="alex", version="0.1", pnet_rand=True,
                                  pretrained=True, eval_mode=True, verbose=False)
        features = _alexnet_features(weights_path, restoring)
        for index, (start, stop) in enumerate(((0, 2), (2, 5), (5, 8), (8, 10), (10, 12)), 1):
            setattr(self.metric.net, f"slice{index}", nn.Sequential(*list(features.children())[start:stop]))
        self.freeze()
        calibration = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
        self.calibration_sha256 = sha256_file(calibration)

    def forward(self, prediction, target):
        return self.metric(prediction.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1), normalize=False).mean()


def build_feature_loss(settings: dict, weights_path: Path | None, *, restoring=False):
    kind = settings["feature"]
    if kind == "none":
        return None
    if kind == "alexnet_l2":
        return AlexNetFeatureL2(weights_path, settings["feature_layers"], restoring=restoring)
    if kind == "lpips_alex":
        return AlexNetLPIPS(weights_path, restoring=restoring)
    raise ValueError(f"Unknown feature loss: {kind}")
