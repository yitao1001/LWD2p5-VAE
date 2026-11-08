# 三套方案的关系与迁移假设

先纠正来源：用户上传的 `sdvae_model_finetune.py`、`kl_f8.yaml` 属于**逆散射方案**；它们不是 CSEM 论文的最终训练脚本。CSEM 按用户确认的论文对齐。旧 LWD 是此前自行实现的基线。本项目保留三条路线的身份，不把它们统称为同一方案。

| 项目 | CSEM 论文参考迁移 | 逆散射代码参考迁移 | 历史 LWD 基线 |
|---|---|---|---|
| 入口 | train_vae.py | train_vae.py | train_baseline.py |
| 配置 | finetuning/csem_reference.json | finetuning/inverse_scattering_reference.json | train_default.json |
| 网络 | 原始 CompVis core，完整 kl-f8，128 通道 | 同左 | 自行适配的 PyTorch 网络，默认 32 通道 |
| 初始化 | 必须加载预训练 VAE | 必须加载预训练 VAE | 默认从头训练 |
| 灰度转换 | encoder 首层和 decoder 末层对 RGB 取平均 | 同左 | 旧加载器 encoder 求和、decoder 平均 |
| 像素损失 | MSE | L1 | MSE |
| 图像特征损失 | 冻结 AlexNet 特征 MSE | 官方 LPIPS AlexNet v0.1 | 无；用相邻像素梯度差 |
| KL | 每样本求和，batch 平均；系数 1e-5 | 同左 | 每元素均值；最终系数 1e-4，10 轮 warmup |
| 特征权重 | 1.0，不衰减 | 1.0；第 8 轮起 0.1 | 梯度差权重 0.05 |
| 优化器 | SGD，lr=2e-4，momentum=.9，weight_decay=5e-4（迁移假设） | 上传代码的 SGD 设置 | AdamW，lr=1e-4 |
| EMA | 默认关闭，论文细节未核实 | 每轮末 LitEma 更新并回写，decay 上限 .99 | 无 |
| LWD 网格 | 三条路线都使用原 48×120 网格，不缩放成正方形 | 同左 | 同左 |
| 证据边界 | 论文支持方法路线，若干实现细节为本项目选择 | 代码可审查，但论文身份和发表时版本未知 | 仅已有代码流程验证，无 LWD 有效性结论 |

所有文件路径在表中省略共同前缀 `configs/`。两条新参考配置的 batch=4、float32、10 epochs 和最佳验证 RMSE 选模均为当前迁移设置，不宣称与论文完全一致。

## 从数据到训练目标

地质数组存储 `p=log10(ρ/(Ωm))`，固定边界 lo=-1、hi=3，经 `u=2(p-lo)/(hi-lo)-1` 进入网络。样本形状 `[B,1,48,120]`，编码器产生均值和 log 方差，采样后的 z 为 `[B,4,6,15]`，解码器重建同样大小的 u。原始电阻率变化、坐标长宽比和单元顺序都保留。

共同目标是 `L = L_pixel + λ_feature L_feature + β L_KL`。

- 像素 MSE/L1 对 batch、通道和全部像素取平均。CSEM 论文使用 L2 范数记号，未给出代码中的归约细节；此处平均化也是明确的迁移选择，权重必须与归约方式一起比较。
- CSEM 参考特征：u 从 [-1,1] 仿射变到 [0,1]，复制为三通道，再用 ImageNet mean/std 标准化。取 torchvision AlexNet 的五个 ReLU（索引 1、4、7、9、11），每层逐元素 MSE，最后对五层等权平均。没有 resize、crop、逐图归一化或裁切。**这些层、标准化和层间权重是迁移假设。**
- 逆散射特征：官方 `lpips==0.1.4` 的 AlexNet LPIPS v0.1，读取本地 ImageNet trunk 和包内校准头，输入复制为三通道的 [-1,1] 图像；不再叠加另一套 ImageNet 标准化。
- 两条路线冻结特征网络参数，但保留对预测图像的反向梯度。冻结不等于对整个特征损失使用 no_grad。
- KL 使用 `0.5 * Σ(μ² + exp(logσ²) - 1 - logσ²)`，每张图求和后 batch 平均。论文公式若省略 1/2，可被 β 吸收，但不能据此假定相同 β 等价。当前 360 个潜变量意味着“求和”是“逐元素均值”的 360 倍；网格变化时需重新评估 β。

## 哪些来自文献，哪些仍需验证

CSEM 的 [公开论文 §II-B](https://arxiv.org/html/2601.02145v1#S2.SS2) 支持预训练 kl-f8 微调、RGB 权重平均及像素 L2／图像特征 L2／KL 的组合。我们没有足够证据确定其完整超参数和特征抽取实现，所以将这些参数写入配置的 `reference.assumptions`。不能把“特征 L2”直接说成“与上传脚本的 LPIPS 完全相同”。

逆散射上传脚本的当前快照存在 `vae_finetune=False` 与 `lpips_loss_weight=args.kl_weight` 两处关键问题。这里显式启用预训练并将两种权重各自接入；保留其 SGD、LPIPS 衰减和每轮 EMA 回写机制。输入加载则使用 LWD 现有固定划分和物理归一化，替代 `np.resize`、固定下界 1 与无 seed 随机分割。修正后的实现不意味着原论文历史实验也运行了这些有问题的设置。

新主模型直接使用 [CompVis 固定版本](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4) 的网络计算模块。适配层保留 AutoencoderKL 的 quant/post_quant 卷积和状态键名；仅另写清晰的训练循环、LWD 检查及混合精度后验保护。它不是完整搬入原 Lightning 训练器，也没有引入其判别器训练。

数据现在采用全局对齐的 v2：删除随机整体倾角与 ood_dip，以水平背景为基础保留局部地质形态；尺度及归一化继续使用 LWD 设计。两条参考训练路线使用同一份 v2 数据，详见 [数据变更](DATASET_GLOBAL_V2_CN.md)。其地质合理性和先验覆盖仍需结合目标 LWD 案例验证，详见 [README](../README.md) 与 [数据设计依据](../LITERATURE_AND_DESIGN.md)。
