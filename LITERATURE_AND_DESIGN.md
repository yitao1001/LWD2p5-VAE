# 文献与数据集设计依据 / Literature and design notes

数据文献检索日期：2026-10-15；训练方案归属与实现状态更新：2026-11-04。区分论文明确报道、提供源码的实际状态和本次工程判断。下文列出的空间尺寸不能直接等同于当前 500 kHz、16/24/32 inch 源距工具的探测范围。

## CSEM 方法来源与可复用部分

**Zhou, H., Guo, R., Sun, H., Li, M., Yang, F., Xu, S. (2026). Feature-Based Inversion of 2.5-D Controlled Source Electromagnetic Data Using Generative Priors. IEEE TGRS 64, 4501017. [DOI:10.1109/TGRS.2026.3661733](https://doi.org/10.1109/TGRS.2026.3661733).**

此前数据设计笔记依据用户提供的论文；本次训练方案再次核对了 [公开全文 §II-B](https://arxiv.org/html/2601.02145v1)。研究问题是如何将学习到的地质先验纳入仍由物理方程约束的 mCSEM 反演。VAE 学习模型分布，GN 更新物理模型，部分迭代进行 decoder-range 投影。论文训练集为 50,000 张 128×128 图像，90/10 划分；均匀导电背景中一或两个薄高阻层，logρ 背景 0、目标 U(1.3,2)。这种训练不需要每张地质图对应的正演响应；这是本次第一阶段省去 EM 数据库的依据。其图像先验不代表所有 LWD 地质。

EN: A model-space VAE prior is coupled to physics-based GN inversion. We borrow the separation of geological prior training from EM data generation, not the marine spatial scale or a claim that its prior transfers unchanged to LWD.

精确短引文（摘要）：“used solely to learn prior information on conductivity distributions”。本文直接照此只生成模型，最终能否改善 LWD 必须实验验证。

用户提供的源码目录中 `VAE_generate_dataset.py:126,136–180,227–275` 实际为 **20 张** 128×128 图像，横向20 km、z=1000–4015 m；背景1 Ωm，目标20–100 Ωm，厚80–150 m；50%单块，50%为2或3块；保存 Fortran 展平的 logρ，无固定seed或分组。**这与论文最终的50,000张、一或两层不完全一致。** 旧 `VAE_mCSEM_train_v1/v2/v3.py` 另包含不同年代的一维/二维 TensorFlow VAE；不可混用形状与归一化。

## 另一套逆散射材料（不是 CSEM 最终脚本）

用户已明确区分两套方案。原 `sdvae_model_finetune.py` 现位于 `references/inverse_scattering/`；以下是该上传快照的审查结果，不能据此推断 CSEM 最终代码或逆散射论文发表时的实际运行参数：

| 位置 | 问题及后续训练前需要处理的事项 |
|---|---|
| 49–55 | `np.resize` 不是空间插值；`(matrix-1)/(max_params-1)` 固定下界1，不适配新 logρ范围 |
| 79–85 | 无seed随机分割；setup可能再次改变划分 |
| 127–128、436 | `vae_finetune=False` 时不会加载预训练权重作为初始化 |
| 186–198 | 逆散射脚本为像素L1+LPIPS+KL；CSEM论文为像素L2+特征L2，两者来源不同 |
| 431、441 | 验证比例硬编码0.1；`lpips_loss_weight=args.kl_weight` 传错参数 |
| 358–363 | nn.Module访问 `self.trainer.datamodule` 且将latent强制square |

原文件保持原样。新主入口 `train_vae.py` 已分别实现两套参考微调配置，修正开关与损失参数接线，并接入 LWD 数据、验证、续训和测试接口。迁移假设及验证边界见 [方法对照](docs/METHODS_AND_ASSUMPTIONS_CN.md) 和 [训练指南](TRAINING.md)。

## 真正 2.5D 的主要 LWD 依据

### 1. 三层基础、断层验证与数据集规模

**Hashemian, A., Garcia, D., Rivera, J. A., Pardo, D. (2021). Database Generation for Deep Learning Inversion of 2.5D Borehole Electromagnetic Measurements using Refined Isogeometric Analysis. Computers & Geosciences. [DOI:10.1016/j.cageo.2021.104808](https://doi.org/10.1016/j.cageo.2021.104808). [作者全文预印本](https://arxiv.org/html/2009.08132v1).**

问题/方法：以 refined IGA 加速2.5D井中EM正演和DL响应数据库。§6.2验证单断层、双断层和倾斜层；§6.3实际100,000样本训练库却为三层模型。表2每层ρ=1–100 Ωm，上下边界距离0.01–10 m，工具与地层法线角80–100°。复杂验证另用3/100/0.5 Ωm。报道双CPU工作站约56 h生成100k带响应样本，此耗时不能外推到本项目。用途：支持层状基线和log域参数抽样；不将“断层验证”误写为“断层训练库”。

EN: True 2.5D simulation; faulted models validate the solver, whereas the reported DL database is three-layer parameterized geology. Its distances and instrument angles are not our local-window dip/thickness priors.

可引用短语（§6.2.2标题）：“Two geological faults and inclined layers”。限制：该数据库用于模型到响应/反演学习，本次只训练VAE模型分布。

### 2. 弯曲层界、不整合与流体界面

**Wu, Z., Li, H., Han, Y., Zhang, R., Zhao, J., Lai, Q. (2022). Effects of formation structure on directional electromagnetic logging while drilling measurements. JPSE 211,110118. [DOI:10.1016/j.petrol.2022.110118](https://doi.org/10.1016/j.petrol.2022.110118). [出版商页面](https://www.sciencedirect.com/science/article/pii/S0920410522000158).**

使用2.5D有限差分分析地质结构对方向性LWD测量的影响，明确研究断层、弯曲层界、不整合、油水界面。用途：直接支持这四个模型家族，而非仅生成随机异常块。本轮可访问摘要/检索正文片段，未可靠取得完整电阻率表，故不据此设定ρ上下限。其探测距离随断层角变化的量化结论有具体工具前提，不用于确定这里的窗口。

EN: Formation structures beyond parallel layers affect directional LWD responses. The work motivates curved interfaces, unconformities and fluid contacts; it does not prescribe a universal VAE mixture distribution. 本次不提供未核实全文的逐字引文。

### 3. 断层分类与2.5D反演

**Noh, K., Torres-Verdín, C., Pardo, D. (2022). Real-Time 2.5D Inversion of LWD Resistivity Measurements Using Deep Learning for Geosteering Applications Across Faulted Formations. Petrophysics 63(4),506–518. [DOI:10.30632/PJV63N4-2022a2](https://doi.org/10.30632/PJV63N4-2022a2). [一手摘要](https://onepetro.org/petrophysics/article-abstract/63/04/506/495116/Real-Time-2-5D-Inversion-of-LWD-Resistivity?redirectedFrom=fulltext).**

方法：高阶网格自适应FEM生成响应，再按不穿层界/断层、穿层界、穿断层等类型分类与反演。摘要支持两个受实测启发的合成例，并强调交叉分量在断层附近的作用。用途：断层家族与按家族报告测试误差。限制：本轮依据公开摘要；不补猜ρ、层厚范围，不把完整三轴测量的可恢复性结论套用到仅Hyy/Hzz的实验。

EN: Specialized DL inversion follows structure classification across faulted, anisotropic formations. This supports structural diversity, not a guarantee that a single-channel isotropic VAE or a reduced measurement set recovers the same features.

补充：**Noh, K., Pardo, D., Torres-Verdín, C. (2022, online2021). 2.5-D Deep Learning Inversion of LWD and Deep-Sensing EM Measurements Across Formations With Dipping Faults. IEEE GRSL19,8023805. [DOI:10.1109/LGRS.2021.3128965](https://doi.org/10.1109/LGRS.2021.3128965).** 倾斜断层、各向异性、短间距与深探测组合；IEEE全文本次不可访问，只作为结构家族补充，不使用其未核实数值。

### 4. 楔形地层与断层正演

**Zhang, P., Deng, S., Yuan, X., Liu, F., Xie, W. (2025). Fast forward modeling and response analysis of extra-deep azimuthal resistivity measurements in complex model. Frontiers in Earth Science13:1506238. [全文/DOI](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1506238/full).**

2.5D FEM研究断层和楔形地层。§3三层验证1/10/1 Ωm、中层5 m；§4断层模型4/20/2 Ωm、中层10 m，断距案例6–9 m。工具8 m源距、24 kHz。用途：支持非平行层界和楔形家族；本次尖灭进一步让厚度连续降到0，是我们的实现选择。限制：extra-deep参数不能原样缩放成当前短源距工具的可辨识范围。

EN: Fault and wedge examples support nonparallel interfaces. Our window and thickness ranges are chosen from the current inversion mesh, not copied from the extra-deep acquisition geometry. 短引文不作为本次设计必要条件，未复制长段原文。

## 辅助对照：不要把拼接1D当作2.5D像素反演

**Wang, L., Fan, Y. (2019). Fast inversion of logging-while-drilling azimuthal resistivity measurements for geosteering and formation evaluation. JPSE176,342–351. [DOI:10.1016/j.petrol.2019.01.067](https://doi.org/10.1016/j.petrol.2019.01.067). [出版商](https://www.sciencedirect.com/science/article/pii/S0920410519300725).**

中央TI层夹于各向同性肩层、7参数、滑窗拼接1D反演，含褶皱/断层合成评估。文中说明断层附近多个边界贡献可能使等效1D不能体现真实结构。用途：解释为什么仅三层平行模型不足以覆盖目标场景；不将其列为真正2.5D反演成果。

EN: Stitched 1D inversion is a useful baseline but is not equivalent to 2.5D pixel-based inversion near faults. 各向异性机制与本次标量ρ先验之间仍有明确差距。

## 本次推导的配置选择

2026-11-04 更新：当前 v2 按用户选择删除随机整体地层倾角，使用全局坐标对齐的水平背景；断层、曲率、侵蚀和尖灭仍作为局部地质形态。固定全局坐标本身不意味着所有真实地层必然水平，此处是本版先验范围的选择。旧文献案例中的倾斜地层并不自动进入当前默认数据集。详见 [v2 说明](docs/DATASET_GLOBAL_V2_CN.md)。

1. 使用当前反演的10×4 ft、1 inch网格，而非marine CSEM的20 km/百米目标；不改变物理长宽比。
2. 主体为有层序的模型；高/低阻都出现，均匀模型占5%，少量孤立异常用于对照。初版两端元层序是可解释的简化，未来可加入每层独立ρ与层内变化。
3. 0.5–200 Ωm是结合公开算例与当前问题提出的覆盖范围；200这个上限和各家族比例不是论文结论。Δlogρ弱对比特别覆盖当前1.0/1.1算例。
4. 各向异性暂不生成，防止先验自由度超过当前标量正演接口。VAE先验并不弥补测量对结构不敏感的问题。
5. 训练评价分为同分布重构、对比度外推、少量重构模型EM响应复核三个层次。当前已完成的是代码与数据格式验证，不是后两个层次的科学结论。
