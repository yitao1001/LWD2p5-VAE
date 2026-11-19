> 训练实现已统一为 CSEM 论文路线；当前以 [对齐说明](CSEM_ALIGNMENT_CN.md)、[训练指南](../TRAINING.md) 和 [数据集 lwd_v4](DATASET_V4_CN.md) 为准。lwd_v2 新增的文献依据见文末“2026-11-09 补充”。

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

原文件保持原样。当前训练只实现 CSEM 论文路线，不复制上述行为；迁移假设及验证边界见 [对齐说明](CSEM_ALIGNMENT_CN.md) 和 [训练指南](../TRAINING.md)。

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

## lwd_v1 的配置选择（历史记录）

以下为 lwd_v1 的设计记录，已被下一节取代。


2026-11-08 更新：数据集命名从 `lwd_v1` 重新开始，网格改为 ver8 反演网格（0.5 inch、96×256、10.67×4 ft）。背景层序水平、使用全局坐标；断层、曲率、背斜、侵蚀和尖灭作为局部地质形态。固定全局坐标本身不意味着所有真实地层必然水平，此处是本版先验范围的选择。详见 [lwd_v1 说明](DATASET_V1_CN.md)。

1. 使用 ver8 反演网格（10.67×4 ft、0.5 inch），而非marine CSEM的20 km/百米目标；不改变物理长宽比。
2. 主体为有层序的模型；高/低阻都出现，均匀模型占5%，少量孤立异常用于对照。初版两端元层序是可解释的简化，未来可加入每层独立ρ与层内变化。
3. 0.5–200 Ωm是结合公开算例与当前问题提出的覆盖范围；200这个上限和各家族比例不是论文结论。弱对比 Δlog10ρ 0.05–0.25 用于覆盖低对比结构；ver8 合成算例的 0.1 对比度偏低，不作为基准。
4. 各向异性暂不生成，防止先验自由度超过当前标量正演接口。VAE先验并不弥补测量对结构不敏感的问题。
5. 训练评价分为同分布重构、对比度外推、少量重构模型EM响应复核三个层次。当前已完成的是代码与数据格式验证，不是后两个层次的科学结论。

## 2026-11-09 补充：lwd_v2 的文献依据

检索日期 2026-11-09。只记录本次实际读到的内容。

**Shahriari, M., Pardo, D., Picón, A., Galdrán, A., Del Ser, J., Torres-Verdín, C. (2020). A deep learning approach to the inversion of borehole resistivity measurements. Computational Geosciences 24(3), 971–994. [DOI:10.1007/s10596-019-09859-y](https://doi.org/10.1007/s10596-019-09859-y)；[arXiv 1810.04522](https://arxiv.org/abs/1810.04522)。**

本次读了预印本 §6–7.2 的全文。训练模型为三层，上、中、下三层电阻率各自独立：
- 上下层 log10ρ ∈ [0, 3]（1–1000 Ωm）；中层 Rv/Rh ∈ [1, 10]；
- 到上下层界的距离在 0.01–10 m 内按对数均匀抽样；
- 地层倾角 β ∈ [−10°, 10°]；井轨迹倾角 83–97°。

用途：
- 每层独立取值（不再是两端元交替）；
- 层厚按对数尺度抽样；
- 地层倾角取几度到十几度；
- 对比度可达 2–3 个数量级。

**Hashemian et al. (2021)**（见上文）：每层 1–100 Ωm，工具与地层法线夹角 80–100°。复杂验证模型 3/100/0.5 Ωm，相邻差 1.5 和 2.3 个数量级。

**Fossum, K., Alyaev, S., Tveranger, J., Elsheikh, A. (2021). Deep learning for prediction of complex geology ahead of drilling. ICCS 2021, LNCS 12743. [DOI:10.1007/978-3-030-77964-1_36](https://doi.org/10.1007/978-3-030-77964-1_36)；[arXiv 2104.02550](https://arxiv.org/abs/2104.02550)；Alyaev, S., Tveranger, J., Fossum, K., Elsheikh, A. H. (2021). Probabilistic forecasting for geosteering in fluvial successions using a generative adversarial network. First Break 39(7), 45–50；[arXiv 2207.01374](https://arxiv.org/abs/2207.01374)。**

地质导向用的 GAN 先验在河流相剖面上训练，含三种相：背景泥岩、河道、决口扇。正演时分别赋 220、3.6、4.1 Ωm（该工区为高阻背景）。用途：生成式先验通常含三种以上岩相。尺度为几百米，与本项目不同，不借用几何参数。

**Chemali, R. (2009). Advanced geosteering with azimuthal deep resistivity helps in optimal well placement. AADE-NTCE-09-03（Halliburton Sperry Drilling Services）。[文件](https://www.aade.org/download_file/2593/489)。**

原文："Typical favorable conditions include a reservoir of 50 ohm-m with an approaching water zone or shale roof of 1 ohm-m." 该典型对比为 1.7 个数量级，已超过 lwd_v1 的上限 1.5。lwd_v2 因此把窗口内最大对比度提高到 2.5，并用 2.5–3.0 做外推测试。

**Talling, P. J. (2001). On the frequency distribution of turbidite thickness. Sedimentology 48, 1297–1329. [Caltech 记录](https://authors.library.caltech.edu/50326)。** 浊积岩层厚更符合对数正态混合分布。用途：每个样本先抽一个层厚中位数，再按对数正态抽各层厚度，使样本之间既有薄互层也有块状厚层。

**断层倾角。** 按 Anderson 断层理论，正断层倾角约 60°（[综述](https://en.wikipedia.org/wiki/Anderson%27s_theory_of_faulting)）。实测活动正断层多为 30–60°（[低角度正断层与地震活动综述](https://darchive.mblwhoilibrary.org/entities/publication/13cfb9f5-1750-55b7-9bea-c0c938f2420a/full)）。lwd_v1 的 `faulted` 只取偏离竖直 ±20°（倾角 70–90°），偏陡；lwd_v2 统一为偏离竖直 25–60°（倾角 30–65°）。

**钙质结核（lwd_v2、lwd_v3 使用，lwd_v4 起去掉）。**
- Albacora 油田（Campos 盆地）根据水平井岩心推断，单个结核长约 1–2 m，聚集体可达 6 m（[AAPG 摘要](https://www.searchanddiscovery.com/abstracts/html/1998/intl/abstracts/sombra.htm)）。
- Delaware 盆地 Cherry Canyon 组的胶结层从不到 1 inch 到 2–12 inch（[AAPG 摘要](https://www.searchanddiscovery.com/abstracts/html/2013/90163ace/abstracts/ch.htm)）。

原用途：结核长度 1.5–10 ft，厚度 0.2–1 ft，并允许只有一部分在窗口内。本次没有找到把结核尺寸与 LWD 电阻率尖峰幅度直接对应的文献，0.5–1.5 个数量级的超出量是工程设定。lwd_v4（2026-11-12）按用户决定去掉了这一类。

**油水过渡带。** 毛细管过渡带的厚度由毛管压力、渗透率等决定，可达 20 m 以上（[1.5D 过渡带井中电阻率模拟](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_journals_2410065712)）。在 10.67 ft 的窗口里，它表现为一段电阻率梯度；lwd_v2 取 0–1.5 ft 的线性过渡，较厚的过渡带由层内趋势近似表示。

## lwd_v2 的配置选择（2026-11-09）

1. **网格**：用户计划的新版反演网格，1 inch、128×128 正方形、全局坐标，不随工具倾角旋转。
2. **倾角**：地层倾角是图像中的地质参数，工具倾角属于正演；流体界面在全局坐标中水平。
3. **电阻率与对比度**：
   - 抽样范围 0.2–500 Ωm；每张图 2–4 种岩性，相邻层不同，同种岩性可重复；
   - 窗口内最大对比度：25% 为弱对比（0.05–0.3），其余为 0.3–2.5。
4. **组合式生成**：取消 9 个互斥家族，改为可叠加的组件（起伏、断层、尖灭、侵蚀面、流体界面、结核、渐变界面、层内趋势），冗余的起伏类型合并为一种。水平层状并入倾斜平面层序；完全常数的均匀模型改为 3% 的厚层样本。
5. **仍为工程设定的部分**：各组件概率、结核电阻率超出量、过渡带宽度，都不是从目标区块统计得到的。有邻井资料后应重新标定。
6. **lwd_v3（2026-11-09）**：层厚中位数由 [0.4, 6] ft 改为 [1, 8] ft，每张图的可见层数中位数从 8 降到约 5。理由是：真实地层确有厘米到分米级的薄互层，但地质导向目标通常为几英尺以上的单元；2 MHz/500 kHz 短源距工具能察觉几英寸的层，却不能反演出远薄于其分辨率的层的电阻率，这类层的响应主要取决于厚度与电导率的乘积（见上文工具资料）。在各向同性反演中，它们表现为一个中间值的等效层。薄层下限 0.2 ft 仍是工程设定，可以用 ver8 正演比较薄层与其等效值的响应差来确定。
7. **lwd_v4（2026-11-12）**：按用户决定去掉钙质结核（生成器 2.1.0），其余组件和参数与 lwd_v3 相同。数据中因此不再有横向有限的孤立地质体。
