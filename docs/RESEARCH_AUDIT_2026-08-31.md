# TriFusion-ReID 研究审计：新颖性、RGBNT201 协议与可复现目标

> 审计日期：2026-08-31
> 范围：RGB–NIR–TIR 多模态目标 ReID；只采用论文官网、作者官方项目页和官方 GitHub 仓库。本文没有下载数据集或运行训练，因此“可复现”仅表示公开工件足够启动复现，不表示本文已复现其指标。

## 结论摘要

截至审计日，在聚焦的主源检索范围内，**没有发现一个已公开的 RGB–NIR–TIR ReID 方法同时包含**：

1. 三个完整的 CNN / Transformer / Mamba 异构架构专家；
2. 多深度、双向、阶段式跨架构信息交换；
3. 显式可靠性同时控制消息交换和最终融合；
4. 依据校准置信度逐样本选择教学方向、且允许拒绝教学的互蒸馏。

这是有边界的否定性检索结论，不是“全球不存在”的证明，也不足以直接使用“首个”。三分支、渐进融合、CNN–Transformer 深层互联、Mamba 跨模态建模、不确定性感知融合、困难样本蒸馏和互蒸馏都已有直接先例。可辩护的新颖性应限定为：**由同一个校准可靠性后验统一调控三类异构专家之间的阶段式通信、融合与方向选择式同伴蒸馏**。

## 1. RGBNT201 协议必须先冻结

RGBNT201 原论文使用 201 个身份和 4 个非重叠视角，共 4,787 组三模态记录；141/30/30 个身份用于训练/验证/测试；测试时每个身份随机选 10 条记录作为 probe，全部测试记录作为 gallery。该定义见 [PFNet / RGBNT201 原始论文的 Dataset Description](https://ojs.aaai.org/index.php/AAAI/article/download/16467/16274)。

当前 DeMo 系公开代码采用了不同的事实标准：

- `train_171` 合并原训练与验证身份；
- 整个 `test` 同时作为 query 和 gallery；
- 排除与 query 同身份且同 camera 的 gallery 样本；
- 通常报告 mAP、Rank-1、Rank-5、Rank-10，不使用 reranking。

这一行为可直接从 [DeMo 的 RGBNT201 loader](https://github.com/924973292/DeMo/blob/b4f323a430b32e3a1637c3e7acb25868cb52e9cd/data/datasets/RGBNT201.py)、[DeMo evaluator](https://github.com/924973292/DeMo/blob/b4f323a430b32e3a1637c3e7acb25868cb52e9cd/utils/metrics.py)、[MFRNet loader](https://github.com/stone96123/MFRNet/blob/ec54a1302321cda4b5fad9ca1c0878dabf0b46b6/data/datasets/RGBNT201.py)、[MDReID loader](https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/data/datasets/RGBNT201.py)、[CoT-ReID loader](https://github.com/Gaoya615/CoT-ReID/blob/db215273d6ee68b9c324fdf36e3d6800370fa21e/data/datasets/RGBNT201_Text_cot.py)、[PRISM loader](https://github.com/zw-absin/PRISM/blob/0067f6d895c522afa2c4f30515b33bc4300fe680/data/datasets/RGBNT201.py) 和 [STMI loader](https://github.com/young6man/STMI/blob/27a74bb90ad46a6d9feab67a6a26753e11a8ad36/data/datasets/RGBNT201_Text.py) 核实。

因此，原论文的“随机 300 probe”结果不能与现代 `train_171 + full-test query/gallery` 结果直接混排。后续实验应保存数据文件清单及哈希、身份划分、camera 解析规则、过滤规则和随机种子。下文的目标值仅讨论现代协议谱系；对没有发布代码的 RoDI、PMKD、NEXT，只能核对论文表述，不能独立审计其实际 loader。尤其是 NEXT v5 的数据集段落仍写 141/30/30 身份划分，而其[官方训练 caption JSON](https://github.com/lsh-ahu/NEXT-ReID/blob/b86b8fd253f3c872f366a88e46282a23a6a84db7/Annotation_Text/NEXT/201/train_RGB.json)实际覆盖 171 个身份（3,951 个文件名），测试 JSON 覆盖 30 个身份（836 个文件名）；没有 loader/evaluator 时仍无法确认 query/gallery 构造。

## 2. 可复现高基线与同协议目标

以下指标均为 RGBNT201 的 `mAP / Rank-1`。工件状态按 2026-08-31 的官方仓库 HEAD 固定；“工件可用”不代表本审计已运行 checkpoint。

### 2.1 静态纯视觉、CLIP-B/16、256×128

| 方法 | 报告指标 | 额外资源与推理 | 官方工件状态 | 审计判断 |
|---|---:|---|---|---|
| DeMo-CLIP | 79.0 / 82.3 | 无文本、掩码、TTT、rerank | [论文 Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/32878/35033)；[代码和配置](https://github.com/924973292/DeMo/tree/b4f323a430b32e3a1637c3e7acb25868cb52e9cd)，未见训练后 checkpoint | 适合作为现有工程实现基线，但不是最高的公开可测基线 |
| ICPL-ReID | 75.1 / 77.4 | CLIP 图文预训练、身份条件 prompt；无外部样本级文本、TTT、rerank | [TMM 作者稿 Table I](https://arxiv.org/abs/2505.17821)；[代码、配置和 RGBNT201 checkpoint](https://github.com/lsh-ahu/ICPL-ReID/tree/47c3d128b16c1183cf8aa66cfa76de9eef334bed) | 工件完整度较高，但指标低于当前强锚点 |
| Signal | 80.3 / 85.2 | CLIP，选择性局部/全局交互；无文本、TTT、rerank | [AAAI 2026 论文 Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/37674/41636)；[代码、配置和 RGBNT201 checkpoint 链接](https://github.com/010129/Signal/tree/cd1b0a672d1fe642e7608731cb4899a19dda7d51) | 固定源码与真实 CPU loader 已审计；公开 test log 指向作者路径 `signal_50.pth`，但 checkpoint 字节尚未取得，80.3 / 85.2 不是本机复现值 |
| MFRNet | 80.7 / 83.6 | 无额外标注、TTT、rerank | [ICML 2025 论文](https://proceedings.mlr.press/v267/feng25i.html)；[代码、配置和 RGBNT201 checkpoint](https://github.com/stone96123/MFRNet/tree/ec54a1302321cda4b5fad9ca1c0878dabf0b46b6) | 官方 checkpoint 已下载并在隔离环境完成 297/297 张量严格匹配；公开 best 由每轮 official-test mAP 选出，本机 GPU 指标仍待复现 |
| MGRNet-CLIP | 80.5 / 85.0 | CLIP，模态感知图推理与缺失模态重构；静态推理 | [作者稿 Table I](https://arxiv.org/abs/2504.14847)；[作者仓库](https://github.com/wanxixi11/MGRNet/tree/5c2cfd68ba566db3fdeaaf9bbc9f4bba3778b763) 仅 README | 只能作为论文报告值，不能独立复现 |
| MDReID | 82.1 / 85.2 | 无文本、掩码、TTT、rerank | [NeurIPS 2025 论文](https://papers.neurips.cc/paper_files/paper/2025/file/3cbe9fcdccb2399bcd6e6d01cbcae1fd-Paper-Conference.pdf)；[代码、配置和 checkpoint](https://github.com/stone96123/MDReID/tree/3525ac2da1a2a90a5a160c930fac674b4f226f6c) | 本次核查中指标最高的 checkpointed 静态 CLIP 基线；另含 any-to-any 模态任务 |
| UGG-ReID | 81.2 / 86.8 | 不确定性图和 MoE；无 TTT/rerank | [NeurIPS 2025 论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/735c847a07bf6dd4486ca1ace242a88c-Paper-Conference.pdf)；[代码和配置](https://github.com/wanxixi11/UGG-ReID/tree/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1)，未见 checkpoint | 静态可靠性融合的重要对照；需自行训练 |
| FUSE | 81.4 / 86.1 | CLIP，频域分解与能量对齐，256×128；静态推理 | [ICML 2026 作者稿 Table 1](https://arxiv.org/abs/2606.20044)；截至审计时未定位到作者代码仓库 | 论文报告值低于当前目标；无 loader/checkpoint，协议实现不可独立审计 |
| PEFT-BoA | **82.7 / 86.1**（test-selected e80）；fixed e120 为 82.2 / 85.8 | 冻结 CLIP 主干、6.62M 可训练 adapter，256×128；无 TTT/rerank | [AAAI 2026 论文 Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/37537/41499)；[代码、配置和完整训练日志](https://github.com/fffunly/PEFT-BoA/tree/d2b198be634ac4f9f5744eebf6e0a6604e490deb)，无 release/checkpoint | 公开日志证明论文值来自每轮 official-test 选出的 epoch80；公平主行必须从头训练并固定报告 epoch120 |
| RoDI-CLIP | **84.1 / 87.2** | 静态纯视觉推理；无 TTT/rerank | [CVPR 2026 Findings 论文](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.html)；[仓库](https://github.com/lsh-ahu/RoDI/tree/2f38911c49d42d4ca259d440a851b8d77dddccbe) 仅 README/assets | 静态 CLIP 论文报告上限，当前不可独立复现 |

Signal、PEFT-BoA 和 ICPL-ReID 的已发布 loader 均显式读取 `train_171`，并以完整 `test` 同时构造 query/gallery；因此其训练身份范围可以代码审计。Signal 的真实 CPU loader 探针进一步确认 171 个训练 ID、3951 组三模态训练样本、836 query 和 836 gallery，B64/K8 批次的三路形状均为 `64×3×256×128`。其训练循环每轮读取 official test 并按 test mAP 写 `Signalbest.pth`；周期 `Signal_50.pth` 在第 50 轮 test 评估前保存，而发布的 `test.py` 硬编码另一个作者路径 `signal_50.pth`。因此公开 80.3 / 85.2 只能写成“上游固定路径 test-log 值”，不能在取得 checkpoint 哈希并本机评估前称为复现。MGRNet 的作者仓库在固定提交仅有 README，FUSE 未定位到作者代码，二者的 loader/evaluator 仍只能按论文表述归类。增量扫描中没有任何静态 CLIP 结果超过 RoDI-CLIP 的 84.1 / 87.2。

工程角色据此固定为：**Signal 是当前同输入静态候选中指标最高且固定提交具有仓库级 MIT 许可证的 baseline；DeMo 是已接入的 MIT 实现脚手架；MDReID 82.0868 / 85.1675 是本机已严格复现的最高 checkpoint 锚点；PEFT-BoA 是待从公开可见源码重训的更高 released-protocol 比较器。** Signal 的 80.3 / 85.2 仍只是上游固定路径日志，必须本机完成 fixed-e50 后才可称复现；PEFT-BoA 固定提交没有仓库级许可证，不能再称“开源”，且其 82.7 / 86.1 只能标为 test-selected epoch80。完整证据与复用边界见 [baseline 选择及许可证审计](BASELINE_SELECTION_AND_LICENSE_AUDIT_2026-08-31.md)。

### 2.2 静态更强预训练、多阶段 KD 与 TTT

| 赛道 | 方法 | 报告指标 | 资源与推理条件 | 官方工件状态 |
|---|---|---:|---|---|
| 静态、更强预训练 | RoDI-DINOv3 | **85.3 / 87.9** | distilled DINOv3 ViT-B/16，224×224；静态纯视觉 | 同 RoDI 论文；仓库无代码、配置或权重 |
| 多阶段 KD | PMKD-DINOv2 | 84.7 / **88.9** | DINOv2；两阶段 source→target 多模型 KD；最终仅静态学生推理 | [AAAI 2026 论文 Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/38338/42300)；[仓库](https://github.com/moonaricc/PMKD/tree/0f597faad5b1432ce37b8be52e9bfac80b259f1f) 仅占位 README |
| Proxy 方法、关闭 TTT | ProxyTTT w/o TTT | 82.3 / 84.7 | CLIP ViT-B/16，256×128；静态推理 | [AAAI 2026 论文 Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/38337/42299)；[代码和配置](https://github.com/liuzhaojun-zwd/ProxyTTT/tree/92fb0fa33d74813566e06820e56e8d8f48ca1205) |
| TTT | ProxyTTT | 85.0 / 88.5 | 无标签测试域图像、代理伪标签、熵筛选和 2 个 TTT epoch 的梯度更新；无 rerank | 同上；官方仓库发布 RGBNT201 checkpoint |

复核结果：RoDI、PMKD、ProxyTTT 的既有数字无需更正。RoDI 的 85.3 / 87.9 是 DINOv3 静态纯视觉结果；PMKD 的 84.7 / 88.9 是 DINOv2 多阶段蒸馏后最终学生结果；ProxyTTT 的 85.0 / 88.5 明确包含 PESA 测试时更新，关闭 TTT 后为 82.3 / 84.7。

### 2.3 非同协议与未公开结果边界

| 工作 | 已核实的公开事实 | 为什么不改变当前 SOTA 门 |
|---|---|---|
| UPCL，NeurIPS 2025 | 论文明确把 RGBNT201 与 RGBNT100 合并训练，再分别测试；RGBNT201 的 `RNT→RNT` 为 64.91 / 67.12；[论文](https://papers.nips.cc/paper_files/paper/2025/file/82a0696bea2c4ebf726fc796eaca7a55-Paper-Conference.pdf)和[代码](https://github.com/ZhouZhongao/UPCL/tree/c2c01c2b4ecbe79b39de555da872647d10a55ff8)均公开 | 训练数据和 seven-task any-to-any 目标均不同，不能与仅用 RGBNT201 `train_171` 的静态融合结果混排 |
| Hyper-ReID，ACM MM 2026 accepted | [作者官方仓库](https://github.com/lsh-ahu/Hyper-ReID/tree/6e895a707c0948d03968b3e812ec6cf5fbcd1eb9) 声明录用，但固定 HEAD 只有一个 README；无论文、指标、代码、配置、权重或 release | 目前没有可加入数值榜单的公开证据；它是任何最终 SOTA 表述前必须重新检查的显式未决项 |

`Hyper-ReID` 的存在否定了“尚无该工作”的旧检索表述，但其占位仓库不能用于推断性能。当前 84.1 / 87.2 只能写成“截至本次主源扫描所定位到的公开静态 CLIP 论文报告上限”，不能写成未来投稿时仍然有效的绝对门槛。

### 2.4 额外语义资源赛道

这些方法在训练或测试中使用样本级文本、分割/关键点掩码，不能与上面的纯视觉静态赛道无条件混排。

| 方法 | 报告指标 | 骨干与输入 | 额外资源及其使用阶段 | 官方代码、权重与协议状态 |
|---|---:|---|---|---|
| PRISM | 80.5 / 84.0 | CLIP ViT-B/16，256×128 | RGBNT201 主配置使用离线 OpenPifPaf 关键点衍生前景 mask；车辆数据使用 SAM2；测试也读取 mask；无 TTT/rerank | [论文 Table I、§IV-A](https://arxiv.org/abs/2607.23451)；[代码、配置、预计算 mask 数据入口和 RGBNT201 权重](https://github.com/zw-absin/PRISM/tree/0067f6d895c522afa2c4f30515b33bc4300fe680) 已发布；loader 可核实为现代协议 |
| STMI | 81.2 / 83.4 | CLIP 视觉/文本编码器，256×128 | 每个 triplet 使用 GPT-4o 生成一条描述，并用 SAM2 生成一个 mask；文本与 mask 在测试 forward 中仍被读取；无 TTT/rerank | [AAAI 2026 正式论文 Table 1、Experiments](https://ojs.aaai.org/index.php/AAAI/article/download/38125/42087)；[代码和配置](https://github.com/young6man/STMI/tree/27a74bb90ad46a6d9feab67a6a26753e11a8ad36) 已发布，但 README 的 Annotations/Masks 链接为空，未见 checkpoint；loader 可核实为现代协议；论文写学习率 3.5e-6，当前 RGBNT201 配置为 3.5e-4 |
| NEXT | 82.4 / **86.6** | CLIP-B/16，256×128 | GPT-4o 与 Qwen-VL 生成带置信度属性，DeepSeek-V3 合成每模态 caption；论文效率表明确测试使用 CLIP 文本编码器；无 TTT；其 `w/o Text` 消融为 79.2 / 85.5 | [arXiv v5 Table II、Table VIII](https://arxiv.org/abs/2505.20001)；[官方仓库](https://github.com/lsh-ahu/NEXT-ReID/tree/b86b8fd253f3c872f366a88e46282a23a6a84db7) 已发布 RGBNT201 caption JSON，但 README 注明模型代码仍待整理；未见模型源码、配置或权重，协议实现不可审计 |
| CoT-ReID | **83.3** / 86.1 | DINOv3-B 视觉骨干 + 冻结 CLIP 文本编码器，256×128 | API Qwen-VL 为训练集和测试集每个模态生成描述与推理链；测试时官方代码拼接图像和文本特征；无 TTT/rerank | [CVPR 2026 Table 2、§4.1–4.2](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.html)；[源代码和配置](https://github.com/Gaoya615/CoT-ReID/tree/db215273d6ee68b9c324fdf36e3d6800370fa21e) 已发布，但官方 README 明确不分发 CoT 文本、预训练权重或 checkpoint，且无锁定环境；论文写 120 epochs，当前 RGBNT201 配置为 60 epochs |

[IDEA](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_IDEA_Inverted_Text_with_Cooperative_Deformable_Aggregation_for_Multi-modal_Object_CVPR_2025_paper.pdf) 同样使用视觉语言模型生成文本，属于额外语义资源对照。上述方法及纯视觉方法均未报告 RGBNT201 的 mINP。

基于现有 DeMo 工程继续开发时，应把 DeMo 限定为 MIT 实现脚手架，把 Signal 作为待 fixed-e50 本机重训的高指标 MIT baseline，把已完成 parity 的 MDReID 作为强 checkpoint 锚点；PEFT-BoA、MFRNet、UGG-ReID 和 MDReID 的固定源码树没有仓库级许可证，只能隔离执行或依据论文独立实现，不能称为开源代码基线。RoDI、FUSE、MGRNet、PMKD、NEXT 只能作为“论文报告上限”。PRISM 虽为 MIT，但结果依赖预计算 mask；CoT-ReID 和 STMI 有代码却缺关键文本/掩码或 checkpoint，不能称为端到端已封装复现。

可审计的目标应分轨表达：

- 静态纯视觉 CLIP-B/16、256×128：RoDI-CLIP 84.1 / 87.2；
- 静态纯视觉 CLIP-B/16、MIT 许可的高指标 baseline：Signal 上游固定路径日志为 80.3 / 85.2，待本机 fixed-e50 复现；
- 静态纯视觉 CLIP-B/16、公开可见但无仓库级许可证的训练源码：PEFT-BoA released-test-selected e80 为 82.7 / 86.1、同一公开日志 fixed e120 为 82.2 / 85.8；两者均待本机从头复现并分栏报告；
- 静态纯视觉、更强外部预训练：RoDI-DINOv3 85.3 / 87.9；
- 多阶段 DINOv2 蒸馏：PMKD 84.7 / 88.9；
- 测试时训练：ProxyTTT 85.0 / 88.5；
- 额外语义资源：分别注明“mask-only PRISM 80.5 / 84.0”“GPT-4o text + SAM2 mask STMI 81.2 / 83.4”“MLLM/LLM captions + CLIP NEXT 82.4 / 86.6”“Qwen-VL CoT text + DINOv3 CoT-ReID 83.3 / 86.1”，不把它们压成一个无条件榜单。

在相同协议、预训练、输入、额外标注、reranking 和推理更新条件下真正超过对应目标前，只能写“目标达到 SOTA”，不能写“已达到 SOTA”。

## 3. 新颖性碰撞矩阵

| 最接近主源 | 已公开内容 | 与目标方案仍可区分的边界 |
|---|---|---|
| [PFNet / RGBNT201，AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16467) | RGB/NIR/TIR 三个模态分支；由局部到全局、由单模态到成对/多模态的渐进融合 | 已否定“三支”和“渐进融合”本身的新颖性；目标三支必须是架构专家，而不是把 CNN/Transformer/Mamba 分别绑定到三个传感器 |
| [IEEE: Interact, Embed, and EnlargE，AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20165) | 在特征提取过程中进行跨模态分支交互 | “在中间层交换信息”不是新贡献，必须说明跨架构尺度对齐、双向路径和私有特征保留机制 |
| [FusionReID](https://arxiv.org/abs/2412.17239) / [官方代码](https://github.com/924973292/FusionReID) | 普通 RGB ReID 中 CNN–Transformer 并行分支和多层异构传输 | “CNN+Transformer 深层协同”已有先例；其不含 RGBNT 三模态或独立 Mamba 专家 |
| [MambaPro，AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/32879/35034) / [官方代码](https://github.com/924973292/MambaPro) | 三个 Transformer 模态流、层间 Synergistic Residual Prompt、最终 Mamba Aggregation | 已覆盖 Transformer+Mamba 和深层跨模态协同；Mamba 是聚合器而非完整专家，无 CNN、可靠性门控或 KD |
| [PRISM，arXiv 2026](https://arxiv.org/abs/2607.23451) / [官方代码](https://github.com/zw-absin/PRISM) | Prompt-S6/Mamba 跨模态建模，以及“模态内→成对模态间→三模态”的阶段式融合；使用离线语义掩码 | 已占据 Mamba+三阶段融合；其阶段是模态维，不是 CNN/Transformer/Mamba 架构维，也没有教师/学生互蒸馏 |
| [NEXT，arXiv v5](https://arxiv.org/abs/2505.20001) / [官方仓库](https://github.com/lsh-ahu/NEXT-ReID) | 文本调制语义专家、跨模态共享结构专家和统一多粒度聚合；caption 在推理时参与路由/表示 | “多粒度专家+文本调制路由”已有直接先例；目标需证明路由来自视觉可靠性而非外部 caption，并证明 CNN/Transformer/Mamba 专家各自保留独特能力 |
| [STMI，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38125) / [官方代码](https://github.com/young6man/STMI) | GPT-4o 文本与 SAM2 mask 引导 token 调制、token 重分配和跨模态超图高阶交互 | “语义先验控制 token”和“跨模态高阶关系”已有先例；其没有三类异构架构专家、可靠性统一控制或选择式互蒸馏 |
| [CoT-ReID，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.html) / [官方代码](https://github.com/Gaoya615/CoT-ReID) | Qwen-VL 推理文本在早期特征、跨模态一致性和最终图文特征层全流程参与 | “辅助信息贯穿多层协同”不是纯结构创新；其依赖测试样本级文本，不能作为纯视觉异构专家方法的等资源先例 |
| [UGG-ReID，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/735c847a07bf6dd4486ca1ace242a88c-Paper-Conference.pdf) | 局部与样本级 aleatoric uncertainty、低不确定性专家路由和不确定性引导融合 | “可靠性/不确定性感知融合”已被直接覆盖；可区分点必须是同一校准后验同时控制交换、融合和蒸馏 |
| [RoDI，CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.html) | evidence/belief/uncertainty 衡量、模态滚动、按高 belief token 进行局部去噪；三路共享类型 ViT 编码器 | 可靠性驱动动态融合和去噪已有强先例；其不是 CNN/Transformer/Mamba 三个异构专家，也无置信度选择互蒸馏 |
| [PMKD，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38338) | 两阶段渐进 KD；困难样本挖掘自适应调节蒸馏强度；最终仅学生推理 | “渐进 KD”和“难度自适应 KD”已有先例；PMKD 是冻结 source→target 的顺序单向迁移，并刻意用同构 source/target 降低能力漂移，不是在线对等异构互教 |
| [DAKD，Knowledge-Based Systems 2025](https://www.sciencedirect.com/science/article/pii/S0950705125015643) 与 [MDPR](https://arxiv.org/abs/2401.06430) / [代码](https://github.com/KuilongCui/MDPR) | 前者在可见光–红外 ReID 中进行 confidence-based selective masking；后者在普通 ReID 中双分支互蒸馏 | “置信度选择”和“互蒸馏”分别已有先例；目标必须定义逐样本教师方向、校准、stop-gradient 和双方不可靠时的拒绝规则 |
| [CTMambaFuse，Infrared Physics & Technology 2026](https://www.sciencedirect.com/science/article/pii/S1350449526002185) | 红外–可见图像融合中组合 CNN、Transformer 和 Vision Mamba | 不属于 ReID，但足以否定跨任务的宽泛“首次组合三种架构”表述 |

### 3.1 最近六个月补充碰撞与独立复核

2026-08-31 的第二轮查新进一步找到四组会改变方案定位的主源：

- [MRUF](https://arxiv.org/abs/2607.10599) 已用 leave-one-out error
  increase 监督模态路由，并用逐模态不确定性校准融合；
- [TIER-MoE](https://arxiv.org/abs/2607.27289) 已用 out-of-fold prediction
  学习条件模态风险，再结合 expert-subspace compatibility 路由；
- [TMUR](https://arxiv.org/abs/2604.09288) 指出独立分支 evidence 数值没有
  可比性保证，统一 global router 也仍需要共同目标与校准约束；
- [TIGER](https://arxiv.org/abs/2606.15765) 已用排除异构视觉专家后的预测
  变化做 counterfactual routing alignment。

蒸馏侧，[Adaptive Teacher Modality Selection](https://www.sciencedirect.com/science/article/pii/S1568494626004497)
已按模态贡献动态选择强教师，[CoReTrack](https://www.sciencedirect.com/science/article/pii/S0957417426009942)
已把可靠性先验与非对称教师调控蒸馏结合。因此，head-level LOO、global
router、causal expert exclusion、动态教师和 rejectable KD 都不能单独写成
首次贡献。

独立 GPT-5.5 xhigh 对抗式复核给出 HFER 5.0/10、原 URGC 4.0/10、RDPT
3.0/10、统一系统 5.5/10，判定 **PROCEED WITH CAUTION**。随后三次实现
就绪性复核经历 **REVISE → REVISE → PASS**。完整结构化结论见
[NOVELTY_CHECK_2026-08-31.md](NOVELTY_CHECK_2026-08-31.md)。据此，v1.1
将 RDPT 降为候选辅助，把可靠性目标改为身份外折、冻结生成器上的真实
完整网络干预，并把“共同尺度学习”和“同一后验跨位置控制”作为一个统一
可靠性贡献的两部分。最终 PASS 只确认规范可实现，不证明论文主张。

## 4. 三个可支撑论文、但必须按窄边界验证的创新点

### C1. 全模态异构三专家与分阶段双向交换

CNN、Transformer、Mamba 都应接收全部 RGB/NIR/TIR 输入或同一组对齐的多模态 token，而不是一类网络绑定一种传感器。在匹配深度重复执行尺度对齐后的双向消息传递，同时保留每个专家的私有残差通道。这样才能把“架构归纳偏置互补”与“模态互补”解耦，并与 PFNet 的模态渐进融合、FusionReID 的双架构互联、MambaPro/PRISM 的聚合式 Mamba 区分。

必要对照：无交换、只末端融合、单向交换、一次交换、同构三分支、随机架构分配，以及严格等参数/等 FLOPs 的单骨干和 ensemble。

### C2. 身份外折、实际干预的共同尺度可靠性学习

候选 CIRC 先训练三个 HFER-uniform 外折 target generator；每个生成器只为
未参与其训练的身份产生 expert×modality target。冻结生成器后，对九项
贡献全量执行 total/direct/relay 移除并完整重跑；relay edge 只按冻结哈希
每个 query-condition、每 stage 抽一条作 audit-only。以跨相机固定 reference
bank 上正确 ID margin 的逐条件 helpful-vs-not-helpful 结果监督一个共享
输出、共享温度/证据质量的 global Beta router。neutral/harmful 仅作有符号
审计；cheap head-only LOO 只能作为与完整干预验证过的辅助量。

必要对照：普通 softmax、端到端 observational target、head-only LOO、
九个独立 evidence heads、TMUR-style global non-evidential router、TIGER
式 expert exclusion、shuffled target、expert permutation、身份/相机泄漏
探针、等容量 context-free floor，以及逐条件/相机/身份频率分组
BCE/Brier/ECE、过度离散、经验 concentration coverage 和 cluster effective
sample size。默认控制只使用 posterior mean \(r\)；\((1-u)\) 只有 coverage
过门后才可晋级。

### C3. 同一个干预校准后验贯穿中继、融合与退化控制

URGC 让同一 CIRC 后验决定 HFER 消息通过量、最终 expert×modality
融合权重，以及缺失/退化抑制。这里可检验的贡献是“同一可信信号形成控制
闭环”，不是 Beta gate 或 dynamic fusion 本身。

必要对照：同一路由器处处复用、relay-only、fusion-only、三个独立的
等容量路由器、UGG/RoDI 风格门控、缺失/低质量分离评估，以及 fused 和
三个独立分支的检索变化。若三个独立路由器不差于单一后验，统一控制主张
必须删除。

RDPT 保留为训练辅助候选，不再预先列为主创新。只有其显著超过 symmetric
KL、固定教师、HeteroAKD/MST-Distill 风格自适应迁移和 wrong-payload
对照，并在不造成 CKA 坍缩的前提下改善弱分支，才允许晋升。

三点组成的统一论证链改为：**完整异构专家的深层互促 → 外折实际干预的
可靠性学习 → 同一可靠性后验的跨位置控制**。C2 与 C3 是一个统一可靠性
贡献的“如何学”和“如何用”，论文中不得伪装成两个互不相关的首次机制。

## 5. 论文结果门槛与复现清单

在使用“SOTA”措辞前，至少满足：

- 固定现代 RGBNT201 协议：`train_171`、全 `test` query/gallery、同身份同 camera 排除；
- 明确 CLIP、DINOv2 或 DINOv3 的具体权重版本，输入分辨率和预训练来源；
- 明确是否使用额外文本、语义掩码、教师模型、生成模型、reranking 或测试时更新；
- 对完整模型和每个创新点至少运行 3 个独立种子，报告均值、标准差和最佳值；
- 提供等参数/等 FLOPs、相同预训练和相同训练预算的公平控制；
- 发布 split/file manifest、配置、环境锁、checkpoint、日志和一次端到端评估命令；
- 将静态、额外语义资源、多阶段 KD 和 TTT 分成不同赛道，不跨赛道宣称公平优越。

## 检索边界

检索覆盖 RGBNT201、multi-modal/multi-spectral object ReID、CNN–Transformer ReID、Mamba ReID、uncertainty/reliability fusion、progressive/selective/mutual knowledge distillation，并追踪 AAAI、CVF、NeurIPS、PMLR、期刊官网、arXiv 作者稿及官方 GitHub。2026-08-31 的补充复核逐项检查了 CoT-ReID、NEXT、STMI、PRISM、FUSE、Signal、PEFT-BoA、ICPL-ReID、MGRNet、UPCL 和 Hyper-ReID 的论文/作者稿与官方仓库；通过 GitHub 官方接口核对源码树、默认分支、release 和 HEAD，并把可引用仓库固定到本文链接中的 commit。领域清单与 arXiv `RGBNT201` 查询还用于发现候选，但所有性能结论都回到作者论文和官方仓库核验。由于否定性结论无法由有限检索证明，投稿前仍应重新检查 2026-08-31 之后的论文、在线优先期刊和新增代码发布，尤其是目前只有占位仓库的 Hyper-ReID。
