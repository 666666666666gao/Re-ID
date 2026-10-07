# 2026-10-07 强参照资源与协议核查

这是四篇论文的有限原始来源更新。数字均为作者报告；没有本项目复现成绩，也没有据此宣布穷尽当前 SOTA。当前五端训练合同保持不变。本报告先存于私有目录，OWN NN 终态后才同步到项目。

| 方法 | 视觉与额外资源 | RGBNT201 mAP / R1 / R5 / R10 | RGBNT100 mAP / R1 | MSVR310 mAP / R1 | 论文轮数与代码状态 |
|---|---|---|---|---|---|
| DEEP | CLIP ViT-B/16；冻结文本编码器及反演提示 | 79.6 / 84.2 / 89.4 / 91.5 | 88.5 / 97.6 | 66.0 / 82.1 | 60；当前仓库无训练／评价源码 |
| RoDI—CLIP | CLIP ViT-B/16 | 84.1 / 87.2 / 92.0 / 93.2 | 88.5 / 97.6 | 64.1 / 77.2 | 总轮数未核到；仓库只有 README、论文和海报 |
| RoDI—DINOv3 | 蒸馏 DINOv3 ViT-B/16 | 85.3 / 87.9 / 93.0 / 94.8 | 89.0 / 99.1 | 71.8 / 84.8 | 同上；不能视为与 CLIP 等预训练资源 |
| DSGM | CLIP；GPT-4o 文本、SAM2 mask | 82.6 / 87.0 / 92.0 / 93.9 | 89.4 / 98.2 | 64.6 / 76.0 | 201 为 60、车辆 50；公开源码存在，当前 201 配置却为 50 |
| CoT-ReID | DINOv3-B；Qwen-VL 文本和冻结 CLIP 文本编码器 | 83.3 / 86.1 / 93.3 / 94.8 | 89.9 / 99.3 | 71.7 / 85.3 | 论文 120；公开配置为 60/60/80，依赖资源未齐 |

DEEP 的文件路径仍含 2025，但 PDF 的 DOI `10.1109/TMM.2026.3660160`、版权和仓库引用均为 2026。MSVR310 过滤同身份且同时间段的匹配；201/100 的精确可执行过滤仍未核到。推理特征仍使用反演提示，不能算仅训练期文本蒸馏。父端递归 API 核查确认，当前提交 `cc177306` 只有说明和图片资产。车辆表的 88.5/97.6 是 RGBNT100 mAP/R1；MSVR310 的 R5/R10 在另一消融表为 90.7/93.9。[作者论文](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf)、[官方仓库](https://github.com/lsh-ahu/DEEP-ReID)

RoDI 的两种骨干分别报告，不能混用 baseline 或资源。仓库提交 `2f38911c` 的完整文件树只有 README、13 页论文含补充材料、海报，没有训练／评价代码或权重。总轮数、准确 query/gallery 及过滤实现尚未明确；论文称推理省去 opinion measurements 和 rolling，但 LMD 的具体执行仍不清楚。补充材料报告的参数、FLOPs、吞吐只作为作者成本参照。[作者主论文](https://aihuazheng.github.io/publications/pdf/2026/2026-Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf)、[官方论文及补充](https://github.com/lsh-ahu/RoDI/blob/2f38911c49d42d4ca259d440a851b8d77dddccbe/assets/RoDI.pdf)

DSGM 当前提交 `6566f786` 已有源码，但 README 入口名与实际 `train.py/test.py` 不同，评价权重路径也写为作者本机路径。推理仍读取 text 和 mask；生成器可离线预处理，并不消除这些依赖。论文写 201 的 141/30/30 身份划分，实际导入的 loader 使用 `train_171`；准确对应待核。代码保留 camera／scene 过滤，但默认评价特征选择与融合诊断入口应分别核对。论文 201 自身消融为 70.3→82.6，不能搬成本项目增量。[论文](https://arxiv.org/html/2607.29207v1)、[固定官方源码](https://github.com/zw-absin/DSGM/tree/6566f78636440c70c85700a45bbf2f48f18260b1)

CoT-ReID 主表的 MSVR310 为 71.7/85.3，补充材料重复该数；主文 Table 3 则写 72.7/86.3，差异保留，不选较高值。它的测试描述和推理链仍参与视觉条件化与文本编码。当前提交 `db215273` 有实质源码，但未包含文本 JSON、模型权重、兼容 DINOv3 实现，也未建立 RGBNT100 可执行入口。论文的 120 轮与现有 YAML 的 60/60/80 不同，不能称精确复现。[CVF 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.pdf)、[官方补充](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Gao_Chain-of-Thought_Guided_Multi-Modal_CVPR_2026_supplemental.zip)、[固定源码](https://github.com/Gaoya615/CoT-ReID/tree/db215273d6ee68b9c324fdf36e3d6800370fa21e)

当前只能将这些工作作为按资源分组的强参照。两篇没有公开可执行实现，两篇存在明确的论文／代码差异；本项目与它们的等协议、等资源比较仍需进一步建立。没有根据这次核查新增文本、mask、DINO 或改变正在执行的损失。

核查使用作者／出版社原文、官方仓库及补充材料；没有项目本地 PDF，Zotero／Obsidian 工具不可用，规范 arXiv helper 未解析到。三份只读论文 shard 提供事实，父端合并去重、保留缺失及冲突，不由 shard 进行 SOTA 或新颖性裁决。`SUMMARY.json` 保存结构化来源、协议和输入 SHA；`REFERENCES.csv` 保存五行参照。
