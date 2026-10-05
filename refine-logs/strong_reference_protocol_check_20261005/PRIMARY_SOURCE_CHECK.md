# 强参照原始来源核查（2026-10-05）

仅核查 RoDI 和 DEEP；以下均为作者论文报告，未作本地复现，也不代表穷尽 SOTA。本次只读文献及公开仓库，只新增此 notes；未访问训练服务器、运行作者模型或调整实验。

## 主表数字（百分比）

| 作者版本 | RGBNT201 mAP / R1 / R5 / R10 | RGBNT100 mAP / R1 | MSVR310 mAP / R1 | 实读原表 |
|---|---|---|---|---|
| RoDI–CLIP ViT-B/16，RoDI* | 84.1 / 87.2 / 92.0 / 93.2 | 88.5 / 97.6 | 64.1 / 77.2 | [CVF 主文 Table 1，PDF p.6 / 印刷 p.6565](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf) |
| RoDI–DINOv3 distilled ViT-B/16，RoDI† | 85.3 / 87.9 / 93.0 / 94.8 | 89.0 / 99.1 | 71.8 / 84.8 | [CVF 主文 Table 1，p.6565](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf) |
| DEEP–CLIP，DEEP† | 79.6 / 84.2 / 89.4 / 91.5 | 88.5 / 97.6 | 66.0 / 82.1 | [作者 accepted TMM PDF Table I，p.6；Table IV，p.7](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) |

所有待核数值均得到原表支持，无数值修正。RoDI 两行必须保留各自预训练条件；DEEP 此处是标准分辨率主表，不能混入高分辨率消融。

## 资源、预算与推理边界

| 项目 | RoDI | DEEP |
|---|---|---|
| 预训练资源 | 两种视觉编码器：CLIP ViT-B/16、DINOv3 distilled ViT-B/16。[§4.2](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf) | CLIP ViT-B/16；视觉分支及共享提示可训练，文本编码器冻结。[§IV.B](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) |
| 额外输入 | §3.1 定义 RGB/NIR/TIR 图像与训练身份标签；按论文架构推断，无需外部文本/掩码。[§3.1 / Fig.2](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf) | 图像经 inversion 网络生成提示，推理保留文本分支；按架构推断，无需外部描述/掩码。词云的外部属性/字幕词汇仅用于分析。[§III.A/B/E、IV.F](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) |
| 训练预算 | 单 RTX4090，batch64、每身份8张；10轮 warmup；总训练轮数 **unknown**。人像尺寸224×224 / 256×128，车辆224×224 / 128×256，按编码器选择。[§4.2](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf) | **60轮**、单 RTX3090Ti；人像256×128、车辆128×256；batch **unknown**。[§IV.B](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) |
| 推理描述 | 融合不依赖 subjective-opinion metrics 和 modality rolling；attention query 是模态融合概念。[§3.1/3.2/3.5](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf) | 图像→伪词→冻结文本编码器是前向生成；测试提示构成预测特征。[§III.B/E](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) |
| 评价条件 | §4.1 引用既有标准协议；本文未核得精确清单/过滤实现。[§4.1](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf) | MSVR310 排除同身份且同时间段的 gallery 项。[§IV.A](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) |

两文的上述推理描述均未说明测试时反向更新或检索 query–gallery 联合适配；这不能证明作者实际执行禁用了这些操作。**实际测试时更新、query–gallery 联合处理、reranking 均为 unknown**：今日完整公开树仍无训练/评价实现。RoDI `2f38911c49d42d4ca259d440a851b8d77dddccbe` 仅 README/PDF/poster；DEEP `cc177306528dd8f8c468de2bb2eeba22978a9fb7` 为10个 metadata/图片文件。[RoDI 固定树](https://github.com/lsh-ahu/RoDI/tree/2f38911c49d42d4ca259d440a851b8d77dddccbe)、[DEEP 固定树](https://github.com/lsh-ahu/DEEP-ReID/tree/cc177306528dd8f8c468de2bb2eeba22978a9fb7)

两项的精确初始化 checkpoint/hash、随机种子数、选点规则、总优化步数/训练时长、query/gallery 文件清单及距离/归一化实现仍 **unknown**；不能据论文数值声称等预算可复现比较。[RoDI 仓库](https://github.com/lsh-ahu/RoDI)、[DEEP 仓库](https://github.com/lsh-ahu/DEEP-ReID)

## 本次来源读取回执

- [RoDI CVF 主 PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf)：web 直取403；原生 HTTPS 成功读取4,879,110字节，SHA256 `745393be9fd1e46e1d6b2804c20c6b82a31498725ea86e169ae7f7885c393fa6`。与已保存原始 PDF 同 SHA；实读其正文提取文本，并视觉查看 p.6 原始页面 Table1/§3.5/§4.2，不以此前笔记代替论文。
- [RoDI 作者 raw PDF](https://raw.githubusercontent.com/lsh-ahu/RoDI/main/assets/RoDI.pdf)：原生 HTTPS 成功5,636,720字节，SHA256 `b816237dbbcccda47dbf02b19c9de9816bce05b4f40723ae88ef1286ff663d86`。与本地作者原文同 SHA，实读 Table1/§4.2 及其 p.6 页面，数字一致。CVF/作者 PDF 字节不同，未声称两者同文件。
- [RoDI CVF 补充材料](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Li_Rolling_and_Denoising_CVPRF_2026_supplemental.pdf)：web403；本次只读已有原始缓存/提取文本，PDF SHA256 `903279991b17b8f9650a27b06ac2bbd02b999c20440bc2ffe2280d9e3394bbf9`。检索 epoch/training 等预算线索未核得总轮数；未用消融替换主表。
- [DEEP 作者 PDF](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf)：web 成功读取16页；实读 TableI/IV、§III.A/B/E、IV.A/B/F；p.1标识 accepted author version / DOI `10.1109/TMM.2026.3660160`，未冒称已读最终 version of record。
- [RoDI 当前完整树 API](https://api.github.com/repos/lsh-ahu/RoDI/git/trees/main?recursive=1)、[DEEP 当前完整树 API](https://api.github.com/repos/lsh-ahu/DEEP-ReID/git/trees/main?recursive=1)：web不可访问，原生 HTTPS JSON 成功；均 `truncated=false`，固定 commit 如上。另实读两项官方仓库 README 页面。

本地 RoDI 原始缓存目录：`C:/Users/gb/.codex_tmp/rodi_primary_source_20261001_713/`。此前发布树记录只用于定位来源；以上数值回到了作者/出版者原文。该文件不含新实验、训练输入或本地 NN 结果。
