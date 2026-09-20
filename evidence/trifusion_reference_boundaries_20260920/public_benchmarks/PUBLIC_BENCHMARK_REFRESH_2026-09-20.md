# 公开基准增量核验：2026-09-20

**本次未核实到需要替换既有 RoDI / PMKD / CoT 代表参照的新主表结果。** 这是限定范围的公开文献刷新，不是穷尽排行榜，也不证明更强结果不存在。检索按数据集名、近期日期、已有高指标方法及此前未决项收敛；官方作者页面与 arXiv 元数据只用于定位变化，未用搜索摘要补填论文数字。

执行者为当前 Codex 后台研究 agent；仅具同族模型的上下文隔离，不构成跨模型独立审计，也未独立认证实际后端模型标识。Signal 预算比较不在本任务范围。

以下数值全部为论文报告的 **mAP / Rank-1（%）**，沿用此前完整主源核验。本次没有重下或重新审计这些未见变化的论文；“沿用”不应写成“本日重新复现”。

| 保留的代表参照与资源 | RGBNT201 | RGBNT100 | MSVR310 | 原始主表、版本及位置 |
|---|---:|---:|---:|---|
| RoDI，CLIP ViT-B/16 | 84.1 / 87.2 | 88.5 / 97.6 | 64.1 / 77.2 | [作者 PDF，当前固定提交](https://github.com/lsh-ahu/RoDI/blob/2f38911c49d42d4ca259d440a851b8d77dddccbe/assets/RoDI.pdf)，Table 1，PDF p.6 |
| RoDI，蒸馏版 DINOv3 ViT-B/16 | 85.3 / 87.9 | 89.0 / 99.1 | 71.8 / 84.8 | 同上 Table 1，PDF p.6；不能与 CLIP 行视为相同预训练资源 |
| PMKD，DINOv2、多阶段蒸馏 | 84.7 / 88.9 | 91.6 / 98.0 | 未报告 | [作者完整 PDF](https://aihuazheng.github.io/publications/pdf/2026/2026-Progressive_Multi-modal_Knowledge_Distillation.pdf)，Table 1 / 2，PDF p.6；[AAAI 2026 正式身份页](https://ojs.aaai.org/index.php/AAAI/article/view/38338) |
| CoT-ReID，DINOv3、额外 MLLM 推理文本 | 83.3 / 86.1 | 89.9 / 99.3 | 71.7 / 85.3 | [CVPR 2026 会议 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.pdf)，Table 1 / 2，PDF pp.6–7 |

RoDI 的旧完整文件核验 SHA256 为 `b816237dbbcccda47dbf02b19c9de9816bce05b4f40723ae88ef1286ff663d86`；PMKD 为 `73086d4c318d610fd44e3d7a875462c5ed797cf8c6c2970820291259037eb094`；CoT 为 `e0aff58aefc2c39001bc7b36e69476fe2e24b074d964b0f1ef7237e6a84b0efa`。这些是既有核验记录中的哈希，本次没有重算 PDF 哈希。PMKD 作者文件与出版社 PDF 的逐字节等同关系仍未建立。

CoT 的同一会议 PDF Table 3 另有 MSVR310 **72.7 / 86.3**；此前已记录其与 Table 1 的 **71.7 / 85.3** 差异。本次继续保留主比较行，不择优替换，也不自行解释表间差异。各方法 mAP 与 Rank-1 的较高值不能拼接为一个实际模型结果。

**本日直接核实的发布状态。** 公开 GitHub API 的 HEAD、完整递归树（均 `truncated=false`）、release 与 tag 请求均成功。以下结论限于这些指定作者仓库及核查时点。

| 作者仓库固定 HEAD | 本日完整树所见 | GitHub release / tag | 相对旧记录 |
|---|---|---:|---|
| [RoDI：2f38911c49d42d4ca259d440a851b8d77dddccbe](https://github.com/lsh-ahu/RoDI/tree/2f38911c49d42d4ca259d440a851b8d77dddccbe) | README.md、assets/RoDI.pdf、assets/poster.png | [0](https://api.github.com/repos/lsh-ahu/RoDI/releases?per_page=100) / [0](https://api.github.com/repos/lsh-ahu/RoDI/tags?per_page=100) | 仍为 README 与论文资产；没有树内训练实现或 checkpoint |
| [PMKD：0f597faad5b1432ce37b8be52e9bfac80b259f1f](https://github.com/moonaricc/PMKD/tree/0f597faad5b1432ce37b8be52e9bfac80b259f1f) | README.md | [0](https://api.github.com/repos/moonaricc/PMKD/releases?per_page=100) / [0](https://api.github.com/repos/moonaricc/PMKD/tags?per_page=100) | 仍未取得训练实现或 checkpoint |
| [Hyper-ReID：6e895a707c0948d03968b3e812ec6cf5fbcd1eb9](https://github.com/lsh-ahu/Hyper-ReID/tree/6e895a707c0948d03968b3e812ec6cf5fbcd1eb9) | README.md；题名、作者及 ACM MM 接收声明 | [0](https://api.github.com/repos/lsh-ahu/Hyper-ReID/releases?per_page=100) / [0](https://api.github.com/repos/lsh-ahu/Hyper-ReID/tags?per_page=100) | 在所核作者入口仍无主表数字，继续未决 |

具体 API URL、提交时间、tree SHA、每项 blob SHA 与大小已写入同目录的 [sources_receipt.json](sources_receipt.json)。这比“页面能打开”更具体，但仍不等于可运行性或复现验证；仓库的 README、论文 PDF 也不能替代训练实现。

**其余有界补查与未知项。** [PRISM](https://arxiv.org/abs/2607.23451)、[DSGM](https://arxiv.org/abs/2607.29207)、[MODAL](https://arxiv.org/abs/2608.15096) 的当前 arXiv submission history 均仍只列既有 v1，因此不重做旧主表审计。原记录中的 PRISM 分割掩码、DSGM 生成文本与掩码、MODAL 生成文本，以及论文／已发布配置差异继续保留；arXiv 版本状态不能证明出版社最终稿完全相同。

[CCL 的 ICML 官方页](https://icml.cc/virtual/2026/poster/65854) 经直接 HTTP 请求成功取得，仍提供摘要及 OpenReview 入口，本次未发现该页的主表或额外论文 PDF。web 工具先报 restricted URL，随后直接读取成功，两种结果都留存。[OpenReview PDF](https://openreview.net/pdf?id=9RI8vsWsqv) 仍转到浏览器验证页；公开 notes API 返回 HTTP 403 / ChallengeRequiredError。因此 CCL 主表继续未知，没有采用第三方“复现”网页或摘要生成数字。[作者主页](https://sunyuan-cs.github.io/)与会议记录可以证明论文身份，不能证明表中成绩。

作者出版目录另指向 [REMIND 官方仓库](https://github.com/skye-1201/REMIND)。本次只读到 README 中的实现入口与模态缺失／完整模态任务说明，没有该页的性能表；没有把代码公开日期当作本次新发布，也没有据此添加更强结果或开展源码审计。

比较边界保持：RoDI 的 CLIP 与 DINOv3、PMKD 的 DINOv2 分列；CoT 的生成推理文本、[STMI 的额外语义资源](https://ojs.aaai.org/index.php/AAAI/article/view/38125)、PRISM / DSGM 的掩码条件须随方法保留。[ProxyTTT](https://ojs.aaai.org/index.php/AAAI/article/view/38337) 的测试时适配与 [AutoSOTA 所述后处理调参](https://github.com/tsinghua-fib-lab/AutoSOTA/blob/main/README.md) 仍属于不同条件；本次未重核后者的实验链，也不把它用作静态、无重排序参照。公开官方 benchmark 报告不能与 TriFusion 内部 identity-disjoint 协议数字相减。

本次只生成这一份说明及 JSON 来源回执。没有访问远程服务器、凭据、数据集或官方测试图像，没有下载权重／安装实现／运行模型，也没有修改仓库、实验、配置或选择门槛。

沿用的本地证据：`docs/SOTA_PRIMARY_REFRESH_2026-09-07_EVENING.md`、`docs/PMKD_AUTHOR_PDF_VERIFICATION_2026-09-07.md`、`docs/PRISM_DSGM_PRIMARY_AND_RELEASE_BOUNDARIES_2026-09-08.md`、`docs/SOTA_PRIMARY_REFRESH_2026-09-06.md`，以及 CoT 主表来源所在的 `docs/SOTA_REFRESH_2026-09-05.md`。这些文件本日的 SHA256 与所用主源链接均已记入回执。
