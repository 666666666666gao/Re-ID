# RGB–NIR–TIR ReID 主源增量核查（2026-09-05）

以下是本次定位并核对的公开报告值，不是本服务器复现值，也不是对所有未公开
工作的穷尽证明。`mAP / Rank-1` 均为百分数。训练集、骨干、额外语义资源、
测试时更新及 checkpoint selection 不同的行不能直接作为公平同协议胜负结论。

| 数据集 | 公开高指标参照 | mAP / Rank-1 | 条件与主源 |
|---|---|---:|---|
| RGBNT201 | RoDI-DINOv3 | 85.3 / 87.9 | 静态 DINOv3 ViT-B/16；[作者 PDF Table 1, p6](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf) |
| RGBNT201 | RoDI-CLIP | 84.1 / 87.2 | 静态 CLIP ViT-B/16；同上 |
| RGBNT201 | PMKD-DINOv2 | 84.7 / 88.9 | 多阶段知识蒸馏；[AAAI 2026 Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/38338/42300)；Rank-1 高于上述 RoDI 行，不能以超过85.3/87.9声称两项全局最高 |
| RGBNT100 | PMKD-DINOv2 | 91.6 / 98.0 | [AAAI 2026 Table 2](https://ojs.aaai.org/index.php/AAAI/article/download/38338/42300)；本次核到的最高 mAP 参照 |
| RGBNT100 | RoDI-DINOv3 | 89.0 / 99.1 | [作者 PDF Table 1](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf)；mAP/Rank-1 最优不来自同一方法 |
| MSVR310 | RoDI-DINOv3 | 71.8 / 84.8 | [作者 PDF Table 1](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf)；本次核到的高 mAP 静态纯视觉参照 |
| MSVR310 | RoDI-CLIP | 64.1 / 77.2 | 同上；与 DINOv3 分开 |

Signal 作者仓库仍列出 RGBNT201 `80.3/85.2`、RGBNT100 `86.3/97.6`、
MSVR310 `53.2/72.4`，且提供 MIT 许可证和代码。
来源：[Signal README](https://github.com/010129/Signal)。本项目同协议开发 baseline
为 `58.0109/57.4545`，其141-fit/30-dev结果不应直接与上述官方测试报告值相减。

另外，ProxyTTT 的 RGBNT201 `85.0/88.5`、RGBNT100 `89.3/97.7`、MSVR310
`63.6/72.1` 包含测试时训练；其静态 w/o TTT 对应为 `82.3/84.7`、
`88.4/97.9`、`62.1/71.7`。[AAAI 2026 Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/38337/42299)

资源状态复核：

- [RoDI](https://github.com/lsh-ahu/RoDI) 当前仅 README/assets，PDF 可下载；
  没有可直接接入本项目的训练实现和 checkpoint。本次 CVF 页面403，已从作者
  GitHub PDF 直接提取第6页表格核验，没有用搜索摘要替代该表。
- [Hyper-ReID](https://github.com/lsh-ahu/Hyper-ReID) 仍只有一份 README 与
  ACM MM 2026 录用声明，无公开指标。它仍是最终 SOTA 表述前的未决项。
- [PRISM 作者稿](https://arxiv.org/html/2607.23451v1) 与
  [作者仓库](https://github.com/zw-absin/PRISM) 可研究 token/前景处理，但论文使用
  OpenPifPaf/SAM2 派生掩码。其 RGBNT201 `80.5/84.0`、RGBNT100 `86.1/97.8`、
  MSVR310 `47.6/64.8` 不是上述数值上限，额外掩码也必须单列资源条件。

本次仅补充主源和数值边界，不下载新数据、不改 backbone、不引入第三方模块。
本次后续已从ICPL-ReID作者链接获取并安装MSVR310，通过SHA/CRC及三模态配对
核验；RGBNT100也已完成SHA/CRC、全模态配对和拼接尺寸检查。二者本项目全训练/全评估尚未执行，完整SOTA
目标仍未达到。数据状态详见主交接文档§32。

## 11:14 增量：CoT-ReID、DSGM与代码可用性

进一步核对了[CoT-ReID作者仓库](https://github.com/Gaoya615/CoT-ReID)及
[CVPR 2026主论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.pdf)
的Table 1/2（PDF第6/7页），补充以下论文主比较值：

| 方法与资源 | RGBNT201 mAP/R1 | RGBNT100 mAP/R1 | MSVR310 mAP/R1 |
|---|---:|---:|---:|
| CoT-ReID，DINOv3与MLLM推理文本 | 83.3/86.1 | 89.9/99.3 | 71.7/85.3 |
| DSGM，CLIP、GPT-4o文本及SAM2软mask | 82.6/87.0 | 89.4/98.2 | 64.6/76.0 |

DSGM数值来自[作者arXiv v1的Table I/II](https://arxiv.org/html/2607.29207v1#S4.T1)，
额外文本/mask资源见同文§IV-A。[DSGM作者仓库](https://github.com/zw-absin/DSGM)
现已含训练、模型实现和MIT许可证，README列出附mask数据链接。

因此，上表之前的RGBNT100 `99.1 Rank-1`与MSVR310 `84.8 Rank-1`不能当作本次
已核文献的最高Rank-1；CoT主表分别为99.3和85.3。已核mAP高参照仍为RGBNT201
RoDI 85.3、RGBNT100 PMKD 91.6、MSVR310 RoDI 71.8。不同资源方法必须分列，
不能把不同方法的mAP和Rank-1拼成一个实际checkpoint结果。

CoT主表的MSVR310行为71.7/85.3；其Table 3文本替换实验另有72.7/86.3，
本表采用主比较Table 1，保留该表间差异，不自行解释或择优合并。
CoT仓库README明确其文本与预训练权重不随仓库提供；TRAINING.md仍有作者
本地训练包路径，不能把这些路径当成本机现成资源。本次未接入其模块。

[PMKD作者仓库](https://github.com/moonaricc/PMKD)本次仍只有README，无训练实现。
另发现ICML 2026 CCL，作者[主页](https://sunyuan-cs.github.io/)可核其录用记录，
但本次OpenReview正文请求返回浏览器验证页，尚未核得其主表数字；它与
Hyper-ReID继续作为最终SOTA措辞前的待核项。现有高参照不构成穷尽证明。

CoT PDF已从CVF直接下载核读，7178132字节，SHA256
`e0aff58aefc2c39001bc7b36e69476fe2e24b074d964b0f1ef7237e6a84b0efa`。
本次只补文献/代码资源状态，不改变已冻结的V18训练和晋级条件。
