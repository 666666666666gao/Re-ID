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
服务器当前数据根仅安装 RGBNT201；RGBNT100/MSVR310 的本项目全训练/全评估
尚未执行。完整 SOTA 目标仍未达到。
