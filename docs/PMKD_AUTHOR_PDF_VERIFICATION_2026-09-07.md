# PMKD作者PDF原表核验（2026-09-07）

通过[作者主页](https://aihuazheng.github.io/publications/)取得
[完整作者PDF](https://aihuazheng.github.io/publications/pdf/2026/2026-Progressive_Multi-modal_Knowledge_Distillation.pdf)：
12,158,023字节、9页，SHA256
73086d4c318d610fd44e3d7a875462c5ed797cf8c6c2970820291259037eb094。
[AAAI正式页面](https://ojs.aaai.org/index.php/AAAI/article/view/38338)确认题名、作者、DOI与13351–13359页。
本轮提取全部9页文字，并实际查看渲染后的PDF第5、6页。
AAAI服务器下载仍断连，因此不声称已证明作者文件与出版社PDF逐字节相同。

| 数据集 | mAP | Rank-1 | 直接位置 |
|---|---:|---:|---|
| RGBNT201 | 84.7 | 88.9 | 第6页Table1 |
| RGBNT100 | 91.6 | 98.0 | 第6页Table2 |
| WMVeID863 | 71.9 | 79.5 | 第6页Table2 |

这补齐此前仅有原表归档/索引复核的证据缺口，原RGBNT201/RGBNT100数值未变。
全文实验使用以上三个数据集，未报告MSVR310，不能用WMVeID863一列补填MSVR310。
它仍是公开高指标参照，不构成已穷尽全部论文的绝对排行榜，也不是本机复现。

第5页Implementation Details写明DINOv2预训练、224×224、batch32（4身份×8实例）、
Adam初始lr4.5e-5和50epoch；本轮未核得多阶段实际总更新数。
这些资源与训练条件需随公开数值保留，不能把其差距单独归因于batch大小。
[作者仓库](https://github.com/moonaricc/PMKD)本轮仍只见README，没有取得训练实现。

PDF、渲染页面与完整提取文字仅保留于本地临时研究目录，未上传项目仓库或占用训练盘；
仓库只保存本文与来源/数值/哈希回执。没有新增模型/数据/训练/测试，没有改变V28 R2配置或门槛。
已有SOTA_REFRESH与SOTA_PRIMARY_REFRESH作为历史记录保留，由本页补充当前证据层级。
