# 公开指标与资源条件补查（2026-09-06）

核查时间：2026-09-06T19:15:51.076311+08:00。补充既有SOTA_REFRESH_2026-09-05.md；本页不是穷尽排行榜，也不改变已登记训练/科学门。

| 方法与依据 | RGBNT201 mAP/R1 | RGBNT100 mAP/R1 | MSVR310 mAP/R1 | 必须注明的条件 |
|---|---:|---:|---:|---|
| ProxyTTT，作者仓库模型表 | 85.0/88.5 | 89.3/97.7 | 63.6/72.1 | 测试时适配，单列于固定参数检索 |
| STMI，AAAI原表索引文本 | 81.2/83.4 | 89.1/97.1 | 64.8/76.1 | CLIP，额外GPT-4o描述与SAM2 mask |
| PMKD，AAAI Table2索引与既有归档一致 | 本轮不重填 | 91.6/98.0 | 本轮未核得 | DINOv2，多阶段蒸馏；本轮未成功渲染完整PDF |

[ProxyTTT作者仓库](https://github.com/liuzhaojun-zwd/ProxyTTT)直接列出以上模型表；[AAAI论文页](https://ojs.aaai.org/index.php/AAAI/article/view/38337)说明PESA测试时适配。其分数须注明目标域适配成本，不能与本项目0测试时优化的固定模型当作完全同条件。

[STMI的AAAI原文](https://ojs.aaai.org/index.php/AAAI/article/download/38125/42087)本轮索引返回Table2及实验设置，报告GPT-4o文本与SAM2前景mask。该额外资源条件解释为何不能只按“CLIP方法”归为等资源组；论文讨论前景调制、token重分配和超图交互。本轮未执行其代码或获取新mask。

[PMKD的AAAI原文](https://ojs.aaai.org/index.php/AAAI/article/download/38338/42300)本轮搜索索引再次返回Table2的91.6/98.0及DINOv2符号说明，与既有归档一致；完整文档open仍Internal Error，不能称为新的完整PDF/截图独立核验。该参照继续保留，不把89.9当最高已知mAP。

## 新发现的开源入口与证据边界

[PRISM作者仓库](https://github.com/zw-absin/PRISM)标注IEEE TIP2026、MIT，并注明基于MambaPro，提供CLIP与mask数据、训练/评估入口。
[DSGM作者仓库](https://github.com/zw-absin/DSGM)标注IEEE TCSVT2026、MIT，并注明基于IDEA，同样提供mask数据与训练入口。
本轮只直接读了两个作者README/仓库页面，未读核心模型、核对完整论文主表或执行复现，因而不填其成绩，也不写成已可直接接入本项目。后续需先核对源码机制、真实许可证和额外mask依赖，再决定是否借鉴；当前固定主训练不加入它们。

[AutoSOTA作者仓库记录](https://github.com/tsinghua-fib-lab/AutoSOTA/blob/main/README.md)另报告MDReID在RGBNT201通过k-reciprocal重排序及特征缩放从82.1/85.2提高到93.6/91.6，涉及调k1/k2/lambda。本轮只核得该README声明，未核其原始训练/评估记录；它属于另一个后处理/调参实验条件，不作为本项目固定静态门的等条件数值，也不据此允许对官方测试扫描参数。

## 对当前实验的直接影响

本项目RGBNT100内部三折91.316540仍不可与官方91.6相减。当前full50 Signal固定30epoch正在训练；
随后原三角色fresh固定20epoch，两端终点固定后才执行官方1715query/8575gallery全部比较。
同源Signal增益、静态同类公开竞争力、额外预训练/文本/mask/测试时适配条件分别报告。
本轮补查没有确认更高的RGBNT100静态主表mAP，不能据有限检索声称绝对最高值已穷尽。
RGBNT201与MSVR310既有科学负结果保持；新文献思路不改写这些失败证据。
