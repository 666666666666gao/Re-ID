# 区域候选重建 v1 追踪

前置研究：§41.857容量三端及唯一报告已闭合，九配对推进0/9；当前raw9控制复用，不重新训练。后继不是原生/容量方案晋级。

| 数据集 | 查询 | CPU组件 | 真实初始化 | 8更新M0 | fresh50 | 首次strict | 状态 |
|---|---|---|---|---|---|---|---|
| RGBNT201 | patch | PASS | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PREPARED |
| RGBNT201 | mean | PASS | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PREPARED |
| MSVR310 | patch | PASS | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PREPARED |
| MSVR310 | mean | PASS | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PREPARED |
| RGBNT100 | patch | PASS | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PREPARED |
| RGBNT100 | mean | PASS | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | PREPARED |

CPU实际见证：105232参数/15张量，两查询在相同合成数据八更新中全活动/变化；零出口初始相同及strict组件重载通过。不能当作模型M0或检索性能。独立上下文源码复核尚待完成；6端初始化/M0/正式训练均未启动。本阶段总计15配对，报告尚未调用。

只用26 GPU0/1，不查询功率/温度；旧失败不改判，Goal ACTIVE/UNMET。

## 登记完成

2026-10-05T15:47:18.958102+08:00：CPU及独立源码复核完成，SOURCE_SCOPE已登记；真实NN各端仍NOT_RUN。九旧候选清理完成、当前187/345保留，原两次清理异常分别有记录。


## §41.859 终态（覆盖上文准备状态）

| 数据集 | 查询 | best轮 | mAP | R1 | R5 | R10 | best→末轮mAP下降 |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | patch | 8 | 74.339751 | 79.425836 | 88.157892 | 91.626793 | 2.790450 |
| RGBNT201 | mean | 8 | 74.678481 | 79.545456 | 88.516748 | 91.866028 | 3.285612 |
| MSVR310 | patch | 38 | 50.542936 | 67.851102 | 80.541456 | 85.448390 | 0.059974 |
| MSVR310 | mean | 38 | 50.542135 | 67.851102 | 80.541456 | 85.448390 | 0.060913 |
| RGBNT100 | patch | 26 | 83.499203 | 96.618074 | 97.084546 | 97.609329 | 1.142334 |
| RGBNT100 | mean | 26 | 83.558437 | 95.743442 | 96.384841 | 96.967930 | 1.027495 |

六端fresh50/首次strict接受、原15配对一次完成；推进0/15，候选重建不晋级。原父磁盘EXIT1与最后首次训练的行政补完EXIT0分别保留。
