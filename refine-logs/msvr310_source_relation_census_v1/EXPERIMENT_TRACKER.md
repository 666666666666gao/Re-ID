# 来源关系普查跟踪

| Run ID | 阶段 | 范围 | 状态 |
|---|---|---|---|
| MSVR-SRC-1 | CPU合同和数学检查 | 全部source索引、标签与几何供体支持 | PREPARED_NOT_RUN |
| MSVR-SRC-1 | 冻结前向 | 三折×三模型×三输入，18576条记录 | PREPARED_NOT_RUN |
| MSVR-SRC-1 | 全量CPU复核 | 全部972条件及668736query关系行 | PREPARED_NOT_RUN |

没有优化器更新、没有新official结果；之前的完整配对Q1保持FAIL。


### 2026-09-07 完整执行终态

| 阶段 | 原PID | 终态 |
|---|---:|---|
| CPU合同与数学 | 175889 | PASS，20:21:47退出0 |
| 三折全部冻结前向 | 175895 | 27条件/18576记录/306batch，20:28:38退出0 |
| 全量CPU独立复核 | 176897 | 972条件/668736query/100440身份行，20:37:51退出0 |
| 本地完整文字复核 | 本地 | 64文件57135882B SHA一致，全部query/身份行再汇总PASS |

原wrapper175887已结束，正式pipeline COMPLETE_VERIFIED_SOURCE_DIAGNOSTIC。参考报告results/MSVR310_COMPLETE_SOURCE_RELATION_CENSUS_2026-09-07.md。源训练权重不变、0新训练更新，原Q1_FAIL不改。下一项实例覆盖训练仍未登记/启动。
