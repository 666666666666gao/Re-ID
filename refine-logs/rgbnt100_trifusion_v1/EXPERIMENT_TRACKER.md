# RGBNT100 原完整三角色实验跟踪

更新时间：2026-09-06T13:56:31.328098+08:00。B0完整核验；新三角色 **REGISTERED_NOT_RUN**。
合同[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)，seed42、远端运行，主目标后才消融。

| 阶段 | 状态 | 固定范围 |
|---|---|---|
| Signal R2 B0 | COMPLETE_EXECUTOR_VERIFIED | 三折30epoch/7794更新/8675 query；89.5241750420 mAP/96.8299711816 R1 |
| 源码/四核验器 | PREPARED | 26实际远端源文件，五已有CRLF差异记录；旧124行FP32回归已过 |
| 原三角色 M0 | REGISTERED_NOT_RUN | 三fold各8步+fresh fold0固定100步，124更新，0heldout |
| 三fold完整比较 | NOT_RUN | M0核验PASS后新初始化20epoch，43375 query-output/50身份 |
| 独立审计 | UNAVAILABLE_SERVICE_LIMIT | 无本数据集三角色最终verdict |
| 官方/消融/多seed | NOT_RUN | 既有主目标和用户限制保持 |

静态预测trainable6248460/6248460/6274572，M0实际值待核对。
未重训B0、未加科学模块或损失、未扫描已封存失败版本。
