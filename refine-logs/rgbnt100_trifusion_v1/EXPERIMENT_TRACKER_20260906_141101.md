# RGBNT100 原完整三角色实验跟踪

更新时间：2026-09-06T14:11:01.774271+08:00。M0完整执行侧核验PASS，正式比较 **REGISTERED_NOT_RUN**。
seed42，原配置/合同及全部五科学门不变。

| 阶段 | 状态 | 固定范围 |
|---|---|---|
| Signal R2 B0 | COMPLETE_EXECUTOR_VERIFIED | 三折30epoch/7794更新/8675 query；89.5241750420 mAP/96.8299711816 R1 |
| 原三角色 M0 | PASS_ENGINEERING_ONLY | 124更新/7936source曝光，203/203梯度；excess ratio0.001551768 |
| M0 files/scalars | PASS_COMPLETE | 3权重/31项目21Signal绑定/全部124精确FP32损失；23文本共5451771字节 |
| 三fold完整比较 | REGISTERED_NOT_RUN | fresh同M0初始SHA，各20epoch，43375 query-output/50身份；估计105–125分钟 |
| 独立审计 | UNAVAILABLE_SERVICE_LIMIT | 无本数据集新独立verdict |
| 官方/消融/多seed | NOT_RUN | 既有主目标和用户限制保持 |

M0实际14:05:20.261521完成，238.8160716秒；首观察14:06:33.454192，原raw全部保留。
真实trainable6248460/6248460/6274572，与预测相同。正式训练不加载M0权重。
