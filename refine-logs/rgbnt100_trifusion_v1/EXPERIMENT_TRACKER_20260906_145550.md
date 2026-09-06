# RGBNT100 原完整三角色实验跟踪

更新时间：2026-09-06T14:55:50.366010+08:00。最近实际进度14:49:35：fold0完整20epoch/1653更新/3125图库五输出评估完成，B0逐元素相同；fold1已127步，合计1780步均有效，0 AMP下降。整体仍RUNNING，无三fold终态；下一阶段15:18，预计15:55–16:10。

| 阶段 | 状态 | 固定范围 |
|---|---|---|
| Signal R2 B0 | COMPLETE_EXECUTOR_VERIFIED | 三折30epoch/7794更新/8675 query；89.5241750420 mAP/96.8299711816 R1 |
| 原三角色 M0 | PASS_ENGINEERING_ONLY | 124更新/7936source曝光，203/203梯度；excess ratio0.001551768 |
| M0 files/scalars | PASS_COMPLETE | 3权重/31项目21Signal绑定/全部124精确FP32损失；23文本共5451771字节 |
| 三fold完整比较 | RUNNING | bbe49e1；fresh三fold各20epoch；43375 query-output/50身份；fold0完成/第二折训练中；下一检查15:18，预计15:55–16:10 |
| 独立审计 | UNAVAILABLE_SERVICE_LIMIT | 无本数据集新独立verdict |
| 官方/消融/多seed | NOT_RUN | 既有主目标和用户限制保持 |

M0实际14:05:20.261521完成，238.8160716秒；首观察14:06:33.454192，原raw全部保留。
真实trainable6248460/6248460/6274572，与预测相同。正式训练不加载M0权重。
