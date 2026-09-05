# V22 Experiment Tracker

更新时间：2026-09-05T17:45:55.245967+08:00；状态PREREGISTERED_NOT_LAUNCHED。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V22-SOURCE | 全1680batch相机监督元数据与整数复核 | DONE_METADATA_ONLY | evidence/trifusion_source_camera_metadata_20260905.json |
| V22-T0 | 三项CUDA数学/行域/梯度契约 | PENDING | tests/test_trifusion_camera_negative_v22.py |
| V22-M0 | 六模型配对、两端容量、100步过拟合 | PENDING | 冻结EXPERIMENT_PLAN.md |
| V22-Q1 | 三fold两端各20epoch完整检索 | PENDING_M0 | 固定3360优化步、120epoch记录 |
| V22-AUDIT | 独立实际结果审计 | PENDING_RESULTS | 实际回执后执行 |
| V22-D1 | 141-fit refit与30-dev | NOT_QUALIFIED_NOT_RUN | 五项Q1门全通过后登记执行细节 |

V21已独立审计并封存M0_FAIL，不能据其零检索结果判断SAM检索有害。
V22是三个residual度量目标的单一替换，MCNL两个margin均0.1，原残差系数0.25；
原表示/七路ID/融合与单分支triplet、数据、优化器、epoch和seed保持。
当前没有V22 T0/M0/Q1执行结果，也没有新checkpoint/dev/official检索。
元数据不是模型性能。整体目标未完成。
