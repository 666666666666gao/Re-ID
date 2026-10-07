# 新增证据目标对照

| 数据集 | 目标 | best轮 | mAP | R1 | 正式更新 |
|---|---|---:|---:|---:|---:|
| RGBNT201 | md_batch_ratio | 8 | 74.4694 | 79.1866 | 2649 |
| RGBNT201 | repair_keep | 8 | 74.6521 | 79.6651 | 2649 |
| MSVR310 | md_batch_ratio | 38 | 50.5542 | 68.0203 | 706 |
| MSVR310 | repair_keep | 38 | 50.5446 | 67.8511 | 706 |
| RGBNT100 | md_batch_ratio | 5 | 83.9106 | 95.6268 | 3129 |

Training-only mechanism comparison on consumed official benchmarks. No new parameters or inference changes. MD ratio is an adaptation, not full author reproduction; repair/keep formula and upstream responsibility are not novel. Complete curves/E50 and all legal queries retained; project gate is not significance, multi-seed stability, P1/P2/P3 or SOTA proof.

Missing formal endpoint: RGBNT100 repair_keep; M0 third-role query/key auxiliary gradient sums were zero. No score or checkpoint is substituted.
