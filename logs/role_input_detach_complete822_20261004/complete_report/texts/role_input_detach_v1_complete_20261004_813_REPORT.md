# 角色读取梯度边界配对结果

| 数据集 | 条件 | best轮 | mAP | R1 | 正式步数 |
|---|---|---:|---:|---:|---:|
| RGBNT201 | semantic | 18 | 72.7798 | 76.9139 | 2649 |
| RGBNT201 | native | 8 | 69.4305 | 72.1292 | 2649 |
| MSVR310 | semantic | 49 | 50.7851 | 68.8663 | 706 |
| MSVR310 | native | 38 | 51.1388 | 69.8816 | 706 |
| RGBNT100 | semantic | 5 | 81.9418 | 94.7522 | 3129 |
| RGBNT100 | native | 5 | 82.5124 | 95.8017 | 3129 |

Seed42 exploratory development on consumed official benchmarks. Same capacity/forward/initial state/author recipe, role-read gradient path intervention only. Reused V6 controls are historical, not concurrent reruns or new seeds. Identity bootstrap is not seed uncertainty. Shared/global path remains trainable, no strict global preservation or universal causal/novelty/SOTA claim.
