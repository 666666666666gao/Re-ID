# Global身份与角色任务职责配对结果

| 数据集 | 条件 | best轮 | mAP | R1 | 正式步数 |
|---|---|---:|---:|---:|---:|
| RGBNT201 | semantic | 8 | 74.3749 | 78.8278 | 2649 |
| RGBNT201 | native | 8 | 74.4182 | 79.0670 | 2649 |
| MSVR310 | semantic | 38 | 50.5422 | 67.8511 | 706 |
| MSVR310 | native | 38 | 50.5523 | 68.0203 | 706 |
| RGBNT100 | semantic | 5 | 84.0903 | 95.8601 | 3129 |
| RGBNT100 | native | 26 | 83.1632 | 96.7930 | 3129 |

Seed42 exploratory development on consumed official benchmarks. Same inference/capacity/initial state/author recipe, global versus fused author-task gradient ownership changed. Reused detached and V6 formal controls are historical, no retired M0 or report replay. Head BN updated once from global; fused head uses cloned buffers and detached values. Identity statistics do not establish seed stability; no universal causal/novelty/SOTA claim.
