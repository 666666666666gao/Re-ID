# 近似同容量语义来源控制

| 数据集 | best轮 | mAP | R1 | 正式步数 |
|---|---:|---:|---:|---:|
| RGBNT201 | 8 | 74.3949 | 78.5885 | 2649 |
| MSVR310 | 38 | 50.5499 | 67.8511 | 706 |
| RGBNT100 | 9 | 83.5905 | 96.3265 | 3129 |

Seed42 exploratory comparison on consumed benchmarks, near capacity not exact capacity. Old raw controls reused with full batch-order equality. New semantic MLP plus resized candidates versus image CNN source; no universal causal, novelty, seed-stability or SOTA claim. No normalized metric-stage rescue.
