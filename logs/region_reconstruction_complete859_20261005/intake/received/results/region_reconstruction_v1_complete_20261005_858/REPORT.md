# 区域候选重建：patch查询与均值广播对照

| 数据集 | 查询 | best轮 | mAP | R1 | 正式步数 |
|---|---|---:|---:|---:|---:|
| RGBNT201 | patch | 8 | 74.3398 | 79.4258 | 2649 |
| RGBNT201 | mean | 8 | 74.6785 | 79.5455 | 2649 |
| MSVR310 | patch | 38 | 50.5429 | 67.8511 | 706 |
| MSVR310 | mean | 38 | 50.5421 | 67.8511 | 706 |
| RGBNT100 | patch | 26 | 83.4992 | 96.6181 | 3129 |
| RGBNT100 | mean | 26 | 83.5584 | 95.7434 | 3129 |

Single seed42 on consumed official benchmarks. Same active parameters and original RAW-semantic pipeline; patch-specific routing versus broadcast mean context. No text resources or full SAGA reproduction; no seed stability, true-part correspondence or SOTA claim.
