# 独立角色训练头完整结果

| 数据集 | best轮 | mAP | R1 | R5 | R10 | 正式更新 |
|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | 8 | 74.19037911 | 78.70813608 | 87.91866302 | 91.50717854 | 2649 |
| MSVR310 | 38 | 50.54300863 | 67.85110235 | 80.54145575 | 85.44839025 | 706 |
| RGBNT100 | 26 | 83.28153653 | 96.44314647 | 97.25947380 | 97.66764045 | 3129 |

| 数据集 | 对照 | ΔmAP | ΔR1 | 首位修复/新增错误 | 身份宏平均ΔAP | 推进 |
|---|---|---:|---:|---:|---:|---|
| RGBNT201 | raw_semantic | -0.18448243 | -0.11961722 | 5/6 | -0.20205792 | FAIL |
| RGBNT201 | raw_global_only | -0.10628447 | -0.23923445 | 2/4 | -0.13486146 | FAIL |
| MSVR310 | raw_semantic | +0.00078372 | +0.00000000 | 0/0 | +0.00064261 | FAIL |
| MSVR310 | raw_global_only | +0.00087977 | -0.16920474 | 0/1 | -0.00579684 | FAIL |
| RGBNT100 | raw_semantic | -0.80876853 | +0.58309038 | 45/35 | -0.62003934 | FAIL |
| RGBNT100 | raw_global_only | -1.25224769 | -0.17492711 | 32/35 | -0.81084828 | FAIL |

| 数据集 | 额外头参数 | 训练及逐轮评价秒 | 末轮mAP | best到末轮下降 |
|---|---:|---:|---:|---:|
| RGBNT201 | 264192 | 2408.741532 | 71.28952405 | 2.90085506 |
| MSVR310 | 239616 | 1400.970894 | 50.48143126 | 0.06157737 |
| RGBNT100 | 78336 | 7137.304020 | 82.41858508 | 0.86295146 |

Original report already ran once. Local aggregation of all received raw histories/steps only, no new scoring or inference. Step ratios describe training batches, not fixed-best deployment angles or independent correction retrieval. Bootstrap from original report is fixed-model identity uncertainty, not seed variance.
