# 同参数跨光谱变化读出：三端完整结果

| 条件 | best | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| raw_semantic | 8 | 74.3749 | 78.8278 | 88.3971 | 91.7464 |
| raw_global_only | 8 | 74.2967 | 78.9474 | 88.2775 | 91.8660 |
| joint_delta | 8 | 74.5331 | 79.1866 | 88.2775 | 91.8660 |
| add_delta | 8 | 74.4934 | 79.4258 | 88.5167 | 91.9856 |
| joint_absolute | 8 | 74.4006 | 79.1866 | 88.2775 | 91.6268 |

| 对照→候选 | ΔmAP | ΔR1 | 修复 | 新增错误 | 身份宏APΔ | 推进 |
|---|---:|---:|---:|---:|---:|---|
| add_delta→joint_delta | +0.0397 | -0.2392 | 2 | 4 | +0.0275 | False |
| joint_absolute→joint_delta | +0.1324 | +0.0000 | 3 | 3 | +0.1336 | False |
| raw_semantic→joint_delta | +0.1582 | +0.3589 | 7 | 4 | +0.1416 | False |
| raw_semantic→add_delta | +0.1186 | +0.5981 | 7 | 2 | +0.1141 | False |
| raw_semantic→joint_absolute | +0.0258 | +0.3589 | 6 | 3 | +0.0080 | False |
| raw_global_only→joint_delta | +0.2364 | +0.2392 | 5 | 3 | +0.2088 | False |
| raw_global_only→add_delta | +0.1968 | +0.4785 | 6 | 2 | +0.1813 | False |
| raw_global_only→joint_absolute | +0.1040 | +0.2392 | 4 | 2 | +0.0752 | False |

First fixed seed42 consumed-benchmark development comparison. Same-parameter three controls estimate product-vs-add conditional on reference, and reference-vs-absolute conditional on product; not a full2x2 interaction or novelty proof. Common dense-readout/pooling changes remain confounded against old semantic. Original camera/gallery rules; no NN/new ranking/test centering/tuning. Fixed-model identity bootstrap is descriptive, not training-seed uncertainty.
