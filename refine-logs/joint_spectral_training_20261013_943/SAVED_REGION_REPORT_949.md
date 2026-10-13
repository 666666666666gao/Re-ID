# 固定区域证据：均值与对称MaxSim比较

| 条件 | 来源 | mean mAP/R1 | MaxSim mAP/R1 | ΔmAP | 修复/新增错误 |
|---|---|---:|---:|---:|---:|
| joint_delta | processed_regions | 5.4876/2.7512 | 5.5352/3.3493 | +0.0476 | 21/16 |
| joint_delta | reference_regions | 7.9806/4.0670 | 9.9460/7.5359 | +1.9655 | 41/12 |
| add_delta | processed_regions | 11.0069/9.9282 | 11.5770/9.5694 | +0.5701 | 31/34 |
| add_delta | reference_regions | 11.2160/7.0574 | 16.7888/16.2679 | +5.5729 | 107/30 |
| joint_absolute | processed_regions | 6.8544/5.8612 | 6.5800/5.3828 | -0.2744 | 11/15 |
| joint_absolute | reference_regions | 8.0454/5.9809 | 7.9856/7.0574 | -0.0598 | 31/22 |

Mean of all pairwise cosine versus symmetric mean of row/column maxima on identical normalized tokens, within each of9 role/modality groups. Local-only scores; no global fusion coefficient, training, teacher, model selection, reranking subset or test centering. New diagnostic comparison rules use more stored vectors and pairwise computation; they are not original1536-vector formal results or complete ColBERT reproduction. Neither sign proves all late comparisons useful/useless or identifies compression/geometry as a causal failure.
