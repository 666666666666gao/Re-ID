# Fixed independent-global difficulty: saved-query CPU diagnosis

Post-development dataset-GT diagnosis only, all original selected-best queries retained. Same identity may occur in multiple strata, so identity counts are not additive. Query-level within-bin AP means differ from contributions to overall mAP. This is not the actual train-time global margin/weight, a deployable GT-based selector, multiple-seed uncertainty, current same-model decomposition, stronger-teacher evidence, or a unique failure cause. No new NN, training or model selection.

| Dataset | Global first rank | Queries | U−global within-bin AP | A−global within-bin AP | A−U within-bin AP | A−U all-query mAP contribution | A−U repairs/new |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | 1 | 660 | +0.4863 | +0.3419 | -0.1444 | -0.1140 | 3/4 |
| RGBNT201 | 2-5 | 78 | +0.6804 | +0.5087 | -0.1717 | -0.0160 | 3/1 |
| RGBNT201 | 6-10 | 30 | +0.6854 | +0.4968 | -0.1886 | -0.0068 | 0/0 |
| RGBNT201 | >10 | 68 | +0.2380 | -0.0244 | -0.2623 | -0.0213 | 0/0 |
| RGBNT201 | global first correct | 660 | +0.4863 | +0.3419 | -0.1444 | -0.1140 | 3/4 |
| RGBNT201 | global first wrong | 176 | +0.5103 | +0.3007 | -0.2096 | -0.0441 | 3/1 |
| RGBNT100 | 1 | 1657 | -1.5476 | -1.5189 | +0.0287 | +0.0277 | 7/10 |
| RGBNT100 | 2-5 | 12 | +8.0137 | +7.4437 | -0.5701 | -0.0040 | 0/1 |
| RGBNT100 | 6-10 | 11 | +8.0727 | +9.8878 | +1.8151 | +0.0116 | 0/0 |
| RGBNT100 | >10 | 35 | +6.8297 | +8.0447 | +1.2150 | +0.0248 | 8/0 |
| RGBNT100 | global first correct | 1657 | -1.5476 | -1.5189 | +0.0287 | +0.0277 | 7/10 |
| RGBNT100 | global first wrong | 58 | +7.3104 | +8.2699 | +0.9595 | +0.0324 | 8/1 |
| MSVR310 | 1 | 402 | -0.0127 | -0.0071 | +0.0057 | +0.0039 | 0/0 |
| MSVR310 | 2-5 | 74 | +0.0685 | +0.0696 | +0.0011 | +0.0001 | 0/0 |
| MSVR310 | 6-10 | 29 | -0.0255 | -0.0255 | -0.0000 | -0.0000 | 0/0 |
| MSVR310 | >10 | 86 | +0.0068 | +0.0067 | -0.0001 | -0.0000 | 0/0 |
| MSVR310 | global first correct | 402 | -0.0127 | -0.0071 | +0.0057 | +0.0039 | 0/0 |
| MSVR310 | global first wrong | 189 | +0.0260 | +0.0264 | +0.0004 | +0.0001 | 0/0 |
