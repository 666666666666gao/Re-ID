# F3 metric feature-scale report

| Dataset | Variant | Epoch | mAP | R1 | Steps |
|---|---|---:|---:|---:|---:|
| RGBNT201 | normalized | 26 | 62.5806 | 62.4402 | 2649 |
| RGBNT100 | normalized | 12 | 77.5843 | 94.1691 | 6559 |
| MSVR310 | normalized | 24 | 50.3567 | 67.5127 | 1000 |
| RGBNT201 | metric_raw | 38 | 59.7343 | 60.2871 | 2649 |
| RGBNT100 | metric_raw | 48 | 75.4520 | 94.7522 | 6559 |
| MSVR310 | metric_raw | 23 | 52.7596 | 71.4044 | 1000 |

One fixed fresh current-package control; changes Triplet input only under normalized BN/CE. Not full F1 decomposition, all scale interactions, a new module or SOTA. Official benchmark consumed.
