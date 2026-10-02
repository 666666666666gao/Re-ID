# F2 training feature-scale report

| Dataset | Variant | Epoch | mAP | R1 | Steps |
|---|---|---:|---:|---:|---:|
| RGBNT201 | normalized | 26 | 62.5806 | 62.4402 | 2649 |
| RGBNT100 | normalized | 12 | 77.5843 | 94.1691 | 6559 |
| MSVR310 | normalized | 24 | 50.3567 | 67.5127 | 1000 |
| RGBNT201 | raw | 26 | 59.6879 | 60.8852 | 2649 |
| RGBNT100 | raw | 47 | 75.1929 | 94.4023 | 6559 |
| MSVR310 | raw | 23 | 51.5776 | 70.5584 | 1000 |

One fixed current-package control; affects BN/CE and Triplet together; not whole F1 gap causality, new module or SOTA. Official benchmark consumed.
