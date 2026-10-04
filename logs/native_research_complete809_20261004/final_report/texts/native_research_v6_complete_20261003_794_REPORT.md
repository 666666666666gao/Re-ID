# Independent native evidence report

| Dataset | Variant | Epoch | mAP | R1 | Steps |
|---|---|---:|---:|---:|---:|
| RGBNT201 | global_only | 8 | 74.2967 | 78.9474 | 2649 |
| RGBNT201 | semantic | 7 | 71.8981 | 74.4019 | 2649 |
| RGBNT201 | native | 20 | 72.1273 | 75.1196 | 2649 |
| MSVR310 | global_only | 38 | 50.5421 | 68.0203 | 706 |
| MSVR310 | semantic | 49 | 50.9636 | 69.2047 | 706 |
| MSVR310 | native | 38 | 50.6755 | 68.6971 | 706 |
| RGBNT100 | global_only | 7 | 84.5338 | 96.6181 | 3129 |
| RGBNT100 | semantic | 5 | 83.4395 | 96.0933 | 3129 |
| RGBNT100 | native | 5 | 82.5610 | 95.5102 | 3129 |

One seed42, consumed official mAP selection, matched author package. Native adds159296 parameters; no capacity-only causal claim, three-new-module success, seed robustness or SOTA. Identity bootstrap is not training-seed uncertainty.
