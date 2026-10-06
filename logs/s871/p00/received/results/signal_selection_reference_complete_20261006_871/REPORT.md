# Signal选择来源参考：完整九端

| Dataset | Arm | Best epoch | mAP | R1 | Steps |
|---|---|---:|---:|---:|---:|
| RGBNT201 | global_only | 27 | 73.4728 | 77.1531 | 2649 |
| RGBNT201 | masked | 29 | 72.7854 | 75.4785 | 2649 |
| RGBNT201 | all_patch | 7 | 72.5484 | 76.6746 | 2649 |
| MSVR310 | global_only | 38 | 50.8388 | 68.6971 | 706 |
| MSVR310 | masked | 47 | 53.2453 | 70.3892 | 706 |
| MSVR310 | all_patch | 48 | 53.2706 | 70.7276 | 706 |
| RGBNT100 | global_only | 9 | 84.0319 | 96.2099 | 3129 |
| RGBNT100 | masked | 6 | 85.3526 | 96.0350 | 3129 |
| RGBNT100 | all_patch | 5 | 86.0717 | 95.4519 | 3129 |

Source-neighbor reference, not new TriFusion contribution. Full author-source RAW global/var joint objectives and3072 SIM retrieval; fixedtopk80/64/112; noAlignM, roles, shared adapters or native CNN. All primarydataset pairs published; stability/SOTA unproven.
