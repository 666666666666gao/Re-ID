# 同模态几何辅助目标：三端结果

| 数据集 | best轮 | mAP | R1 | own-global mAP | 末轮mAP | 正式更新 |
|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | 8 | 74.3864 | 79.1866 | 74.2967 | 71.6687 | 2649 |
| MSVR310 | 38 | 50.5468 | 68.0203 | 50.5421 | 50.4720 | 706 |
| RGBNT100 | 26 | 83.5272 | 95.8017 | 83.9655 | 82.5514 | 3129 |

主要配对过线：0/3；正式端完成：3/3。

Fixed same-image same-modality geometry auxiliary package; zero new model parameters, original raw tasks/global ownership/1536 inference. Full50 on consumed official development benchmarks. Correspondence/entropy are proxies, not identity mechanism proof. Missing endpoints have no substituted score; no full-pipeline seeds, cross-spectral true parts, formula novelty or SOTA claim.
