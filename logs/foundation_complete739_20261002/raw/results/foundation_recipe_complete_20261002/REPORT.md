# F1 six-endpoint foundation-package report

| Dataset | Recipe | Best epoch | mAP | R1 | R5 | R10 | Steps |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | author | 27 | 73.4728 | 77.1531 | 85.8852 | 89.9522 | 2649 |
| RGBNT100 | author | 9 | 84.0319 | 96.2099 | - | - | 3129 |
| MSVR310 | author | 38 | 50.8388 | 68.6971 | - | - | 706 |
| RGBNT201 | current | 26 | 62.5806 | 62.4402 | 75.8373 | 83.0144 | 2649 |
| RGBNT100 | current | 12 | 77.5843 | 94.1691 | - | - | 6559 |
| MSVR310 | current | 24 | 50.3567 | 67.5127 | - | - | 1000 |

Foundation package comparison, not a new model contribution or isolated hyperparameter causality.
Full official galleries and original camera/time filtering; all metrics from one mAP-best checkpoint per endpoint.
Seed42 only, consumed official benchmark selection; identity bootstrap is fixed-model diagnosis, not training-seed significance.
Author RGBNT100 schedule extended from30 to50; common workers4/AMPscale256/evaluation path differ from the original entry.
Reported training interval includes epoch evaluation but excludes construction/M0/final evaluation; different B/K means unequal update counts.
Phase-progress threshold is not a SOTA or significance gate; overall objective remains ACTIVE_UNMET.
