# Visual-update/readout full50 tracker

Recorded 2026-10-01T16:43:53.7499958+08:00; sequential snapshot 2026-10-01T16:35:12.056198+08:00. Source222/registration/commoninitialization unchanged.

| Dataset | Condition | Parent | Child | GPU | Complete epochs |
|---|---|---|---|---:|---:|
| RGBNT201 | low_lr_roles | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT100 | low_lr_roles | COMPLETE | COMPLETE | 1 | 50 |
| MSVR310 | low_lr_roles | COMPLETE | COMPLETE | 2 | 50 |
| RGBNT201 | low_lr_global_only | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | low_lr_global_only | RUNNING | RUNNING | 2 | 34 |
| MSVR310 | low_lr_global_only | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT201 | frozen_roles | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | frozen_roles | RUNNING | RUNNING | 0 | 39 |
| MSVR310 | frozen_roles | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT201 | frozen_global_only | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | frozen_global_only | RUNNING | RUNNING | 1 | 19 |
| MSVR310 | frozen_global_only | RUNNING | RUNNING | 3 | 21 |

First complete2012x2; every row full50 and one mAP-best checkpoint.

| RGBNT201 condition | best epoch | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| frozen_global_only | 2 | 72.4944 | 74.0431 | 82.4163 | 87.9187 |
| frozen_roles | 2 | 72.5087 | 73.9234 | 82.7751 | 87.9187 |
| low_lr_global_only | 1 | 73.7442 | 74.7608 | 84.8086 | 88.7560 |
| low_lr_roles | 1 | 73.8535 | 75.1196 | 84.3301 | 88.7560 |

Single-dataset comparisons; fullC1/C2/report/audit pending.

| RGBNT201 comparison | delta mAP | delta R1 | delta R5 | delta R10 |
|---|---:|---:|---:|---:|
| visual_global | 1.2498 | 0.7177 | 2.3923 | 0.8373 |
| visual_roles | 1.3448 | 1.1962 | 1.5550 | 0.8373 |
| role_frozen | 0.0143 | -0.1196 | 0.3589 | 0.0000 |
| role_low_lr | 0.1093 | 0.3588 | -0.4785 | 0.0000 |

Global-only retains M1 averaged adapters and removes role operators/readout. Fine-tuning control is not novelty. GoalACTIVE_UNMET; no new LR/seed/configuration from partial scores.
