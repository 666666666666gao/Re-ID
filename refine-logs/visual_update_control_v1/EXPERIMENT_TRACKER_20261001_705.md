# Visual-update/readout full50 tracker

Recorded 2026-10-01T16:06:51.4411393+08:00; sequential snapshot 2026-10-01T16:05:12.018799+08:00. Fixed source222, budget, registration and common initialization unchanged.

| Dataset | Condition | Parent | Child | GPU | Complete epochs |
|---|---|---|---|---:|---:|
| RGBNT201 | low_lr_roles | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT100 | low_lr_roles | COMPLETE | COMPLETE | 1 | 50 |
| MSVR310 | low_lr_roles | COMPLETE | COMPLETE | 2 | 50 |
| RGBNT201 | low_lr_global_only | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | low_lr_global_only | RUNNING | RUNNING | 2 | 26 |
| MSVR310 | low_lr_global_only | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT201 | frozen_roles | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | frozen_roles | RUNNING | RUNNING | 0 | 25 |
| MSVR310 | frozen_roles | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT201 | frozen_global_only | RUNNING | RUNNING | 3 | 25 |
| RGBNT100 | frozen_global_only | RUNNING | RUNNING | 1 | 1 |
| MSVR310 | frozen_global_only | PENDING | - | - | - |

Parent complete 7/12; child verified 7/12. One mAP-best checkpoint per complete50-epoch endpoint.

| Dataset | Condition | best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | low_lr_roles | 1 | 73.8535 | 75.1196 | 84.3301 | 88.7560 |
| RGBNT100 | low_lr_roles | 1 | 85.3580 | 94.9854 | - | - |
| MSVR310 | low_lr_roles | 10 | 53.0027 | 68.0203 | - | - |
| RGBNT201 | low_lr_global_only | 1 | 73.7442 | 74.7608 | 84.8086 | 88.7560 |
| MSVR310 | low_lr_global_only | 10 | 52.9354 | 67.8511 | - | - |
| RGBNT201 | frozen_roles | 2 | 72.5087 | 73.9234 | 82.7751 | 87.9187 |
| MSVR310 | frozen_roles | 10 | 52.3728 | 67.3435 | - | - |

C1 partial pair comparisons; full C1/C2/report/audit pending.

| Dataset/readout | low_lr minus matched frozen mAP | delta R1 | status |
|---|---:|---:|---|
| RGBNT201/roles | 1.3448 | 1.1962 | Complete pair; final full12 audit pending |
| RGBNT201/global_only | - | - | Pending both complete endpoints |
| RGBNT100/roles | - | - | Pending both complete endpoints |
| RGBNT100/global_only | - | - | Pending both complete endpoints |
| MSVR310/roles | 0.6299 | 0.6768 | Complete pair; final full12 audit pending |
| MSVR310/global_only | - | - | Pending both complete endpoints |

Existing observer240sec owns one CPU complete report; no restart/new LR/seed/interim-score rescue. GoalACTIVE_UNMET.
