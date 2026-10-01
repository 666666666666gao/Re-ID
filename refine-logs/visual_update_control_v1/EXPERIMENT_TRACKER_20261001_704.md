# Visual-update/readout full50 tracker

Recorded 2026-10-01T15:29:47.1131486+08:00; sequential snapshot 2026-10-01T15:25:20.630179+08:00. Registered sources/plan/initialization unchanged.

| Dataset | Condition | Parent status | Child status | GPU | Complete epochs |
|---|---|---|---|---:|---:|
| RGBNT201 | low_lr_roles | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT100 | low_lr_roles | RUNNING | RUNNING | 1 | 36 |
| MSVR310 | low_lr_roles | COMPLETE | COMPLETE | 2 | 50 |
| RGBNT201 | low_lr_global_only | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | low_lr_global_only | RUNNING | RUNNING | 2 | 14 |
| MSVR310 | low_lr_global_only | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT201 | frozen_roles | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | frozen_roles | RUNNING | RUNNING | 0 | 7 |
| MSVR310 | frozen_roles | RUNNING | RUNNING | 3 | 6 |
| RGBNT201 | frozen_global_only | PENDING | - | - | - |
| RGBNT100 | frozen_global_only | PENDING | - | - | - |
| MSVR310 | frozen_global_only | PENDING | - | - | - |

Parent/child verified complete5/12. All complete entries have actual three stage exit0 and full50.

| Dataset | Condition | best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | low_lr_roles | 1 | 73.8535 | 75.1196 | 84.3301 | 88.7560 |
| MSVR310 | low_lr_roles | 10 | 53.0027 | 68.0203 | - | - |
| RGBNT201 | low_lr_global_only | 1 | 73.7442 | 74.7608 | 84.8086 | 88.7560 |
| MSVR310 | low_lr_global_only | 10 | 52.9354 | 67.8511 | - | - |
| RGBNT201 | frozen_roles | 2 | 72.5087 | 73.9234 | 82.7751 | 87.9187 |

First complete visual-update comparison; full C1/C2 and audit pending.

| low_lr_roles minus frozen_roles | delta mAP | delta R1 | delta R5 | delta R10 |
|---|---:|---:|---:|---:|
| RGBNT201 complete pair | 1.3448 | 1.1962 | 1.5550 | 0.8373 |

Existing240sec observer owns the one full12 CPU report. No new LR/seed/configuration or interim-score rescue. GoalACTIVE_UNMET.
