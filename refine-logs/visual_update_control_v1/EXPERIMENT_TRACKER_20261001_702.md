# Visual-update/readout full50 tracker

Recorded 2026-10-01T14:46:17.3483184+08:00; one dated sequential snapshot 2026-10-01T14:44:41.705454+08:00. Immutable registration/plan/initialization remain unchanged.

| Dataset | Condition | Parent status | Child status | GPU | Complete epochs |
|---|---|---|---|---:|---:|
| RGBNT201 | low_lr_roles | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT100 | low_lr_roles | RUNNING | RUNNING | 1 | 18 |
| MSVR310 | low_lr_roles | COMPLETE | COMPLETE | 2 | 50 |
| RGBNT201 | low_lr_global_only | RUNNING | COMPLETE | 3 | 50 |
| RGBNT100 | low_lr_global_only | RUNNING | RUNNING | 2 | 3 |
| MSVR310 | low_lr_global_only | RUNNING | RUNNING | 0 | 5 |
| RGBNT201 | frozen_roles | PENDING | - | - | - |
| RGBNT100 | frozen_roles | PENDING | - | - | - |
| MSVR310 | frozen_roles | PENDING | - | - | - |
| RGBNT201 | frozen_global_only | PENDING | - | - | - |
| RGBNT100 | frozen_global_only | PENDING | - | - | - |
| MSVR310 | frozen_global_only | PENDING | - | - | - |

Parent complete 2/12; child verified 3/12. Official selected metrics only from parent-confirmed complete endpoints.

| Dataset | Condition | best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | low_lr_roles | 1 | 73.8535 | 75.1196 | 84.3301 | 88.7560 |
| MSVR310 | low_lr_roles | 10 | 53.0027 | 68.0203 | - | - |

Full C1 and C2 remain pending. No replacement of frozen controls, interim-score rescue, new LR, seed or observer. Complete CPU report invocations0; existing waiter owns its one execution after full12.

| Dataset | low_lr roles minus independent global | delta mAP | delta Rank-1 | Full C2 |
|---|---|---:|---:|---|
| RGBNT201 | Await both complete endpoints | - | - | PENDING_ALL_THREE |
| RGBNT100 | Await both complete endpoints | - | - | PENDING_ALL_THREE |
| MSVR310 | Await both complete endpoints | - | - | PENDING_ALL_THREE |

Goal remains ACTIVE/UNMET; ordinary fine-tuning control is not novelty.
