# Visual-update/readout full50 tracker

Recorded 2026-10-01T17:14:17.3154333+08:00; sequential snapshot 2026-10-01T17:10:12.484135+08:00. Ten complete endpoints, two RGBNT100 global-only controls continue.

| Dataset | Condition | Parent | Child | GPU | Complete epochs |
|---|---|---|---|---:|---:|
| RGBNT201 | low_lr_roles | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT100 | low_lr_roles | COMPLETE | COMPLETE | 1 | 50 |
| MSVR310 | low_lr_roles | COMPLETE | COMPLETE | 2 | 50 |
| RGBNT201 | low_lr_global_only | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | low_lr_global_only | RUNNING | RUNNING | 2 | 45 |
| MSVR310 | low_lr_global_only | COMPLETE | COMPLETE | 0 | 50 |
| RGBNT201 | frozen_roles | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | frozen_roles | COMPLETE | COMPLETE | 0 | 50 |
| MSVR310 | frozen_roles | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT201 | frozen_global_only | COMPLETE | COMPLETE | 3 | 50 |
| RGBNT100 | frozen_global_only | RUNNING | RUNNING | 1 | 39 |
| MSVR310 | frozen_global_only | COMPLETE | COMPLETE | 3 | 50 |

MSVR310 full matched2x2; all rows full50, one official-mAP-best checkpoint.

| MSVR310 condition | best epoch | mAP | R1 |
|---|---:|---:|---:|
| frozen_global_only | 10 | 51.9480 | 67.3435 |
| frozen_roles | 10 | 52.3728 | 67.3435 |
| low_lr_global_only | 10 | 52.9354 | 67.8511 |
| low_lr_roles | 10 | 53.0027 | 68.0203 |

| MSVR310 comparison | delta mAP | delta R1 |
|---|---:|---:|
| visual_global | 0.9874 | 0.5076 |
| visual_roles | 0.6299 | 0.6768 |
| role_frozen | 0.4248 | 0.0000 |
| role_low_lr | 0.0673 | 0.1692 |

RGBNT100 roles matched pair; global-only controls unfinished.

| RGBNT100 roles condition | best epoch | mAP | R1 |
|---|---:|---:|
| frozen_roles | 1 | 85.1001 | 95.1020 |
| low_lr_roles | 1 | 85.3580 | 94.9854 |

Visual-update roles delta mAP/R1: 0.2579425949691654/-0.11661648750305176. Original all-pair R1>=0 requirement is not met by this pair; no rule retuning.
Full12 report/audit still pending. Source222/commoninit unchanged. No new neural or report execution; goalACTIVE_UNMET.
