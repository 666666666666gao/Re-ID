# Visual-update/readout full50 tracker

Recorded 2026-10-01T18:09:29.4413101+08:00; all12 full50 endpoints verified; original CPU report once exit0 at 2026-10-01T17:37:10.786035+08:00.

| Dataset | Condition | best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | low_lr_roles | 1 | 73.8535 | 75.1196 | 84.3301 | 88.7560 |
| RGBNT100 | low_lr_roles | 1 | 85.3580 | 94.9854 | 95.7434 | 96.3265 |
| MSVR310 | low_lr_roles | 10 | 53.0027 | 68.0203 | 82.5719 | 88.8325 |
| RGBNT201 | low_lr_global_only | 1 | 73.7442 | 74.7608 | 84.8086 | 88.7560 |
| RGBNT100 | low_lr_global_only | 1 | 85.3799 | 95.2187 | 96.0350 | 96.5015 |
| MSVR310 | low_lr_global_only | 10 | 52.9354 | 67.8511 | 82.9103 | 88.6633 |
| RGBNT201 | frozen_roles | 2 | 72.5087 | 73.9234 | 82.7751 | 87.9187 |
| RGBNT100 | frozen_roles | 1 | 85.1001 | 95.1020 | 95.6851 | 96.0350 |
| MSVR310 | frozen_roles | 10 | 52.3728 | 67.3435 | 83.7563 | 88.8325 |
| RGBNT201 | frozen_global_only | 2 | 72.4944 | 74.0431 | 82.4163 | 87.9187 |
| RGBNT100 | frozen_global_only | 1 | 85.1884 | 95.1020 | 95.6851 | 96.0933 |
| MSVR310 | frozen_global_only | 10 | 51.9480 | 67.3435 | 83.5871 | 89.1709 |

| Dataset | Visual update: global ΔmAP/ΔR1 | Visual update: roles ΔmAP/ΔR1 | low_lr roles − independent global ΔmAP/ΔR1 |
|---|---:|---:|---:|
| RGBNT201 | +1.2498/+0.7177 | +1.3448/+1.1962 | +0.1093/+0.3588 |
| RGBNT100 | +0.1915/+0.1166 | +0.2579/-0.1166 | -0.0219/-0.2332 |
| MSVR310 | +0.9874/+0.5076 | +0.6299/+0.6768 | +0.0673/+0.1692 |

Registered visual gate: FAIL; registered role gate: FAIL; joint gate: FAIL.
Actual fresh gpt-6-astra max audit: WARN; same-family/provisional. See results/visual_update_control_complete_20261001/EXPERIMENT_AUDIT.md.

No source222, metric, gate, learning-rate, seed or checkpoint selection change. All complete scientific failures preserved. GoalACTIVE_UNMET.
