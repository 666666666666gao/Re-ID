# Complete visual-update and independent-role controls

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | low_lr_roles | 1 | 73.8535 | 75.1196 | 84.3301 | 88.7560 |
| RGBNT201 | low_lr_global_only | 1 | 73.7442 | 74.7608 | 84.8086 | 88.7560 |
| RGBNT201 | frozen_roles | 2 | 72.5087 | 73.9234 | 82.7751 | 87.9187 |
| RGBNT201 | frozen_global_only | 2 | 72.4944 | 74.0431 | 82.4163 | 87.9187 |
| RGBNT100 | low_lr_roles | 1 | 85.3580 | 94.9854 | 95.7434 | 96.3265 |
| RGBNT100 | low_lr_global_only | 1 | 85.3799 | 95.2187 | 96.0350 | 96.5015 |
| RGBNT100 | frozen_roles | 1 | 85.1001 | 95.1020 | 95.6851 | 96.0350 |
| RGBNT100 | frozen_global_only | 1 | 85.1884 | 95.1020 | 95.6851 | 96.0933 |
| MSVR310 | low_lr_roles | 10 | 53.0027 | 68.0203 | 82.5719 | 88.8325 |
| MSVR310 | low_lr_global_only | 10 | 52.9354 | 67.8511 | 82.9103 | 88.6633 |
| MSVR310 | frozen_roles | 10 | 52.3728 | 67.3435 | 83.7563 | 88.8325 |
| MSVR310 | frozen_global_only | 10 | 51.9480 | 67.3435 | 83.5871 | 89.1709 |

| Dataset | Comparison | Delta mAP | Delta R1 | Repairs | New errors | Identity macro AP delta | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| RGBNT201 | frozen_global_only to low_lr_global_only | +1.2498 | +0.7177 | 16 | 10 | +1.2841 | PASS |
| RGBNT201 | frozen_roles to low_lr_roles | +1.3448 | +1.1962 | 19 | 9 | +1.3801 | PASS |
| RGBNT100 | frozen_global_only to low_lr_global_only | +0.1915 | +0.1166 | 9 | 7 | +0.2361 | PASS |
| RGBNT100 | frozen_roles to low_lr_roles | +0.2579 | -0.1166 | 6 | 8 | +0.3133 | FAIL |
| MSVR310 | frozen_global_only to low_lr_global_only | +0.9874 | +0.5076 | 23 | 20 | +0.7049 | PASS |
| MSVR310 | frozen_roles to low_lr_roles | +0.6299 | +0.6768 | 26 | 22 | +0.4208 | PASS |
| RGBNT201 | frozen_global_only to frozen_roles | +0.0143 | -0.1196 | 0 | 1 | +0.0103 | FAIL |
| RGBNT201 | low_lr_global_only to low_lr_roles | +0.1093 | +0.3589 | 3 | 0 | +0.1063 | FAIL |
| RGBNT100 | frozen_global_only to frozen_roles | -0.0883 | +0.0000 | 1 | 1 | -0.0861 | FAIL |
| RGBNT100 | low_lr_global_only to low_lr_roles | -0.0219 | -0.2332 | 1 | 5 | -0.0089 | FAIL |
| MSVR310 | frozen_global_only to frozen_roles | +0.4248 | +0.0000 | 6 | 6 | +0.3883 | FAIL |
| MSVR310 | low_lr_global_only to low_lr_roles | +0.0673 | +0.1692 | 8 | 7 | +0.1042 | FAIL |

Frozen-role comparisons are diagnostic; the registered role gate uses only low_lr.

Registered visual-update gate: **FAIL**.
Registered role gate: **FAIL**.
Registered joint gate: **FAIL**.

- All12 fresh full50 endpoints required; one official-mAP-best checkpoint supplies all metrics.
- All four conditions use FP32 visual storage and matched common initialization; trained camera/nonvisual state retained.
- Global-only is independently trained, not a same-checkpoint slice; roles have different parameters/cost.
- Only global-only visual improvement is partial C1 evidence; the overall visual gate still fails unless both readouts qualify.
- Ordinary fine-tuning is a control, not novelty. No gate, learning rate, seed or epoch rule is changed by this report.
- One seed and official-best selection do not establish stability, untouched-test significance or SOTA.
- Identity bootstrap resamples fixed-model identities, not training seeds; all flips/AP are post-selection real-GT diagnoses.
- Memory is peak allocated usage after initialization, excluding initialization transients and reserved memory.
- This report loads fixed CPU arrays; it performs no neural forward, optimizer or inference-rule change.

The full three-dataset baseline/SOTA goal is not established by this gate.
