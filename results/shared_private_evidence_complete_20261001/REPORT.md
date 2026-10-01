# Complete shared/private evidence-flow controls

| Dataset | Flow | Best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | coupled_roles | 1 | 74.0506 | 75.5981 | 85.0478 | 88.6364 |
| RGBNT201 | separated_roles | 1 | 73.8242 | 75.2392 | 84.4498 | 88.5167 |
| RGBNT201 | global_only | 1 | 73.7442 | 74.7608 | 84.8086 | 88.7560 |
| RGBNT100 | coupled_roles | 1 | 85.5581 | 95.2770 | 95.9767 | 96.5598 |
| RGBNT100 | separated_roles | 1 | 85.4176 | 95.3353 | 95.8601 | 96.3848 |
| RGBNT100 | global_only | 1 | 85.3799 | 95.2187 | 96.0350 | 96.5015 |
| MSVR310 | coupled_roles | 10 | 53.2434 | 67.3435 | 83.0795 | 89.0017 |
| MSVR310 | separated_roles | 10 | 53.0721 | 67.6819 | 83.0795 | 89.0017 |
| MSVR310 | global_only | 10 | 52.9354 | 67.8511 | 82.9103 | 88.6633 |

| Dataset | Comparison | Delta mAP | Delta R1 | Repairs | New errors | Identity macro AP delta | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| RGBNT201 | coupled_roles to separated_roles | -0.2264 | -0.3589 | 3 | 6 | -0.2340 | FAIL |
| RGBNT100 | coupled_roles to separated_roles | -0.1405 | +0.0583 | 6 | 5 | -0.1835 | FAIL |
| MSVR310 | coupled_roles to separated_roles | -0.1714 | +0.3384 | 12 | 10 | -0.1892 | FAIL |
| RGBNT201 | global_only to separated_roles | +0.0800 | +0.4785 | 4 | 0 | +0.0794 | FAIL |
| RGBNT100 | global_only to separated_roles | +0.0377 | +0.1166 | 3 | 1 | +0.0521 | PASS |
| MSVR310 | global_only to separated_roles | +0.1367 | -0.1692 | 8 | 9 | +0.2139 | FAIL |
| RGBNT201 | global_only to coupled_roles | +0.3064 | +0.8373 | 9 | 2 | +0.3134 | FAIL |
| RGBNT100 | global_only to coupled_roles | +0.1782 | +0.0583 | 8 | 7 | +0.2356 | PASS |
| MSVR310 | global_only to coupled_roles | +0.3080 | -0.5076 | 10 | 13 | +0.4031 | FAIL |

Coupled-versus-global comparisons are diagnostic and do not determine S1/S2.

Registered S1 structural gate: **FAIL**.
Registered S2 independent-role gate: **FAIL**.
Registered joint gate: **FAIL**.

- All nine fresh full50 endpoints required; one official-mAP-best checkpoint supplies all metrics.
- All flows use FP32 low-LR visual storage and matched shared initialization; trained ReID/camera state retained.
- Coupled/separated roles have identical full initial model state and parameter count; only direct private writeback differs.
- Both role flows use the same reviewed private-MLP FP32 numerical correction; failed original M0 evidence retained.
- Global-only is independently trained, not a same-checkpoint slice; it has different capacity/cost and repeats the prior recipe.
- Separation removes direct writeback; role gradients can still change shared/visual parameters and global behavior.
- S1/S2 each require three positive mAP and nonnegative R1 deltas, with at least0.5pp mAP on RGBNT201/MSVR310.
- Old visual C1 and role C2 gates remain FAIL; this report cannot relabel those results.
- One seed and consumed official-best selection do not establish training stability, untouched-test significance or SOTA.
- Identity bootstrap resamples fixed-model identities, not training seeds; flips/AP are post-selection real-GT diagnoses.
- Memory is peak allocated usage after initialization, excluding initialization transients and reserved memory.
- This report loads fixed CPU arrays; it performs no neural forward, optimizer update or inference-rule change.

The full three-dataset baseline/SOTA goal is not established by this gate.
