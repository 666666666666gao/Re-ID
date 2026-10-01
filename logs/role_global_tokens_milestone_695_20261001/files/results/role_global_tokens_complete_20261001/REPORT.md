# Complete global-token interaction panel

Seed42; full50; one official fused-mAP-best per row.

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 | Endpoint hours |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | static | 2 | 72.5308 | 74.0431 | 82.7751 | 87.9187 | 0.577 |
| RGBNT201 | token | 2 | 72.5621 | 73.9234 | 82.4163 | 87.7990 | 0.487 |
| RGBNT201 | direct | 2 | 72.5220 | 73.9234 | 82.6555 | 87.9187 | 0.584 |
| RGBNT100 | static | 1 | 85.1035 | 95.0437 | 95.6851 | 96.0350 | 2.124 |
| RGBNT100 | token | 1 | 85.1178 | 94.9854 | 95.6851 | 95.9767 | 1.781 |
| RGBNT100 | direct | 1 | 85.1020 | 95.1020 | 95.6851 | 96.0350 | 1.613 |
| MSVR310 | static | 10 | 52.4849 | 68.0203 | 83.4179 | 89.3401 | 0.358 |
| MSVR310 | token | 10 | 52.4877 | 68.1895 | 83.4179 | 89.0017 | 0.415 |
| MSVR310 | direct | 10 | 52.4341 | 68.1895 | 83.5871 | 89.1709 | 0.403 |

| Dataset | Control to token | Delta mAP | Delta R1 | Repairs | New errors | Identity macro delta AP | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| RGBNT201 | static to token | +0.0313 | -0.1196 | 1 | 2 | +0.0320 | FAIL |
| RGBNT201 | direct to token | +0.0401 | +0.0000 | 2 | 2 | +0.0426 | FAIL |
| RGBNT100 | static to token | +0.0143 | -0.0583 | 0 | 1 | +0.0111 | FAIL |
| RGBNT100 | direct to token | +0.0158 | -0.1166 | 0 | 2 | +0.0131 | FAIL |
| MSVR310 | static to token | +0.0028 | +0.1692 | 8 | 7 | -0.0363 | FAIL |
| MSVR310 | direct to token | +0.0535 | +0.0000 | 7 | 7 | -0.1269 | FAIL |

Registered advancement gate: **FAIL**. The full three-dataset baseline/SOTA goal is not established by this gate.

- All nine fresh endpoints and all six registered candidate-control comparisons are required.
- joint_local is the total correction and contains global information; no pure-local or independent-control claim.
- Within-checkpoint shared_global also received joint training; not a separately trained global-only control.
- Parameter equality is not equal effective capacity or computation.
- Scalar task support is not optimizer update share or unique cause.
- One seed with official-best development selection cannot establish stability, novelty or SOTA.
- No model forward, optimizer, new inference rule or test-based retuning occurs in this CPU analysis.
