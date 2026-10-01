# Complete frozen visual initialization comparison

Seed42; full50; one official fused-mAP-best per row.

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 | Endpoint hours |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | reid_visual | 2 | 72.5247 | 73.9234 | 82.7751 | 87.9187 | 0.573 |
| RGBNT201 | public_visual | 34 | 70.1618 | 72.3684 | 82.4163 | 86.4833 | 0.486 |
| RGBNT100 | reid_visual | 1 | 85.1051 | 95.0437 | 95.6851 | 95.9767 | 2.175 |
| RGBNT100 | public_visual | 16 | 77.9913 | 93.4111 | 94.1691 | 94.8688 | 1.775 |
| MSVR310 | reid_visual | 10 | 52.4159 | 67.8511 | 83.2487 | 89.0017 | 0.355 |
| MSVR310 | public_visual | 24 | 48.0166 | 66.1591 | 81.2183 | 86.9712 | 0.412 |

| Dataset | ReID visual to public visual | Delta mAP | Delta R1 | Repairs | New errors | Identity macro delta AP | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| RGBNT201 | reid_visual to public visual | -2.3629 | -1.5550 | 64 | 77 | -2.0830 | FAIL |
| RGBNT100 | reid_visual to public visual | -7.1137 | -1.6327 | 63 | 91 | -6.7850 | FAIL |
| MSVR310 | reid_visual to public visual | -4.3993 | -1.6920 | 42 | 52 | -4.3028 | FAIL |

Registered advancement gate: **FAIL**. The full three-dataset baseline/SOTA goal is not established by this gate.

- All six fresh endpoints and all three registered candidate-control comparisons are required.
- joint_local is the total role correction; no pure-local or independent-control claim.
- Within-checkpoint shared_global also received joint training; not a separately trained global-only control.
- Parameter equality is not equal effective capacity or computation.
- Scalar task support is not optimizer update share or unique cause.
- One seed with official-best development selection cannot establish stability, novelty or SOTA.
- No model forward, optimizer, new inference rule or test-based retuning occurs in this CPU analysis.
