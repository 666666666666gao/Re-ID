# Complete clean-public CLIP joint controls

| Dataset | Readout | Best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | global_only | 13 | 68.1117 | 69.7368 | 79.0670 | 84.5694 |
| RGBNT201 | roles | 13 | 69.8764 | 71.4115 | 79.1866 | 84.9282 |
| RGBNT100 | global_only | 3 | 78.4913 | 93.8776 | — | — |
| RGBNT100 | roles | 20 | 80.9231 | 94.2857 | — | — |
| MSVR310 | global_only | 16 | 51.7224 | 67.1743 | — | — |
| MSVR310 | roles | 16 | 52.0042 | 68.6971 | — | — |

| Dataset | Roles minus global mAP | R1 | Repairs | New errors | Identity macro AP delta | J1 |
|---|---:|---:|---:|---:|---:|---|
| RGBNT201 | +1.7647 | +1.6746 | 29 | 15 | +1.9876 | PASS |
| RGBNT100 | +2.4318 | +0.4082 | 58 | 51 | +2.5193 | PASS |
| MSVR310 | +0.2818 | +1.5228 | 16 | 7 | +0.1673 | FAIL |

Registered J1: **FAIL**.

- Clean public CLIP visual weights; fresh trainable camera, adapters and heads. No trained ReID state loaded.
- Global-only includes the same shared adapters; it is an independent trained control, not raw or zero-shot CLIP.
- Roles add capacity and may consume randomness differently. Common initial states were actually compared bitwise.
- All six ends completed50 epochs; one highest official fused-mAP checkpoint per end, tied epochs resolved later.
- Full real-GT query/gallery and original camera/time-block exclusions; gallery-only distractors retained; no reranking.
- Official benchmarks already participated in epoch and method development. Fixed-model identity bootstrap is not training-seed stability.
- Training/epoch-evaluation intervals exclude construction and upstream costs. Two RGBNT201 M0s precede the formal controller.
- Trainable counts are reported from initializer witnesses; equal output width/epochs do not imply equal capacity or computation.
- Warm ReID comparisons are historical and differ in camera training and initialization; no single-factor causal attribution.
- J1 is an intermediate role-contribution gate, not the full three-dataset baseline/SOTA goal. N1/N2/N3 are not tested here.
