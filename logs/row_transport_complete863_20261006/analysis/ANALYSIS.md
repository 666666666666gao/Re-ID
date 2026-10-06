# Row transport complete evidence — 2026-10-06

All six fresh 50-epoch endpoints and first strict reloads completed; 300 epochs / 12,968 updates. All 15 paired training batch orders match. The slot-versus-uniform primary gate passes 0/3, all comparisons 0/15. The fixed-best diagnosis is complete: six models, 18 deployment modes, 24 full-query comparisons.

| Dataset | Mode | Best epoch | mAP / R1 / R5 / R10 |
|---|---|---:|---|
| RGBNT201 | slot_mass | 8 | 74.338779 / 78.588516 / 88.397127 / 91.985649 |
| RGBNT201 | uniform_mass | 8 | 74.256865 / 78.827751 / 88.277513 / 91.746414 |
| MSVR310 | slot_mass | 38 | 50.546624 / 68.020302 / 80.541456 / 85.448390 |
| MSVR310 | uniform_mass | 38 | 50.547712 / 68.020302 / 80.541456 / 85.448390 |
| RGBNT100 | slot_mass | 26 | 83.418072 / 96.268219 / 96.734697 / 97.142857 |
| RGBNT100 | uniform_mass | 26 | 83.895222 / 96.384841 / 97.084546 / 97.609329 |

## Observations and limits

1. Switching allocation in the same best model changes mAP by at most 0.001059942 percentage points; all six R1/R5/R10 values stay unchanged. This does not explain the separately trained RGBNT100 gap of 0.477150 mAP.
2. Removing only the effective peer write changes mAP by at most 0.030482853 points. It retains private Mamba, CNN/Transformer, readout and existing shared context. This is not a retrained deletion of all cross-modal computation.
3. RGBNT201 conditional assignments are nearly uniform (entropy approximately ln(16)=2.772589), and receiving-slot mass variation is tiny. RGBNT100 has more varied masses and stronger writes, yet its fused representation remains worse than its own global: -0.547380/-0.070231 mAP for slot/uniform. Neither branch activity nor correction-only recognition proves useful complementarity.
4. Matching-matrix norms shrink sharply in RGBNT201/RGBNT100. The records do not isolate task gradients from weight decay across training, so no unique causal explanation is claimed.
5. RGBNT100 own globals are 83.965452 mAP, below the independent global-only 84.533784. Duty separation does not establish identical global trajectories; the cause of this difference is not identified here.

| Dataset | Mode | Mean within-object mass std | Conditional entropy | Scaled correction/global | Fused − own global mAP |
|---|---|---:|---:|---:|---:|
| RGBNT201 | slot_mass | 0.000150 | 2.772587 | 0.063735 | 0.042116 |
| RGBNT201 | uniform_mass | 0.000334 | 2.772582 | 0.057653 | -0.039799 |
| MSVR310 | slot_mass | 0.006049 | 2.558328 | 0.006098 | 0.004495 |
| MSVR310 | uniform_mass | 0.006150 | 2.555933 | 0.006071 | 0.005584 |
| RGBNT100 | slot_mass | 0.026106 | 2.657813 | 0.228580 | -0.547380 |
| RGBNT100 | uniform_mass | 0.013724 | 2.749737 | 0.178549 | -0.070231 |

## Execution provenance

The initial fixed-best process exited 1 at 09:31:57 after four models/12 modes. Its shared YACS configuration carried MSVR steps (20,40) into RGBNT100, whose YAML omits that key. A CPU replay shows fresh RGBNT100 uses (40,70) and exactly matches its initializer dump. Original formal training used fresh processes and is unaffected. Only the two missing RGBNT100 models were resumed, each in a fresh process with unchanged scientific source and original strict gates. Both exited 0; the sole CPU report exited 0 at 10:00:00. Original failure receipts and the old four-model progress file remain historical, not rewritten as successful.

The original 50-epoch traces, all query/identity changes, full measurement distributions, checkpoint/distance/source hashes and failure provenance are retained. Arrays and weights remain remote. Fixed-model identity summaries cannot replace complete training seeds; official test data has participated in development.

## Next question

Do not promote row allocation as an effective main module or tune null/gain/LR/seed from these official results. Prioritize the formation of identity-relevant regional content. A next intervention must have a direct active control and preserve the matched raw-duty recipe; no simultaneous N2/N3 stack or claim of three validated contributions.

Overall Goal remains ACTIVE/UNMET.
