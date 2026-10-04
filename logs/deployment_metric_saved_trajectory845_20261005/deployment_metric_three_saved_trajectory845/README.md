# Three accepted full50 trajectory pairs

This local saved-text analysis covers 150 completed epochs and 6,004 formal optimizer steps. It runs no neural model and does not inspect the current training process.

| Dataset / variant | Selected epoch | Selected ΔmAP / ΔR1 | Scaled correction/global at selected epoch (old → new) | Epochs with ΔmAP > 0 | Epochs with ΔR1 < 0 |
|---|---:|---:|---:|---:|---:|
| RGBNT201 / semantic | 8 | +0.561419 / -0.119615 | 4.8096% → 61.8222% | 47/50 | 13/50 |
| RGBNT201 / native | 8 | +0.566356 / -1.196170 | 4.0596% → 61.7549% | 44/50 | 19/50 |
| MSVR310 / semantic | 38 | -0.030107 / +0.000000 | 0.5318% → 0.5772% | 1/50 | 29/50 |

All 50 recorded epoch means of the global objective and shared-global feature norm match their respective raw-role control exactly for all three endpoints. This supports the intended isolation at the recorded-scalar level; it does not establish bitwise equality of every model tensor.

The two RGBNT201 endpoints show mAP improvement across most saved epochs, while Rank-5 and Rank-10 decrease in most epochs. MSVR310 semantic has lower mAP in 49/50 epochs. These are descriptive training trajectories; dependent epochs do not replace independent training seeds. Each registered selected best and the original mAP +0.5 / Rank-1 non-decrease gate remain unchanged; all three selected pairs fail that gate.

Changing the role metric changes its training loss scale and geometry. The displayed role objective combines identity and Triplet terms; absolute old/new loss values cannot alone show worse fitting, an inactive Triplet, or a gradient conflict. Larger correction norms are observed behavior, not a demonstrated cause of the CMC changes.

Three accepted original full50/first-strict pairs only. This is a saved-text descriptive analysis, not a new model run, full six-endpoint report, best reselection or training-seed uncertainty. Epoch counts are dependent observations. Global loss and norm equality are recorded scalar equality, not a claim of all model tensors being bitwise equal. Role losses combine classification and Triplet and change metric geometry; their absolute sizes do not directly measure relative fitting quality. Correction/global ratios are means over training batches, not query ratios or causal proof. MSVR310 native failed its original M0 and has no formal result; RGBNT100 is not included. All registered mAP-best and progress gates remain unchanged.
