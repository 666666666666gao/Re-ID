# Paired RGBNT201 reader activity — closed 901 logs

This is a descriptive analysis of the original 2,649 training-step records and all 50 epoch records per condition. Four inputs match the fresh partial audit's recorded SHA256 values. The analysis runs locally without torch, a new forward, retrieval rescoring or a runtime progress query. Complete epoch summaries are in PAIRED_R201_READER_ACTIVITY_901.json (SHA256 540ae3d7cb76f0c3f271e6ace48c4bb8e1c7d6ededfad13d342c167466d3da05).

Both arms selected epoch 8. Pretrained fused is 74.355298 mAP / 78.947371 Rank-1; random is 74.477435 / 79.425836. The primary pretrained-minus-random comparison is negative and does not satisfy the registered advancement criterion. At epoch 50, their mAP values are 71.346461 and 71.537659. This is not evidence for pretraining necessity or statistical equivalence.

The Transformer/Mamba readers are almost uniform already at the selected epoch, and their Q/K magnitudes become much smaller by the final epoch in both arms. The phenomenon is shared by the two packages, so it cannot be attributed solely to pretrained text.

| Condition / epoch | CNN entropy / ln(128) | Transformer entropy / ln(128) | Mamba entropy / ln(128) | Transformer Q / K norm | Mamba Q / K norm |
|---|---:|---:|---:|---:|---:|
| pretrained / 8 | 0.97220248 | 0.99999807 | 0.99999760 | 1.47886 / 1.38415 | 1.52902 / 1.43184 |
| random / 8 | 0.92820746 | 0.99999784 | 0.99999755 | 1.47446 / 1.40162 | 1.52487 / 1.43064 |
| pretrained / 50 | 0.99903043 | 1.00000010 | 1.00000010 | 1.71e-5 / 1.19e-5 | 1.78e-5 / 1.21e-5 |
| random / 50 | 0.99416119 | 1.00000010 | 1.00000010 | 1.58e-5 / 1.39e-5 | 1.63e-5 / 1.36e-5 |

Selected-epoch Transformer/Mamba logit standard deviations are roughly 0.004 in both arms. At epoch 50 they are 8.47e-19 / 0 for pretrained and 2.28e-18 / 3.55e-18 for random. CNN retains materially larger Q/K norms: pretrained 1.35781 / 1.40961 and random 2.25982 / 1.41364 at epoch 50. Do not describe all three readers as having 1e-5 Q/K norms.

The actual scaled correction/global norm ratio decreases from 0.0509136 to 0.0217169 in pretrained and from 0.0459127 to 0.0310932 in random. These are batch diagnostic means, not per-query error attributions or proof that correction magnitude caused the mAP decline.

The logged same-identity and different-identity text dot-product means coincide at both selected and final epochs in each arm. These diagnostics are produced in an AMP region, and the pretrained final mean is 1.00042379. They do not prove exact equal text vectors, perfect cosine agreement, identity-information absence or a unique numerical root cause. Entropy ratios slightly above one also retain the original finite-precision values; no clamping or threshold adjustment was applied.

The analysis supplies a reason to investigate whether readers learn useful content selection. It does not validate a replacement mechanism, identify a single cause such as weight decay, or authorize a new experiment before the fixed pilot/report closes. No model, loss, optimizer, acceptance rule or active source file changed.
