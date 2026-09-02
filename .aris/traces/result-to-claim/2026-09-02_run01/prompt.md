# RESULT-TO-CLAIM EVALUATION — V15 Q1

Intended claim: CRDE creates stable CNN/Transformer/Mamba collaboration on
identities unseen by each fold, improving fused and all three receiver branches
over the exact same-fold no-exchange comparator; Q1 pass would only authorize
D1.

Experiment: remote RTX3090, RGBNT201 fit-only complete-path identity-OOF,
seed42, three registered folds, B64/K8, 20 epochs/fold, final-only, dev0/
official0, clean commit `71152d3848c05177da0af30b0b921c6a3aa9942a`.

Key results: fold fused gains `+0.0951795/-0.8310606/+0.1605191`; aggregate
fused `-0.1720681`; CNN/T/M aggregate `-0.1575651/-0.2605891/+0.2897778`;
bootstrap lower bound `-0.9503304`; five scientific gates fail. All integrity
checks pass, 110/110 gradients per fold, zero overflow, frozen state unchanged,
dev0/official0, D1 false.

Return claim support, supported/unsupported evidence, gaps, revised wording,
next action inside the frozen boundary, confidence, integrity and routing.
