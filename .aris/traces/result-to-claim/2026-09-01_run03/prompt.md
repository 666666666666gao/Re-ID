# Result-to-Claim Reviewer Prompt

Intended claim: Signal-preserving V5 retains the exact 3072D Signal baseline while CNN, Transformer and Mamba exchange features and produce a fused representation that beats the baseline and every expert, reaches the frozen 65 mAP held-out-dev gate, and can proceed toward an official SOTA attempt.

Evidence supplied to the independent reviewer:

- RGBNT201 fixed 141-fit/30-dev, seed 42, B32/K4, 60/60 epochs, official-test access 0.
- Selected epoch 51 after fused-dev-mAP selection and strict same-checkpoint reload.
- baseline `58.0108866 mAP / 57.4545443 Rank-1`.
- fused `58.0167562 / 57.4545443`.
- CNN `58.0180597 / 57.4545443`.
- Transformer `58.0137158 / 57.4545443`.
- Mamba `58.0135269 / 57.4545443`.
- Signal state SHA unchanged before training, after training and after strict reload.
- 5,498 optimizer steps, zero AMP overflow, one dataset and one seed.

Requested fields: claim support, supported and unsupported statements, missing diagnostic evidence, revised claim, next experiment under the no-ablation/no-multi-seed/no-official-test constraints, and confidence.
