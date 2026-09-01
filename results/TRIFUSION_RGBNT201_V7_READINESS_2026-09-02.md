# TriFusion Signal-preserving V7 readiness — 2026-09-02

V7 is a single main-only correction derived from the terminal V6 failure. It
does not constitute an ablation, official-test run, or retrieval result.

## Why all three experts remain

The read-only V6 ground-truth Oracle processes all 825 fixed-dev queries with
optimizer steps and official-test access both zero.

| Diagnostic | Best fixed mAP | Oracle mAP | Oracle gain |
|---|---:|---:|---:|
| baseline + expert residual | CNN 59.1022 | 63.6089 | +4.5067 |
| residual-only expert | CNN 56.9293 | 61.1233 | +4.1940 |

Branch-Oracle leave-one-expert-out marginal mAP is +0.6137 CNN, +0.4804
Transformer and +0.9620 Mamba. The learned V6 router, rather than absent expert
complementarity, is therefore the primary demonstrated failure. Oracle uses
ground truth and is not a deployable metric; 63.6089 also remains below 65.

## V7 method identity

1. One sampled flip/crop is shared by RGB/NIR/TIR; erasing remains modality
   independent after geometry is aligned. Expert residuals are computed from
   matched `expert_token - anchor_token` fields.
2. The existing two HFER exchanges and three reliability refreshes are retained
   across CNN, Transformer and Mamba residual experts on frozen Signal tokens.
3. Routing factorizes as `P(modality) * P(expert | modality)`, is supervised by
   every expert-modality slot's L2-normalized marginal identity gain and by a
   separate controlled-degradation view, then uses sample alpha in `[0, 0.5]`.

The exact 3072D Signal embedding remains the fused prefix and an independent
`baseline_only` output. The experts are shallow heterogeneous residual adapters,
not three independent full CNN/ViT/Mamba backbones.

## Remote RTX 3090 readiness

| Gate | Result | Evidence |
|---|---|---|
| focused regression | PASS | 32 tests; only timm deprecation warnings |
| preflight | PASS | exact Signal tensor and 58.0109/57.4545 metric parity; V6 load missing exactly four new alpha tensors; official0 |
| capacity | PASS | B64/K8, two views, 8 steps, 222/222 gradients, overflow0, 11,486 MiB peak reserved |
| fixed-batch overfit | PASS | 100 steps; analytic smoothing floor 0.610636; excess loss ratio 0.08048; Signal unchanged; official0 |

The epoch-10 router-warmup boundary additionally requires blur of each modality
to strictly lower that modality's mean routing mass. Failure blocks the joint
phase. The frozen development gate is unchanged: fused mAP at least 65 and
strictly above baseline_only, CNN, Transformer and Mamba. Only seed 42 is
permitted; no ablation, multiple seed or official test is authorized.
