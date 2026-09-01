# V8 Phase-B independent experiment-integrity audit

Date: 2026-09-02

Scope: OOF-margin target repair, frozen-expert hierarchical Router training,
and the single frozen RGBNT201 141-fit/30-dev evaluation

Overall verdict: **WARN**

Integrity status: **warn**

The warning is about reproducibility packaging, not a discovered metric,
ground-truth, or data-leakage failure. The three lightweight Phase-B receipts
are included in Git by the sealing commit, while the large checkpoint and
cache files remain on the remote RTX3090 host and cannot be re-hashed from a
fresh local clone.

## Audit matrix

| Area | Verdict | Evidence-based finding |
|---|:---:|---|
| A. Ground-truth provenance | PASS | OOF identity margins use model-derived fit-only features with real identity/camera labels. No dev or official labels enter Router training. Phase-A Oracle remains explicitly diagnostic and non-deployable. |
| B. Score normalization | PASS | Margin and dev retrieval use ordinary feature L2 normalization and Euclidean ReID distance. Router softmax/KL and bounded residual scaling do not self-normalize AP/mAP. |
| C. Result existence and provenance | WARN | The three result JSON files and their reported numbers are mutually consistent, configs pin input SHAs, and loaders verify them. Large `.pth` checkpoint/cache artifacts remain remote-only, so the independent local reviewer could not re-hash them. |
| D. Dead code / called path | PASS | Margin repair, OOF fold training/evaluation, quality loss, Router gate, routed fusion, and dev evaluator are all on executed paths. |
| E. Scope and leakage | PASS | Protocol is identity-disjoint 141-fit/30-dev; Router training reports dev0/official0; the combined checkpoint has one frozen dev evaluation and official0. Claim language keeps the failed 65 gate explicit. |
| F. Evaluation type | PASS | OOF margin and frozen dev retrieval use real GT; the controlled blur quality gate is a synthetic proxy; Phase-A Oracle is real-GT diagnostic only. |

## Numeric consistency

| Check | Audited value |
|---|---:|
| OOF queries | 571 |
| Router optimizer steps | 400 |
| Router training dev / official access | 0 / 0 |
| Learned / fixed OOF expected margin | 0.1020340 / 0.1017202 |
| Learned / majority Top-slot accuracy | 17.8634% / 17.6883% |
| Frozen dev baseline / fused mAP | 58.0109 / 58.4050 |
| Frozen dev CNN / Transformer / Mamba mAP | 57.6071 / 56.3031 / 56.6260 |
| Dev optimizer steps / dev access / official access | 0 / 1 / 0 |
| Frozen dev promotion gate | FAIL; 6.5950 below 65 |

The local review could not directly open the remote large artifacts. A
separate read-only remote verification during sealing re-hashed the Phase-A
checkpoint as
`d37ca17fad8a2786355d575b009f453e170238f3057c5db8f68789bb32b1b40f`
and the combined checkpoint as
`6f95f99a86763580c3bd8592974347825659a5336f9afec43062516d21fbfe02`,
matching the receipts. This reduces accidental-copy risk but does not remove
the audit's remote-only packaging warning.

## Claim impact

- Supported weakly: the fit-only OOF and controlled-quality training path has
  the intended behavior, but learned-vs-fixed OOF advantages are extremely
  small and must not be described as strong routing generalization.
- Supported narrowly: on the single frozen seed42 dev evaluation, the complete
  Phase-B fused output beats the exact same-checkpoint Signal baseline and all
  three fixed experts.
- Unsupported: adequate causal contribution from learned routing, 65 mAP,
  HFER effectiveness, official-test improvement, SOTA, robustness across
  seeds, or cross-dataset generalization.

## Required disposition

Seal Phase-B as positive but not promoted. Keep HFER, ablations, multiple
seeds, official test, and Router/alpha/epoch/LR scans closed. A successor must
pre-register a new representation-level hypothesis and repeat the train-only
integrity gates before another frozen dev read.

Lightweight receipts:

```text
evidence/trifusion_v8_oof_router_margin_targets_seed42.json
evidence/trifusion_v8_oof_margin_router_phase_b_seed42.json
evidence/trifusion_v8_oof_margin_router_dev_seed42.json
```
