# MSVR310 freshness diagnostic figure scope

This reporting plan does not change EXPERIMENT_PLAN.md, configuration, training code, losses or endpoint selection.

| Artifact | Data | Design | Scope |
|---|---|---|---|
| Complete preflight check | Verified 72-update text reaggregation, three folds and both endpoints | Six-panel paired dots, fold on x axis; no interpolated time curve | Completed short preflight only, one epoch per endpoint |
| Complete source diagnostic | All 1560 updates after six endpoints and full CPU verification | Same six metrics, epoch curves for all six trajectories separately | Not yet executed or render-verified; no partial-source plot |

Panels: pair-weighted absolute stale/fresh historical-distance difference; mean fresh-minus-stale expanded hinge; changed hardest-negative identity-of-record selections per historical anchor exposure; and runtime parameter-gradient cosine for CNN, Transformer and Mamba. Here a changed selection means a different candidate record, not necessarily a different identity.

Both coordinate variants use the same current graph and the same historical records/views; historical gradients remain detached. The actual Control update uses the within-batch metric loss and the memory update uses its registered cached expanded loss. Fresh losses are diagnostic only. Source model arrays/images stay remote; plotting consumes text, with source/CPU hashes in the receipt.

Every fold/endpoint/epoch is included. No-history values remain missing rather than zero; undefined gradient cosine counts are reported. No fold pooling, confidence interval, or claim of held-out improvement. The short full-learning-rate preflight cannot stand in for the complete warmup training trajectory.

Deliverables are the standalone generator, exact plotted values, vector PDF, inspection PNG, external caption/LaTeX snippet, and a render/binding receipt. A fresh same-family Codex figure review is provisional and does not establish model efficacy.
