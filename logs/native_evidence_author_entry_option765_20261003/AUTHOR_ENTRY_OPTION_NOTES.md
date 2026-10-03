# Full50 author entry option — unregistered and unexecuted

This completes a source draft for the author-head option, without selecting
it as the next foundation. No runtime imports, initializer, paired forward,
M0, CUDA job, training or evaluation have occurred for this draft.

It reuses the sealed F1 full50 training/evaluation/state functions and the
F2 BatchOrderLoader, with the author loader, sampler, augmentation, loss,
per-dataset learning rates and scheduler. All roles/heads/detail are placed
under the full optimizer through the wrapper. Historical normalized heads
remain frozen and unused. Three draft variants are global-only shared
adaptation, semantic roles, and semantic roles with independent native detail.
No candidate here is a replacement for F1's no-adapter plain baseline.

The native constructor is isolated in the existing CPU RNG fork, copies the
complete original semantic state into the augmented state, and then restores
visual/camera gradients for every variant. The constructor otherwise freezes
Signal when creating its new backbone. The common backbone/neck/classifier
state is checked before wrapping. A real initializer must still check all
common role tensors and inputs, not infer equality from this source.

The inherited loop retains eight-update M0, cumulative gradient support,
finite AMP updates, full-state save/reload, all50 epochs, latest exact-tie
mAP-best selection and strict full-gallery evaluation. The author wrapper's
heads supervise raw corrected features; the deployed vector remains L2_1536.
Consumed benchmark selection and all timing/reload reconstruction limits
remain disclosed. The inherited history.seconds measures training-loop time,
so future wall-time estimates must use real epoch completion timestamps,
not the inherited timing label or sums of that field.

Before execution, the complete selected plan still needs initial real-batch
semantic/native equality, matched all-epoch batch-order records, exact
production optimizer ownership/groups, cumulative detail gradients, BN
running-mode checks and full-model reload. A source reviewer must read this
entry, both model drafts, the selected immutable plan and all actual callees.
Neither AST parsing nor the old synthetic component CPU witness proves
these gates or accepts this trainer. No queue, final report or campaign is
registered; the current F3 process remains untouched.
