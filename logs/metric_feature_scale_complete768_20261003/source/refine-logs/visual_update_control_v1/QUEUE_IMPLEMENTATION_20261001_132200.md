# Visual update/readout implementation addendum

PREPARATION ONLY. No new M0, initialization construction or formal endpoint has run.

The unchanged scientific design is seed42, twelve fresh full50 endpoints,
frozen/low_lr visual weights crossed with global_only/roles for all three datasets.
Ordinary fine-tuning is a control, not a novelty claim. Original gates remain unchanged.

The first RGBNT201 low_lr_roles and low_lr_global_only eight-batch M0 processes
run before formal registration. Their actual command, PID, start/end timestamps,
exit status and output paths are written to a preflight receipt. After a separate
twelve-model common-initialization witness, the formal queue verifies and reuses
these two original M0 receipts; it does not execute those same M0 stages again.
Their child campaign rows explicitly carry origin=verified_preflight. The other
ten endpoints run their own M0 before full50 training and strict evaluation.
Full50 training always starts from the original baseline and seeded initialization,
never from an M0 optimizer-updated checkpoint.

The preflight records the complete existing source snapshot before either launch.
The initialization witness checks this same snapshot before and after construction;
formal registration requires exact equality. Preflight completed_at is the parent
observation after wait, not an exact process finish timestamp. M0 runtime is taken
from the child training receipt, with its stated construction/measurement boundary.

The witness constructs all four production models per dataset on one free GPU,
compares backbone/neck/classifier state tensors bitwise, checks FP32 visual
storage and the allowed visual trainability, and writes all twelve actual bindings.
It performs no input forward, optimizer step or retrieval evaluation.

The scheduler reuses the existing 240-second GPU reservation loop. It requires
the original six-end campaign and once-only complete analysis to have finished,
binds the original analysis SHA, current sources, protocols, baseline weights,
initialization witness and preflight receipt. Any subprocess or verification
failure stops pending launches and drains existing jobs; no retry or batch change.

The collector validates full50, the single official-mAP best checkpoint, full
FP32 visual state saving, M0/reload evidence, exact protocol query/gallery arrays,
legal camera/time filtering and independent complete-gallery score recomputation.
It also checks logged CE+Triplet reconstruction and reports actual elapsed time
and allocated peak memory. Peak memory excludes initial construction transient;
it is not reserved memory. All weights and distance arrays stay on the server.

Source review and native syntax checking are required before this code is used.
Passing them does not mean production M0, performance, stability or SOTA passes.
