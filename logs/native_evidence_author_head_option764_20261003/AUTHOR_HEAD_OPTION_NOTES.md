# Author-head option — preparation, not foundation selection

`evidence_author_heads.py` provides a concrete option for the observed author
training boundary. It does not choose this option over the current package,
register a plan, or authorize model execution. The all-six original F3
report and fresh audit/claim review remain prerequisites for that decision.

The global-only control registers only the shared backbone and historical
normalized heads, without roles or their readout parameters. Semantic and
native evidence models expose the common raw-feature interface from section
41.763. `AuthorHeadEvidence` uses the same existing Signal head initialization
for all of these conditions and freezes their unused normalized heads.
It introduces no new classification parameters. The future initializer must
still prove shared state/input equality, rather than infer it from this code.

The author wrapper applies the existing one1536 head for DIRECT=1 or three512
heads for DIRECT=0 to `raw_fused`, after role correction. It does not call
Signal.forward again, detach features, or use the normalized vector for
author head training. Deployment still returns the normalized1536 vector.

Calling train/eval on the wrapper explicitly restores Signal's corresponding
mode after CrossLayerAdaptedCLIP forces eval, including native BN heads and
the visual encoder. Thus a future matched global/semantic/native comparison
can use the same mode contract. Do not claim mode parity with older role
runs, whose inherited Signal mode was different.

`build_author_optimizer_and_loss` delegates to the pinned author factories
with the whole wrapper, retaining their existing name-based LR/decay rules.
It checks intended trainable parameter identities against optimizer groups
exactly once, without omitting roles or detail. This is source preparation;
actual optimizer groups and cumulative gradients still require a real M0.
The returned loss factory is the author loss, not a new retrieval objective.

The future trainer must retain the chosen dataset's exact sampler,
augmentation, schedule, head-loss aggregation and full-state serialization.
It must record frozen unused heads and total/trainable parameter counts.
This file does not supply that trainer, and AST parsing does not prove any
runtime behavior. No full integration, real-data paired forward, CPU model
execution, AMP/CUDA M0 or formal training was performed for this option.
