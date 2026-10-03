# Matched foundation training interface — source preparation only

This preparation does not select a foundation or register a training run.
F3 is still running under its original sealed source, on server2026 only.
The component CPU witness from section41.762 remains bounded to that component;
the revised full integration has not been imported or executed.

## Actual caller and callee facts

`tools/run_foundation_recipe.py` selects the pinned Signal source through
`run_clean_clip_joint._configure_signal_source`. The canonical project's
`modeling/make_model.py` belongs to the DeMo base and is not the F1 author
callee. This inspection uses the actually archived
`comparators/Signal-cd1b0a6/modeling/make_model.py` and its loss/optimizer/configs.

The author RGBNT201 configuration has DIRECT=1: one BN/classifier consumes
the raw concatenated1536-dimensional feature. RGBNT100 and MSVR310 have
DIRECT=0: each uses three BN/classifiers on the512-dimensional modality
features. The author `make_loss` selects soft Triplet when NO_MARGIN is true,
and F1 sums the designated head losses. This is different from the current
package's one BN/classifier on the L2-normalized1536-dimensional feature and
margin0.3 Triplet.

The existing `GlobalTokenTriFusion.forward` normalizes the fused feature
before its own neck/classifier. Reusing that auxiliary output with the
author optimizer does not reproduce author head or loss behavior. Calling
`signal.forward(training=True)` again would encode the images again and
supervise the Signal feature, not the role-corrected fused feature.

The existing `CrossLayerAdaptedCLIP.train` forces `signal.eval()`. Its Signal
heads are initially frozen along with the original Signal state. A future
author-head trainer must explicitly enable only the intended heads and set
their train/eval mode; blindly reusing that mode behavior leaves BN heads in
evaluation mode. This is an observed source boundary, not a claim that every
visual training-mode difference is irrelevant.

The author F1 optimizer receives only `model.signal`; applying it unchanged
to the new model omits roles/adapters/detail. The future optimizer must own
every intended trainable parameter exactly once, retaining the chosen
foundation's actual per-parameter learning rates and weight decay. The
MSVR310 author config has BASE_LR=5e-6 and special classifier handling;
do not replace it with a blanket3.5e-4 group while calling it author-matched.

## Minimal interface change

The unregistered integration now contains `RawFeatureSemanticTriFusion`,
a static-token semantic control, and `IndependentNativeTriFusion`, which
changes only the role-evidence call to include image-native detail.
Both share `forward_features`: one backbone/role/readout execution returns
`raw_fused`, its L2-normalized `fused`, `shared_global`, and `correction`.
The feature-only method does not invoke any classifier or BN neck.

The ordinary `forward` retains normalized deployment and the historical
normalized auxiliary classifier for explicit old-interface comparisons.
It must not be mistaken for the future selected foundation training path.
No head/optimizer/loss/sampler/scheduler choice is made by this draft.
The previous integration revision is preserved in this directory.

## Remaining gates

After the original all-six F3 report and fresh audit/claim review, select
and register a credible matched foundation for global-only, semantic roles,
and semantic-plus-detail. Explicitly specify head ownership, BN modes,
CE/metric input features, loss aggregation, optimizer groups, augmentation,
sampler, scheduler and complete state persistence before source review.

Actual paired initialization/forward, full optimizer coverage, cumulative
gradient support, AMP/CUDA updates and full-model strict reload remain
necessary before formal training. No AST/source inspection substitutes for
these checks or for retrieval results. No scale/margin/seed rescue is opened.
