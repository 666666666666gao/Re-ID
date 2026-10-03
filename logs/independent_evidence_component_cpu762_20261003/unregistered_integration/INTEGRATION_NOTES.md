# Independent native evidence: integration preparation

Status: unregistered full integration draft. The standalone component passed
one isolated CPU witness on 2026-10-03 at 10:00:13, using synthetic inputs and
eight toy updates. Its 14 tensors had cumulative nonzero finite gradients;
the initial additive exit was zero and strict component reload outputs matched.
Fresh same-family source review passed for this bounded witness only. The full
integration has not been imported or deployed, no real-data CUDA/AMP M0 or
formal training has run, and no foundation or training plan is registered.
The original F3 panel and its sole report remain in progress.

The proposed integration retains `GlobalTokenRoles` in static token mode,
including its original 128-token semantic samplers, context queries, CNN
spatial operation, CNN-to-Transformer-to-Mamba bridges, transformer global
token, two Mamba scans, and 1536-dimensional regional readout. It adds one
512-position image-native reader to CNN evidence before the existing output
normalization. Keys and values come from image-native features; semantic
context and anchor queries only supply its queries. No new matching, loss,
router, teacher, external resource or retrieval dimension is introduced.

`ImageNativeEvidenceReader.py` is the component draft. Its eventual package
filename is `image_native_evidence.py`, as referenced by the integration.
Current files are deliberately outside the canonical runtime source tree.

The added exit starts at zero. This proposes initial functional equality with
the matched semantic control; it does not guarantee retrieval preservation
after training. The first backward is expected to reach the output matrix
while inner detail parameters initially have zero gradient. A future M0 must
check cumulative nonzero gradients after actual updates, not reject the
first-step zero inner gradient or claim eventual support without observing it.

The fresh source reviewer found that a key bias adds the same score shift to
all candidates and cancels in softmax. Only that redundant bias was removed,
before any model execution. The component now has 159296 parameters in 14
trainable tensors. The earlier source-only revision is retained separately;
its 159424/15 count must not be used for the corrected implementation.

The new constructor copies all original role state and isolates extra random
initialization in the same CPU RNG fork used by the existing constructors.
A future paired initialization witness must still check original parameters,
model buffers, input order, original embeddings and role outputs directly.
This source convention alone does not prove functional parity.

The trainer remains undecided until F3 completes and is reviewed. The actual
`run_foundation_recipe.py` author branch passes only `model.signal` to
`make_optimizer`; that call cannot be reused unchanged for the new model.
Optimizer ownership must cover all intended trainable parameters exactly
once. Check missing/extra parameter identities against the model, and retain
the selected foundation's visual and task learning-rate groups. Do not silently
leave the new detail branch or existing roles unoptimized.

This is an addition of capacity and computation as well as a new information
source. A positive treatment-versus-semantic result would support the complete
addition, not isolated causality of detail content. A matched reduced-detail
or extra-semantic control is required before attributing the improvement to
the new content or resolution. Do not introduce these controls into the live
F3 campaign, and do not retune its losses or seeds.

Required next steps, after F3 closure: select a credible matched foundation;
register the finite comparison and unchanged full-gallery evaluation; obtain
fresh same-family source review; then perform actual paired forward, optimizer
coverage, cumulative gradient, finite-update and strict reload witnesses.
Only those receipts can authorize formal training. Source parsing proves none
of the runtime or retrieval claims above.
