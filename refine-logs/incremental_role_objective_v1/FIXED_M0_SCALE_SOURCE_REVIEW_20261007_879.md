# Fixed retained M0 scale diagnostic: source review 879

Date: 2026-10-07. Verdict: **SOURCE_ONLY PASS**. BLOCKING: 0. NON-BLOCKING defects: 0. Required implementation changes: none.

Reviewer: `gpt-6-astra`, reasoning effort `max`; continuation of the same independent objective/M0/full-stage reviewer, with prior context retained. `review_independence: same-family`; `acceptance_status: provisional`. This is not a new fresh-context review. No SSH, GPU access, model construction/forward, optimizer, diagnostic execution, or new agent was used. Implementations were not edited.

Reviewed `tools/diagnose_incremental_m0_gradient_scale.py` (79 lines) against `FIXED_M0_SCALE_PLAN_20261007_879.md`, the current entry/configuration chain, actual loader/load/state producers, and locally received failed-M0 records. Source SHA256: `c0c7b4c96fe291290f83ff5c42edfc47847995bfdc4c07b97ab12e3f90a1e4b4`. Plan SHA256: `e0059a58b7d155a897c6ddf425de925ca15d8ca30bfc70bb0b820523fc34d513`.

## BLOCKING

None. No concrete current source integration or gradient-measurement defect was found in this fixed-state, single-graph comparison.

## NON-BLOCKING

None requiring an implementation change. Runtime execution and its result remain unaccepted by this source review.

## Checks against actual callees

1. **The retained state and original initializer are used.** Lines 21-39 select only the failed RGBNT201/md_batch_ratio run, require its production eight-step M0 receipt, and match the retained probe hash. The Namespace keeps semantic/seed42/epochs50, the same public CLIP, protocol, author package, and original `RGBNT201_md_batch_ratio.json`. `entry.configure()` follows the reviewed incremental -> global-task-role -> detached-role-input -> native-research -> partitioned -> independent-evidence chain. Its final `foundation.build` calls the incremental build, checks the old initialization witness schema and exact binding, then line 38 additionally checks the actual training receipt's initializer. `foundation.load` verifies checkpoint schema/dataset, exact current condition (including the original initializer file hash), and protocol hash, then calls `load_state_dict(..., strict=True)`. This loads the retained eight-update weights; it does not mistake the public initialization for the measured state.

2. **The original batch log is not opened for writing.** `entry.inner.original_train_loader` is the unwrapped foundation loader captured before configuration. It does not call `BatchOrderLoader`, its `touch(exist_ok=False)`, or its append loop. The script reads the old batch log only. The actual semantic path uses the author `ImageDataset`, transforms, `RandomIdentitySampler(..., 42)`, and collate function. The sampler resets Python/NumPy RNG at construction. The received old first row has 64 paths/labels/cameras and the initializer specifies B64/K8. Lines 44-45 require exact first-row path and label order before any model forward. `ImageDataset` and collate produce the expected raw[4] basenames and raw[1] labels. The cached author sampler/dataset/collate sources read for this review have hashes matching the current 393-item source scope.

3. **Metadata and model mode match this endpoint.** The environment map uses RGBNT201 protocol camera labels and the existing training converter. The actual previously collected basename witness has no conflicts; no MSVR view-proxy inference is introduced into this fixed RGBNT201 diagnostic. `model.train()` invokes the existing author-head mode restoration, so the single forward has the production train-mode BN behavior and existing detached role-read/global-reference policy. No training wrapper, M0 receipt writer, optimizer constructor, or evaluator is invoked.

4. **The scale comparison shares one graph.** Lines 48-50 perform one AMP float16 model forward and call the unchanged `batch_ratio_loss`; that objective's internal arithmetic remains FP32 as in production. The six requested Q/K weights are the actual three query and three key projection matrices. Both `autograd.grad` calls use the same scalar and graph with `retain_graph=True`; changing only the scalar multiplier from 1 to 256 does not rerun data, dropout, model construction, or the forward. The shared-global derivative is asserted absent. Each c/Q/K derivative is checked finite when present and reported with an explicit `unused` flag; norms and maxima are computed after conversion to FP32 and division by the scale. Thus an unused derivative and a present zero tensor are distinguishable. Production `foundation.train` initializes GradScaler to 256 at line 198. The diagnostic preserves, rather than edits, that training source.

5. **The diagnostic has zero optimizer updates and restores model tensors.** `autograd.grad` returns derivatives without populating parameter `.grad`; line 63 asserts every such field remains None. No optimizer/backward/step or model checkpoint save occurs. All named buffers are cloned before the forward and copied back under no_grad after both VJPs. The existing `_module_state_sha256` covers all named state_dict tensor bytes, and equality before/after restoration is required before writing success. Together with no parameter update, this checks exact retained model tensor state. It is not a claim about restoring process RNG or replaying historical augmentation. The only diagnostic output is the new requested JSON and stdout; the probe, old initializer, old training receipts and logs are only read.

6. **The registered interpretation is narrow.** The output identifies the probe, one batch, one forward, two VJPs and zero updates. Its boundary explicitly excludes reconstruction of the old eight gradients, reversal of their FAIL, a general underflow-root-cause claim, or full-campaign qualification. The fixed-scale plan also keeps the other five M0s and formal queue stopped. No acceptance thresholds, loss/gain/seed/margin, or old 393-item implementation source is changed here.

## Actual failed-M0 evidence remains unchanged

The received original receipt reports production `M0_PASS`, eight updates, 281/281 total-loss gradient tensors, author BN8 and reload difference 0. The additional isolated gate failed: c cumulative norm `0.06013888958841562`, CNN q0/k0 sums `3.4250635962962406e-6` / `1.9858277937601088e-6`, and q1/k1/q2/k2 sums zero. Global gradients were absent. The controller ended with exit 1, and no auxiliary acceptance was produced. These facts are not converted into an accepted M0 by this source PASS.

The old isolated diagnostic used unscaled `autograd.grad`, whereas production total-loss backward used GradScaler. The proposed fixed-state comparison can establish scale sensitivity for its particular retained state and source batch only. Equal source filenames/labels are not a saved augmented-tensor replay, and the retained eighth-update state is not any earlier old-step state. Even a nonzero scaled result cannot establish all six parameters' historical eight-step activity or prove the sole root cause of the recorded failure.

## Remaining runtime boundary

The executor still must complete its stated prelaunch pinning of checkpoint/initializer/receipt/source inputs, preserve the original failure and retained probe, and receive the actual diagnostic JSON before interpreting any scale effect. This review did not inspect a new remote launch, accept actual first-batch equality/state restoration, or measure a new derivative. It does not authorize or qualify the remaining five M0s or six formal runs.
