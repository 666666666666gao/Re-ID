# MSVR310 paired history-candidate gradient: independent preexecution audit

**Overall verdict: WARN — no blocking implementation defect remains in the final inspected revision. The registered fixed pipeline is ready to enter T0/M0, with Q1 conditional on its actual M0 and complete CPU receipts. This is not an engineering-runtime PASS or a scientific qualification.**

Date: 2026-09-08. Reviewer: fresh native Codex agent `/root/audit_msvr_history_gradient_training`; model **gpt-6-astra**, reasoning effort **max**; **same-family / provisional**. Sources were read directly. I did not import project modules, instantiate a model, run the author's synthetic/T0 code, execute a backward, train, infer, open dataset images, or issue a remote command. Only independent review/check artifacts were written under `C:/Users/gb/.codex_tmp/history_gradient_training_preexecution_audit_20260908`.

`R` below means `C:/Users/gb/.trifusion_github_publish_22c3bee`; source references are relative to R. The exact final input inventory is `audited_input_hashes.json` (74 entries, including the independent checker). Initial and final versions of the seven primary files are retained under `inputs/` and `inputs_final/`. The current checkout HEAD observed by the checker is `e83efaa06ae87f8825e3451778cda787b05ea6d7`; the future execution must bind its own actual commit and final configuration SHA through the wrapper.

## A–F findings

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance and complete-path isolation | WARN | All 1,032 protocol records were independently reconstructed from the SHA-bound existing label inventory; all three source/heldout partitions, local label maps, 600 valid query masks and complete galleries agree. Model and external Signal-source bytes still require the runtime contract checks; no new image/weight execution occurred here. |
| B. Score/gradient normalization and parameter alignment | PASS | Standard feature normalization, Euclidean batch-hard Triplet, true-ID masks, current 64-anchor mean, lambda=1, and matching 42/54/93 role-encoder blocks. One scaled backward, grouped candidate VJP accumulation, optional history addition, one unscale and one AdamW step. No model-output normalization of AP/CMC. |
| C. Existence, hash contracts and stated status | PASS | Four new scripts pass independent syntax checks and their exact configured hashes. Final plan/config binding is exact. Tracker correctly says no new T0/M0/Q1 result. Prior fixed-state source summary/CPU mirrors match their two prerequisite hashes and are not treated as results of this training implementation. |
| D. Live calls and proof coverage | WARN | T0 → M0 → M0 CPU → Q1 → Q1 CPU are live wrapper stages; original complete evaluation and both five-gate decisions remain live. The only detected plan/code discrepancy, the first M0 history step number, is corrected and rechecked. Model gradients/reencoding are runtime witnesses; the direct real-model proof is one single-history-group encoder comparison per capacity arm. |
| E. Scope, compute, budget and runtime assumptions | WARN | Fixed seed42, original initializations, B64/K8, 20×13 updates, both arms and all folds are preserved. Queue replay covers every registered batch. Full pipeline completion, measured capacity/speed, paired augmented pixels and direct-gradient tolerances remain untested. The single-seed, repeatedly used internal evaluation and unexercised capacity-eviction branch remain limitations. |
| F. Evaluation classification | PASS | Planned internal identity-OOF retrieval uses `real_gt`; T0 tensor math is `synthetic_proxy`; capacity/overfit are source engineering checks; saved-matrix/component arithmetic is a CPU consistency proof; parameter gradients and exact reencoding are runtime-model witnesses. No new retrieval or training result is asserted. |

## A. Source identities, inherited imports and scene filtering

The inherited context chain is `train_msvr_history_gradient.context` → `train_msvr_fresh_coordinate.context` → `train_msvr_instance_memory.context` → `train_msvr310_source_style.context` → `train_msvr310_signal_oof.configure`. It validates the new scripts/plan/prior audit, earlier memory settings and immutable source receipts, original model/loss source, protocol, source metadata, Signal checkpoint hashes and baseline retrieval-array hashes (`tools/train_msvr_history_gradient.py:26-40`; `tools/train_msvr_fresh_coordinate.py:26-38`; `tools/train_msvr_instance_memory.py:23-36`; `tools/train_msvr310_source_style.py:25-51`).

The source-style configuration is inherited for its frozen original model/optimizer/protocol values. This runner calls the original `build_model`, not the source-style `build`; it does not activate the inherited STYLE intervention (`tools/train_msvr_history_gradient.py:15-19,281`; `tools/train_msvr310_source_style.py:54-61`). Signal is loaded strictly from each fold's own epoch50 checkpoint; source/heldout identity metadata and final Signal state SHA must match. Roles are freshly initialized at seed42, with `role_weights_loaded=False`; M0 weights are not reused for Q1 (`tools/train_msvr310_trifusion_oof.py:27-55`; new runner `251-286`).

The independent stdlib checker read the complete existing label inventory and verified each identity/camera/scene filename field, each three-modality path, every record index, every fold, every contiguous source-label mapping and every valid-positive/exclusion count. The builder's source is dataset-provided labels, not model predictions (`tools/build_msvr310_train_oof_protocol.py:12-55,57-123`).

| Fold | Source records / IDs | Gallery records | Valid queries / query IDs | Records without query positives retained in gallery | True ID assigned local class0 |
|---|---:|---:|---:|---:|---:|
| 0 | 672 / 103 | 360 | 210 / 20 | 150 | 5 |
| 1 | 683 / 103 | 349 | 207 / 20 | 142 | 1 |
| 2 | 709 / 104 | 323 | 183 / 20 | 140 | 1 |

The fold-local galleries cover all 1,032 official-training records exactly once; the 600 query records are unique. All 95 single-scene identities remain gallery distractors. `records_for(..., True)` only constructs the source image records; `False` uses the complete fold gallery after the M0 gate. Local classifier class0 is retained, while memory mining uses original identity values and an equality mask. Scene is used for retrieval filtering and diagnostic counts, not as a negative identity label (`tools/train_msvr310_signal_oof.py:68-99,223-238`; `tools/msvr_instance_memory.py:40-80`).

The actual upstream `utils.metrics` and sampler imports are resolved after `_configure_signal_source` inserts the pinned external Signal repository at the front of `sys.path` (`tools/run_signal_baseline_dev.py:57-64`; `tools/train_msvr310_signal_oof.py:31-65`). The same-named local `utils/metrics.py` has different bytes and is not evidence that those local bytes will be executed. The intended external evaluator path is hash-bound in `configs/MSVR310/Signal-source-oof-v1.json`. Its imported `eval_func_msrv` is called and numerically cross-checked against the explicit same-ID AND same-scene evaluator; complete raw ranking/AP/CMC replay is also implemented in the new CPU verifier (`tools/train_msvr310_trifusion_oof.py:196-241`; `tools/verify_msvr_history_gradient.py:179-216`). Live external-source resolution was not executed by this audit.

## B. Exact learning difference and complete metric parameter path

Let `L(U,H)` be the expanded fused Triplet with current normalized features U as its only anchors, and fresh historical normalized candidates H. Both endpoints compute the same scalar definition, same source instances/queue policy, and same no-grad historical refresh algorithm at their respective current parameters. Their parameter trajectories can diverge after the first historical update, so corresponding numerical features, hard candidates and VJP group counts are not required to remain equal.

`G` is the original weighted 14-term current total gradient with H detached. `V` is the candidate-side chain-rule contribution to `L`, restricted to the trainable role encoder. The control applies G; the candidate applies `G + LOSS.TRIPLET_FUSED * V`, with that weight asserted to equal1. Outside the encoder, the two update rules both apply the ordinary current total gradient. This is the sole endpoint-dependent gradient mutation in `fit` (`tools/train_msvr_history_gradient.py:55-75,111-123,154-203`). It is not a history-anchor or symmetric enlarged-batch objective.

The differentiable helper independently computes the same current/current and current/history distances and masks, takes their joint hard positive/negative and averages exactly over the current rows. Current features remain on both sides of `cdist(unit,unit)` for G; detaching current features in the separate leaf calculation only isolates the historical partial derivative. The historical leaf is FP32 and the scalar must be bitwise equal to `pooled` before its upstream gradient is accepted (`tools/probe_msvr_history_candidate_gradients.py:26-50`; `tools/msvr_instance_memory.py:40-80`; new runner `103-123`).

The live criterion provides seven CE and seven Triplet components, with original margin0.3 and label smoothing0.1. The current fused Triplet is the only replaced component. The inherited architecture-specific `weighted_training_loss` excludes unrelated losses for this V8 architecture and uses the unchanged configured weights (`modeling/trifusion/signal_preserving_v8.py:690-742`; `tools/run_signal_preserving_v5.py:99-142`).

The model source establishes the complete metric path: baseline and pretrained CLIP tail frozen; trainable CNN/Transformer/Mamba stages and heads; parameter-free `ExpertFormationFusion`; the seven BN necks and seven classifiers sit downstream only on the ID-logit paths. History replay calls encoder/fusion directly and therefore has no missing trainable neck/classifier derivative. The fusion trainability assertion would reject an added trainable fusion tensor (`modeling/trifusion/signal_preserving_v8.py:63-77,264-339,391-464,483-517,545-612,614-653`; new runner `55-58`).

I also read the exact existing initialization-name inventory through its bound local prior-Q1 mirror, without loading weights: 203 trainable tensors comprise CNN42 + Transformer54 + Mamba93 encoder tensors and14 ID neck/classifier tensors. The three role groups exhaust all189 encoder tensors; no additional trainable metric-path tensor is omitted. The inherited inventories record8,076,300 trainable scalars in folds0/1 and8,102,412 in fold2. These inventories corroborate the unchanged builder; they are not a new model instantiation by this reviewer.

The actual update ordering is sound on inspected source:

1. Capture the current GradScaler scale; compute the FP32 history-leaf upstream.
2. Compute the direct single-group check, when applicable, without populating `.grad`.
3. `scaler.scale(loss).backward()` once at line146; clone current encoder gradients and discard the current loss/output references before history replay.
4. Replay each original B64 group with nonzero upstream. Assert selected coordinates equal the no-grad refreshed rows; place each unique selected record's upstream at its original last occurrence; scale the surrogate once and accumulate its parameter VJP in FP32.
5. Prove RNG/buffers unchanged and existing encoder `.grad` unchanged; add the scaled historical gradient only for `history_gradient`. Assert bitwise equality of the actual and expected applied gradient.
6. `scaler.unscale_(optimizer)` once at line198; check finite gradients; `scaler.step(optimizer)` once at line203, then update the scaler.

No history optimizer, second optimizer step, double unscale, extra loss coefficient, detached current-peer path or unwanted shared-parameter omission was found. The applied-gradient summaries concern the pre-AdamW input gradient; they are not measurements of the momentum/weight-decay-adjusted parameter displacement. The203/203 nonzero-gradient gate is cumulative over a fit, not a claim that every tensor is nonzero on every step (`tools/train_msvr_history_gradient.py:199-238`).

## D. Replay, direct proof and CPU verifier coverage

`ViewFields.capture` stores detached CPU frozen anchor/reference/Signal fields, original CPU/CUDA RNG at encoder entry, and the final position for each repeated record. Queue deduplication and exclusion ensure selected metadata indices are unique, so the assignment of coefficients at selected positions cannot overwrite two distinct retained records. Replay uses the original B64 shape and `fork_rng`, keeps the same train/eval roles and original checkpointing path, and bypasses all ID/BN necks. No extra mutable BN state exists in the inspected CNN/Transformer role code; Mamba production implementation/binary behavior remains a runtime dependency (`tools/msvr_freshness_probe.py:43-85`; `tools/probe_msvr_history_candidate_gradients.py:43-50`; `modeling/trifusion/experts/semantic_residual.py:28-129`; `modeling/trifusion/experts/mamba.py:29-97`).

History groups are processed in sorted original-step order. A group is skipped only if its entire saved leaf upstream has strict zero absolute sum, and actual group IDs/forward counts and per-record upstream norms are recorded. The CPU verifier reconstructs group support from recorded norms and checks exact group/count agreement; it does not regenerate the upstream vectors (`tools/train_msvr_history_gradient.py:154-172,207-218`; `tools/verify_msvr_history_gradient.py:85-120`).

The capacity direct proof uses the same current graph and one directly differentiable historical group. `direct_loss = loss + weight*(full-pooled)` replaces only that scalar's gradient while preserving the other13 terms, then compares all189 encoder gradients to `(current + weight*history_gradient)/scale`. The scale and weight are consistent and the relative-L2 tolerance is0.005. This direct proof is performed for both endpoints as a validation of the combined gradient even though the control discards V (`tools/train_msvr_history_gradient.py:127-142,182-188`). The new CPU verifier demands exactly one such check for every capacity arm, and none in Q1/overfit (`tools/verify_msvr_history_gradient.py:108-120`). It does not independently prove a direct full graph for later multiple-history-group batches or all203 optimizer tensors.

One concrete documentation mismatch was found and resolved during this review: the initial plan called that first historical capacity batch “step3”; the implementation and independent queue replay place it at **logged step4, zero_based_step=3** in every fold. The final plan now states both bases explicitly. The independent check proved that this wording replacement and its configured plan digest were the only changes, and all four execution/verifier scripts are byte-identical to the initial reviewed versions. Initial and final input snapshots and both static-check receipts are preserved.

The new CPU verifier covers all saved bytes in every training distance stream, exact offsets/shapes/finiteness, current and historical true-ID masks, historical age/order/identity/scene metadata, and every13 saved queue/distance statistic:11 integer statistics plus current/expanded Triplet scalars. It recomputes the fused Triplet and checks its selected component, then reconstructs the weighted total from the exact14 saved component scalars with the inherited weights. It also checks per-epoch means, role-summary norm identities, recorded applied-gradient equality and final compact checkpoint reconstruction (`tools/verify_msvr_history_gradient.py:17-134,156-178`).

**The precise 14-loss proof is weighted scalar reconstruction.** The other13 CE/Triplet values are not regenerated from saved logits and separate branch embeddings/distances, because those are not saved per step. This satisfies the plan's component-reassembly statement, but must not be reported as an independent model recomputation of all14 component losses. Raw per-step gradients, pixel/field equality, direct-gradient agreement and state preservation remain runtime witnesses, as the verifier's own scope states (`tools/verify_msvr_history_gradient.py:283-290`).

Strict reload reconstructs all baseline aliases against the exact B0 state, requires complete role-state key sets, performs `strict=True` state loading and hashes the reconstructed state. M0 additionally compares all five output tensors bitwise on eight source records. Q1 evaluates only after reloading the final checkpoint (`tools/train_msvr_instance_memory.py:94-111`; `tools/train_msvr310_source_style.py:164-174`; new runner `292-309`).

The complete Q1 CPU path recomputes all five feature-distance matrices, exact full rankings, each valid query's AP and first match, Rank1/5/10, pooled/fold/identity metrics, changes against Signal, both identity bootstrap lower bounds, and all ten candidate qualification checks. `fused_strictly_best` explicitly includes Signal. Final `next_phase_qualified` requires both paired and candidate-versus-Signal five-gate groups; the control's own Signal gates are also reported, but are not incorrectly made a prerequisite for candidate promotion (`tools/verify_msvr_history_gradient.py:179-281`; `tools/train_msvr310_source_style.py:196-221`; `tools/train_msvr310_trifusion_oof.py:244-285`). No best-epoch selection, reranking, official-test input or cross-fold feature-distance calculation was found.

## E. Full metadata replay, scope and resources

All780 source batch registry entries contain64 records in8 true-ID groups of8 and stay within their own source identity set. Across260 batches, each fold covers every one of its source records. An independent ordered-map replay verified latest-view retention, last B64 position selection, current-record exclusion, age1–8, chronological group binding and capacity512. Actual M0 capacity lists and fold0's fixed100-step overfit list were additionally replayed.

| Fold | Q1 history batches per endpoint | Historical candidate exposures per endpoint | Maximum selected history / post-update queue | First capacity history records |
|---|---:|---:|---:|---:|
| 0 | 194 | 58,133 | 356 / 391 | 31 |
| 1 | 194 | 59,505 | 372 / 408 | 32 |
| 2 | 194 | 59,984 | 362 / 406 | 39 |

Q1 first uses history at logged step67; each first capacity history is logged step4 and has one stored group. The overfit batch has zero selected historical records throughout, as registered. The queue's capacity eviction branch is not exercised by these actual schedules (maximum408<512); synthetic `check_math` includes a small-capacity example, but that author's synthetic code was inspected, not run here. These are source-label schedule calculations, not new image/model results.

The wrapper fixes the only stage sequence and stops on a nonzero child exit. M0 actual requirements are6×8 capacity +2×100 overfit =248 optimizer steps; Q1 is6×260 =1,560 updates and2,064 fold-gallery record forwards. It passes the actual M0 and CPU receipt digests into Q1, and checks terminal Q1 CPU status against the final summary SHA (`tools/run_msvr_history_gradient.py:14-65`; new runner `251-339`). Original learning rate3.5e-4, weight decay1e-4, 5-epoch warmup/cosine path, seed42, workers4,128×256 inputs and B64/K8 remain fixed. Both endpoints are freshly initialized and the runtime compares their full augmented-pixel SHA sequences.

Extra fresh reencoding and history-VJP record forward counts are separately saved and checked; direct and zero-update replay costs are included. The algorithm computes V in the control too, while selected group counts may differ after the arms' trajectories diverge. Whole-epoch and wrapper-stage timings and peak allocated/reserved GPU memory are available. Counts are full three-role encoder/fusion calls, not additional full Signal image forwards and not measured FLOPs. Internal checkpoint recomputation and backward cost are not represented by a forward count alone.

The wrapper requires3GiB free before launch; the plan estimates at most1.8GiB added storage and3–5 GPU hours on the existing remote RTX3090 environment, with no environment install. These are estimates and prelaunch requirements, not newly measured resources. I did not inspect live disk/GPU state or remote binary versions. The inherited environment field primarily identifies Signal commit/diff; it is not a complete Python/PyTorch/CUDA/Mamba binary certificate (`tools/run_msvr_history_gradient.py:16-22,39-45`; `tools/train_msvr310_signal_oof.py:31-65`; final plan `46-52`).

## Untested runtime assumptions and claim limits

- The remote raw-byte contract, external Signal imports/weights, dataset images and production Mamba/CUDA libraries must pass their actual checks. Five inherited local files use CRLF while pinned runtime files use LF; the local LF-normalized bytes and current Git HEAD blobs match those five digests. This audit did not rewrite or normalize executable files and did not live-rehash remote files.
- T0 synthetic scalar/current-gradient/VJP tests are correctly wired but have not been executed by this reviewer. Static compilation does not certify torch/NumPy/CUDA compatibility.
- Current and history encoder coordinates must be bitwise identical under the actual grad/no-grad/checkpoint/autocast paths; RNG/buffer preservation, all role witnesses, finite gradients, zero overflows, actual203/203 gradient coverage, direct relative error, and capacity must pass M0. Source-diagnostic success cannot substitute for these new-training checks.
- New-training applied-gradient summaries and direct proof have not been observed. The direct proof's scope is one historical group per capacity arm; grouped VJP correctness otherwise rests on the inspected chain-rule implementation and per-group runtime equality assertions.
- M0 overfit verifies only the current-path fixed-batch engineering behavior; the capacity stages establish the historical path. Saved14 component scalars enable total-loss reconstruction, while raw component model outputs and per-step parameter gradients remain unavailable for independent regeneration.
- Exact B0 feature/distance parity, paired training pixels, strict final reload, all rankings and two original gate groups remain mandatory. No Q1, generalization gain, causal explanation of old failures, novelty, independent-seed validation or SOTA claim is supported before those actual results exist. Even a future gate pass remains seed42-only on repeatedly used internal folds.

There is no outstanding requested code fix, fallback, compatibility layer, changed scientific threshold or unrelated refactor. The resolved step-label correction is documented above. Proceeding means running the already registered engineering sequence and honoring its failure stops, not treating this static WARN as a fabricated runtime PASS.

## Reproducible independent checks and exact final primary hashes

The independent script uses only Python's standard library. It performs AST parsing with Python3.10 grammar plus compilation on local Python3.13.12, never imports or executes the experiment. All29 inspected Python sources pass; all four new scripts pass. Among67 inherited project-hash checks,62 are exact local bytes and5 are the already described LF-bound/CRLF-local files; zero unexplained mismatches remain. The full74-entry manifest records each exact local input digest.

```powershell
& 'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -X utf8 'C:\Users\gb\.codex_tmp\history_gradient_training_preexecution_audit_20260908\independent_static_checks.py'
```

| Final inspected input | SHA256 |
|---|---|
| configs/MSVR310/TriFusion-history-gradient-paired-v1.json | `d03c7be1e738a206bf6db3b55580f53fbf33bfcc9050a46c4ddd0c81d7134c4e` |
| refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md | `152d7568bb985d5ccfb173d86374eec912e9500798488a9863f5a702f5e72be3` |
| refine-logs/msvr310_history_gradient_v1/EXPERIMENT_TRACKER.md | `69f946349430d66f2f1ef42afe5b09438e4174435ca84b0ec698feda6b5ed228` |
| tools/train_msvr_history_gradient.py | `087acdf4b8805f0783a460f5adee0e90227bf20342ae4eda6a033dc27b5869a4` |
| tools/verify_msvr_history_gradient.py | `bea8870dec7355921cf65ab61be68d73d6aab1773d4a60a457f785e91d35636f` |
| tools/check_msvr_history_gradient.py | `d2a650240cb4ec5a5cdcbc68224461a815eb4505f263939bca7ae90b3d0cded3` |
| tools/run_msvr_history_gradient.py | `b5412505122721c1ae7b60c27a472c14302b2923d5b6fceb8586c58489f1b593` |
| tools/probe_msvr_history_candidate_gradients.py | `cfa56255f6dd5a471cf3bfbfc99be3d0693f5c9c5dfc29bd8c6e4d15c79090de` |
| tools/msvr_freshness_probe.py | `4542785ace8307dc8a92325f7d52f2542e526f53ad8c3c2c0d9e91e48ed97886` |
| protocols/msvr310_train_oof_v1.json | `4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4` |

The initial plan/config hashes were `aca0920c5bb58e691da75b77e2548386f8fd79fb535c931bc5fe2cf022c3ca2d` / `11c5fb22d7a95d2915a501a85972db79e8620276944b3dc0c58695550c3f6a00`. Their initial snapshots are preserved. Independent checker final SHA256: `e5e2012a76994907afc872fb643c9ce05b62c1681128931270d6c79fb6d31a5f`.
