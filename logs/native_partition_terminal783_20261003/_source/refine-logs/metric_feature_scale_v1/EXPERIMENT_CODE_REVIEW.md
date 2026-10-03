# F3 metric feature-scale source review

**Verdict: PASS — SOURCE_ONLY. No blocking or nonblocking source defects found; no patch requested.**

Reviewer: `gpt-6-astra`, reasoning `max`, fresh context `fork_turns=none`; native task `/root/review_metric_feature_scale_source755`. Review independence is **same-family** and acceptance is **provisional**. No external reviewer backend was called.

This review clears the fresh source-review gate only. It does not attest to a new launch, numerical parity, M0 passage, complete training, statistical validity or efficacy.

## Scope and evidence

`NEW` means `C:/Users/gb/.trifusion_github_publish_22c3bee`.
`FROZEN` means `C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source`.

Read all four new Python entries and the plan directly, then followed their active initializer, configuration, queue, training, loader, loss, evaluator, scorer and analyzer paths in the frozen intake. All four new entries parsed as Python AST. Independent stdlib metadata arithmetic checked complete protocol counts, identity splits, label maps and filename camera/time fields. All 259 frozen dependency files matched the original intake and F2 manifest.

No project/Torch import, model/optimizer/GPU work, SSH, evaluation/scoring/report execution or package installation was performed. Only this Markdown and its JSON companion were written; source, plan, original records and trace were not edited.

## Findings

- Blocking: none.
- Nonblocking source defects: none.
- Minimal fixes: none justified by the inspected source.

## Source conclusions
**C1 — PASS_SOURCE.** Only the Triplet input scale differs between the fresh arms. Both share one raw Signal forward; CE receives F.normalize(raw) through the same neck/classifier, and deployment returns that normalized embedding. Raw and normalized training tensors are attached to autograd; detach is used only for norm diagnostics.

References: `NEW/tools/run_metric_feature_scale.py:18-61`; `FROZEN/tools/run_foundation_recipe.py:155-162`; `FROZEN/tools/run_training_feature_scale.py:75-81`; `FROZEN/comparators/Signal-cd1b0a6/modeling/make_model.py:197-228`; `FROZEN/modeling/trifusion/criterion.py:17-34`.

**C2 — PASS_SOURCE.** The trainer captures the original F1 build/loader/loss functions when F2 is imported; F2.configure is not run on import. F3 then rebinds foundation schema/recipes/build/loader/loss. F3 build and loader explicitly use recipe=current for construction and sampler; inherited optimization/loss choose the non-author branch for both normalized and metric_raw. There is no F3-to-F3 recursion or accidental author recipe.

References: `FROZEN/tools/run_training_feature_scale.py:13-19`; `FROZEN/tools/run_training_feature_scale.py:84-94`; `NEW/tools/run_metric_feature_scale.py:36-74`; `FROZEN/tools/run_foundation_recipe.py:103-162`; `FROZEN/tools/run_foundation_recipe.py:316-337`.

**C3 — PASS_SOURCE.** Queue imports leave F1 globals intact until F3.configure binds base schema/recipes/source_map/command/start_command/coordinate. The reused base.main/worker/verify functions resolve those globals dynamically. F3 owns run_phase and its worker command, so no mutation of the unused lower queue.run_phase callbacks is needed. Preparation, worker training/evaluation, pair check and the sole report point to F3 entries.

References: `NEW/tools/queue_metric_feature_scale.py:26-43`; `NEW/tools/queue_metric_feature_scale.py:46-101`; `NEW/tools/queue_metric_feature_scale.py:104-163`; `FROZEN/tools/queue_foundation_recipe.py:57-155`; `FROZEN/tools/queue_foundation_recipe.py:212-231`.

**C4 — PASS_SOURCE.** Effective fixed recipe remains CE coefficient 1, smoothing 0.1; Euclidean batch-hard Triplet coefficient 1, margin 0.3; B64/K8 and four workers; AdamW weight decay 1e-4; visual base LR 5e-6, other trainable base LR 3.5e-4; the inherited five-epoch warmup/cosine multiplier; seed42, full50 and AMP initial scale256. The AdamW default LR is overridden from base_lr before every update. YAML author defaults do not replace the active current-package path.

References: `FROZEN/tools/run_foundation_recipe.py:62-100`; `FROZEN/tools/run_foundation_recipe.py:110-162`; `FROZEN/tools/run_foundation_recipe.py:189-244`; `FROZEN/tools/run_signal_preserving_v5.py:1621-1627`; `FROZEN/tools/official_three_dataset_data.py:21-39`; `FROZEN/tools/train_rgbnt100_signal_oof.py:113-123`; `FROZEN/tools/train_msvr310_signal_oof.py:89-99`; `FROZEN/comparators/Signal-cd1b0a6/data/datasets/sampler.py:18-68`.

**C5 — PASS_SOURCE.** Public visual tensor equality and fresh camera/head construction are retained. F3 preserves model-state names and verifies its full initial-state digest. Six prepared witnesses and three two-real-train-record pair checks require equal fresh states, equal CE logits/deployment and an active raw-vs-normalized metric change. The historical F2 state equality is additional context. Initial pair checks are eval-only and are correctly labelled as such.

References: `FROZEN/tools/run_clean_clip_joint.py:37-107`; `FROZEN/tools/run_foundation_recipe.py:62-100`; `NEW/tools/run_metric_feature_scale.py:36-46`; `NEW/tools/check_metric_feature_scale_pair.py:20-65`; `NEW/tools/queue_metric_feature_scale.py:111-135`.

**C6 — PASS_SOURCE.** Each M0 makes exactly eight optimizer updates, rejects nonfinite loss/gradients and AMP skips, requires every trainable parameter to have nonzero gradient at least once, exact optimizer-parameter coverage, unchanged frozen parameters and changed visual/camera parameters. Strict state loading and unchanged 1e-5 gates remain. All six M0s are verified before any full phase; formal outputs are new directories built from the public initializer, never the M0 state.

References: `FROZEN/tools/run_foundation_recipe.py:174-284`; `FROZEN/tools/queue_foundation_recipe.py:66-107`; `FROZEN/tools/queue_foundation_recipe.py:118-148`; `NEW/tools/queue_metric_feature_scale.py:130-149`.

**C7 — PASS_SOURCE.** The controller requires the registered /data/gaob/Re-ID/Trifusion root, a completed F2 predecessor/report and no predecessor controller PID. CLI restricts physical GPU indices to 0-3; startup checks the preparation device. Scheduling uses current nvidia-smi memory<500 MiB, one active job per physical device and >=10 GiB free disk per launch; workers set CUDA_VISIBLE_DEVICES. Polling is 240 seconds. Failures stop new launches while preserving logs/artifacts and allowing active children to finish; there are no retries or preemptions. The source is consistent with the existing 2026 host binding, not a fresh remote-capacity observation.

References: `NEW/tools/queue_metric_feature_scale.py:46-91`; `NEW/tools/queue_metric_feature_scale.py:104-118`; `FROZEN/tools/queue_foundation_recipe.py:118-148`; `FROZEN/tools/queue_foundation_recipe.py:212-231`; `NEW/.aris/compute/tri_reid_four_gpu.md:1-9`.

**C8 — PASS_SOURCE.** F3 source_map preserves the original 259-entry F2 seal and adds its five protected inputs plus the review Markdown; it does not rewrite F1/F2 seals. Preparation takes and rechecks the source snapshot, and each worker mode and terminal report call require_sources, including initialization/pair witness hashes. BatchOrderLoader logs the actual consumed labels/cameras/paths after each processed yield; the M0 ninth fetched batch is not recorded as an update. The final report aligns all step/epoch indices and requires fresh-pair and historical F2 batch-order bytes to match.

References: `NEW/tools/queue_metric_feature_scale.py:19-29`; `NEW/tools/queue_metric_feature_scale.py:111-135`; `FROZEN/tools/queue_training_feature_scale.py:102-120`; `FROZEN/tools/queue_foundation_recipe.py:110-135`; `FROZEN/tools/run_training_feature_scale.py:50-65`; `NEW/tools/report_metric_feature_scale.py:21-59`.

**C9 — PASS_SOURCE.** Evaluation extracts every protocol query/gallery record and asserts the full (count,1536) shapes. Identity labels are actual protocol ground truth. RGBNT removes same identity AND same camera; MSVR removes same identity AND the filename time/scene group. Different identities and gallery-only identities remain. The saved-distance analyzer uses the same filtering and asserts every metric against the chosen receipt within 1e-5. Percent multiplication is metric units, not performance scaling; there is no query subset, pseudo-GT, target-derived training or reranking in the active path.

References: `FROZEN/tools/run_correspondence_roles.py:60-108`; `FROZEN/tools/run_official_three_dataset_roles.py:58-68`; `FROZEN/tools/run_official_three_dataset_roles.py:230-238`; `FROZEN/tools/official_three_dataset_data.py:8-39`; `FROZEN/tools/train_rgbnt100_signal_oof.py:253-268`; `FROZEN/tools/train_msvr310_signal_oof.py:223-238`; `FROZEN/comparators/Signal-cd1b0a6/utils/metrics.py:13-170`; `FROZEN/tools/analyze_correspondence_distances.py:28-75`.

**C10 — PASS_SOURCE.** Full training always completes epochs1-50. The mAP-best checkpoint uses the latest epoch for an exact mAP tie, consistently in saving, reload and queue verification. All reported ranks/mAP come from that single checkpoint. Six strict full completions precede accepted_matrix and the one CPU-only report invocation. The report retains all histories, final-minus-best metrics, norms, Triplet support, time/memory, repairs/errors and identity-level diagnosis. The per-dataset progress gate uses unrounded hybrid-minus-fresh-control mAP>=0.5 percentage points and Rank-1>=0.

References: `FROZEN/tools/run_foundation_recipe.py:210-313`; `FROZEN/tools/queue_foundation_recipe.py:84-107`; `NEW/tools/queue_metric_feature_scale.py:138-158`; `NEW/tools/report_metric_feature_scale.py:23-70`; `NEW/refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md:20-47`.

## Independent protocol metadata checks

| Dataset | Train / Query / Gallery | Train / Query / Gallery IDs | Minimum eligible positives | Gallery-only IDs / records |
|---|---:|---:|---:|---:|
| RGBNT201 | 3951 / 836 / 836 | 171 / 30 / 30 | 7 | 0 / 0 |
| RGBNT100 | 8675 / 1715 / 8575 | 50 / 50 / 50 | 50 | 0 / 0 |
| MSVR310 | 1032 / 591 / 1055 | 155 / 52 / 155 | 1 | 103 / 464 |

Every saved record's identity and camera/time field matches its filename. Training and test identity sets are disjoint; training label maps are contiguous and match all training rows. Every query has an eligible positive. The old and new-root protocol contents are identical except for `dataset_root`. This was metadata inspection, not model evaluation or scoring.

The historical F2 manifest matches the same 259-entry intake. Its local campaign record says `COMPLETE`, one report invocation and report exit0. Normalized historical batch logs contain 2,649 / 6,559 / 1,000 rows for RGBNT201 / RGBNT100 / MSVR310. They are context for future byte comparisons; they do not replace new controls or prove seed replication. Current remote controller liveness was not checked.

## Limits and next gate

- Static reasoning does not prove the new forward pair, numerical gradients, strict reload, all50 completion or batch-order equality will pass on the runtime. The registered checks must actually execute.
- Two-record initial pair checks establish only the stated eval-mode witness; they do not establish training-mode or whole-dataset functional identity.
- Logged batch-order equality covers recorded labels, cameras and paths, not independently replayed augmented tensors.
- Historical F2 files were read locally; this review did not deserialize checkpoints/arrays or independently verify current remote binaries, live processes, GPUs or disk.
- Official-set checkpoint/method selection and single seed42 remain consumed-benchmark development limitations. Identity bootstrap is fixed-model identity uncertainty, not training-seed uncertainty.
- The preserved strict 1e-5 numeric gates must remain unchanged if later runtime evidence fails.

No source fix is requested. Continue only the already registered exact source sync/resource checks, fresh preparation/pair witnesses and six production-data M0s; full50 remains conditional on all six actual M0 passes. Preserve any failure and diagnose its actual artifact before a separately reviewed fix.

## Protected-input integrity

All five protected inputs were SHA256-read before review and re-read at report creation; their exact values are below. A post-write read also confirmed no change. All 259 frozen files still matched the unchanged intake (`e446b628da46959c939ab5e7de9a30801b6dd6bfd5f053595d439fbb082f8ee0`). The JSON contains the complete before/after verification fields and source references.

| Protected input (under NEW) | Bytes | SHA256 before = after |
|---|---:|---|
| tools/run_metric_feature_scale.py | 2851 | `4bc89febd69f75f8a4b97d6445a9100bac73db6600fe8bb3df8401973fd1f593` |
| tools/queue_metric_feature_scale.py | 8516 | `ff19f479fba2da677298456ba850254ddd40d6882375d63d7c0fa6d98df05e1c` |
| tools/check_metric_feature_scale_pair.py | 4081 | `8fbcfba96e3a1d8810412dc100fb63679542d69bf44a589d40b0297e963c4cf0` |
| tools/report_metric_feature_scale.py | 5034 | `181cda28e9a18197a6b17350f6e07743602ec299401ceee0b3838938f5f200c8` |
| refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md | 5753 | `a2706c9306b0bd5fe5797e3da3198bb852ff5ba15d0abf81cda47c3c757bd53d` |
