# Experiment audit — completed F1 foundation packages

Date: 2026-10-02  
Reviewer: gpt-6-astra, reasoning effort max; canonical agent ID /root/audit_foundation_complete739  
Review route: fresh agent, same-family, provisional  
Overall verdict: **WARN**  
Integrity status: **WARN**  
Evaluation type: **real_gt**

The six recorded endpoint results are internally consistent with the inspected training, save/load, and scoring paths. No generated evaluation ground truth, metric self-normalization, nonexistent result, or per-metric checkpoint cherry-picking was found. The evidence supports a qualified, single-seed comparison of two complete foundation packages on an already-consumed official benchmark.

The WARN is material: the paired-diagnostic helper was omitted from the F1 runtime source seal; its recovered source has an earlier matching hash but cannot retrospectively prove runtime immutability. Model/distance binaries and image bytes were not locally inspected, and this is neither an independent-seed study nor an untouched-test estimate. The MSVR310 progress gate fails.

## Evidence scope and assurance

The following aliases resolve every file:line reference in this report:

- S = C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002/source_intake737
- E = C:/Users/gb/.codex_tmp/foundation_recipe_complete_20261002
- R = E/raw/trained-model
- C = E/raw/logs/foundation_recipe_20261002_v1
- P = E/raw/results/foundation_recipe_complete_20261002

The JSON companion records absolute paths, observed hashes, verification counts, and detailed assurance fields. All 249 manifest source hashes, all 266 original source-intake catalog entries, and all 91 completed primary-text hashes matched locally. These are byte-integrity checks, not substitutes for semantic review. The 24 binary hashes were cross-referenced as remote receipt evidence; the binary files themselves were not available in this local intake.

Only local text/JSON reading, hashing, AST-free source inspection, and arithmetic reconciliation were performed. No repository/model import, scorer invocation, GPU use, SSH, installation, retraining, report rerun, or original-evidence edit occurred. The executor's RECORDED_TRAINING_ACTIVITY.json was not used as semantic evidence. The unresolved evidence_check.py precheck remains unavailable (E/EVIDENCE_PRECHECK_UNAVAILABLE.json:2–5); no substitute helper PASS is claimed.

This reviewer was started with fork_turns none, model gpt-6-astra, reasoning_effort max, as confirmed by the parent. Only the canonical agent ID was exposed by the spawn tool. The task prompt nevertheless included executor interpretations and expected metrics rather than only paths. The reviewer independently derived the reported numbers below from primary files, but the review was not blinded to those interpretations. It remains same-family/provisional, with this additional context limitation.

## A. Ground-truth provenance — PASS within the supplied record evidence

The active scorer takes identity, camera, and scene arrays directly from the fixed protocol's query/gallery records. It does not generate reference labels from model outputs (S/tools/run_correspondence_roles.py:79–108). Training labels come from the protocol training label map; evaluation uses original dataset identities (S/tools/official_three_dataset_data.py:8–18). The current CE/triplet and author CE/triplet objectives use those training labels (S/tools/run_foundation_recipe.py:155–162).

The fixed records agree with the pinned author filename parsing:

- RGBNT201: identity from the filename prefix; camera digit minus one (S/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:61–85).
- RGBNT100: identity/camera regex and camera minus one (S/comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py:63–84).
- MSVR310: identity, camera, and scene from the filename; scene is also the recorded view code (S/comparators/Signal-cd1b0a6/data/datasets/msvr310.py:67–87).

I independently parsed the filename-derived fields for all 27,266 train/query/gallery record entries and found no mismatches; every training label agrees with its identity-to-label map. Train/evaluation identity and path intersections are empty in all three protocols. Query/gallery overlap is expected by these retrieval protocols and is handled by the exclusion rules.

| Dataset | Train records / identities | Query records / identities | Gallery records / identities | Gallery-only distractor identities | Queries with valid positives |
|---|---:|---:|---:|---:|---:|
| RGBNT201 | 3,951 / 171 | 836 / 30 | 836 / 30 | 0 | 836 |
| RGBNT100 | 8,675 / 50 | 1,715 / 50 | 8,575 / 50 | 0 | 1,715 |
| MSVR310 | 1,032 / 155 | 591 / 52 | 1,055 / 155 | 103 | 591 |

Record/count anchors are S/logs/official_three_dataset_protocols_20260923/RGBNT201.json:184,78309; RGBNT100.json:63,218978; MSVR310.json:168,38538. These three exact files all retain the original inventory reference at line 4.

All fields other than dataset_root are equal to the respective S/.aris/compute/source2026_protocols originals. The original root /data/gaob/Re-ID/dataset/... changes to /data2/gb/Re-ID/dataset/...; record ordering and labels do not change. The queue's explicit root-only check is S/tools/queue_foundation_recipe.py:27–45. This establishes record-list equivalence, not independent verification of image bytes on the two machines. The referenced original inventory and image contents were not re-created from disk in this audit.

MSVR310 removes **same identity AND same scene/time code**, not same camera; different identities remain, including the 103 distractor identities. The other two datasets remove **same identity AND same camera**. The actual author scorer implements these at S/comparators/Signal-cd1b0a6/utils/metrics.py:68 and :137. The separate scene_scores/camera_scores implementations use the same masks at S/tools/train_msvr310_signal_oof.py:223–238 and S/tools/train_rgbnt100_signal_oof.py:253–268. The stale camera-language docstring on the MSVR scorer does not describe its executed line 68.

## B. Score normalization — PASS

The retrieval path applies per-feature L2 normalization, computes squared Euclidean distances, and ranks the full gallery. This is feature geometry, not rescaling a reported metric by prediction extrema (S/tools/run_foundation_recipe.py:49–53; S/tools/run_official_three_dataset_roles.py:230–238).

AP divides precision-at-positive-rank contributions by the number of dataset positives; mAP averages query AP. CMC divides hit counts by valid query count. The author implementation is S/comparators/Signal-cd1b0a6/utils/metrics.py:82–108 and :142–170. The separately written scorers assert at least one positive for every query and compute the same measures (the scorer references under A). The final factor of 100 is a fixed percentage conversion (S/tools/run_correspondence_roles.py:100–102).

There is no division of mAP/CMC by this model's maximum, minimum, or mean prediction. The high RGBNT100 Rank-1 values are dataset hit rates; they are not suspicious self-normalized scores. Training-time normalization differs across the two packages: author triplet uses unnormalized features by default (S/comparators/Signal-cd1b0a6/layers/triplet_loss.py:121–135), whereas current triplet consumes the normalized fused vector (S/tools/run_foundation_recipe.py:50,160–162). This difference is one of several simultaneous recipe changes.

## C. Result existence, values, execution, and selection — PASS for recorded primary evidence

Every result exists and agrees with its selected history row, accepted-matrix row, and full report row. The primary result anchors are the following exact files, each at lines 3–8 (status/seed/epochs/selection), 16–24 (bindings/metrics), and 26–30 (evaluation flags/completion):

- R/foundation_recipe_20261002_v1_full_author_RGBNT201/official_metrics.json
- R/foundation_recipe_20261002_v1_full_author_RGBNT100/official_metrics.json
- R/foundation_recipe_20261002_v1_full_author_MSVR310/official_metrics.json
- R/foundation_recipe_20261002_v1_full_current_RGBNT201/official_metrics.json
- R/foundation_recipe_20261002_v1_full_current_RGBNT100/official_metrics.json
- R/foundation_recipe_20261002_v1_full_current_MSVR310/official_metrics.json

| Dataset | Package | Selected epoch | mAP (%) | Rank-1 (%) | Logged updates |
|---|---|---:|---:|---:|---:|
| RGBNT201 | author | 27 | 73.47275487487326 | 77.15311050415039 | 2,649 |
| RGBNT100 | author | 9 | 84.03193833734271 | 96.20991349220276 | 3,129 |
| MSVR310 | author | 38 | 50.83882666558341 | 68.69712471961975 | 706 |
| RGBNT201 | current | 26 | 62.580551571472775 | 62.44019269943237 | 2,649 |
| RGBNT100 | current | 12 | 77.58429184381957 | 94.1690981388092 | 6,559 |
| MSVR310 | current | 24 | 50.35673958098721 | 67.51269102096558 | 1,000 |

Each full training.json contains exactly epochs 1–50, and every epoch's stdout record equals its corresponding history row. Every logged batch index forms its epoch's consecutive 0-based sequence, losses and logged LRs are finite, epoch step counts match JSONL counts, and arithmetic mean logged loss matches mean_loss. Total: 300 epochs and 16,692 full-run updates. The six M0s add 48 separate engineering-probe updates; they are excluded from the formal total. The full histories begin at line 38 of each training.json, and their completion/best fields are at lines 642–647. The exact 16,692 step rows and their hashes are preserved in the JSON companion.

Selection uses >= on mAP, saving later ties, and strict evaluation independently selects max(mAP, epoch). These six realized maxima have no ties. The evaluator loads exactly that checkpoint, checks its stored epoch and four metrics, and recomputes all four under the same full-gallery scorer within 1e-5 percentage points (S/tools/run_foundation_recipe.py:248–255,287–312). Rank-1/5/10 are not separately maximized. Rank-5/10 for RGBNT100/MSVR310 exist in JSON; the Markdown table deliberately prints dashes for them (S/tools/report_foundation_recipe.py:86–89).

The 18 actual child command records comprise six M0 commands and six train/evaluate pairs, all calling run_foundation_recipe.py with the correct dataset, recipe, initialization, seed42 and epochs50. Each child exits 0. Example exact command/exit anchors: C/foundation_recipe_20261002_v1_full_author_RGBNT201/campaign.json:15,41,49,75; all other corresponding child files were checked. All six M0 completion times precede full-run starts.

The parent reports one report invocation and exit0 at C/campaign.json:925–930; C/report.log:1 contains the completed six-endpoint report. The report creation time follows all six full endpoints. The queue performs acceptance before the report call (S/tools/queue_foundation_recipe.py:188–208), and the report independently requires the complete 12-job matrix (S/tools/report_foundation_recipe.py:25–45). P/SUMMARY.json:7–8 agrees with the independently counted totals. These observations support one recorded invocation in this campaign, not a claim about unrecorded activity elsewhere.

Initialization and state checks:

- Clean construction sets USE_A/USE_B false, builds from the public CLIP file, checks all 152 public visual state entries after prescribed positional interpolation, and enables fresh camera/visual updates (S/tools/run_clean_clip_joint.py:37–99). The recorded public hash matches the pinned public CLIP URL's hash at S/comparators/Signal-cd1b0a6/modeling/clip/clip.py:35.
- PlainFoundation retains only the Signal backbone and the recipe's heads; the temporary role/adapter wrappers are discarded. Its forward calls Signal directly; no role hook/adapter forward is used (S/tools/run_foundation_recipe.py:25–53,62–100). The Signal no-module retrieval output is concatenated RGB/NI/TI global features (S/comparators/Signal-cd1b0a6/modeling/make_model.py:197–226).
- The shared visual/camera/current-head witnesses match author/current within each dataset. Full model-state digests differ for all three pairs. Author's current_head_initial_sha256 witnesses the temporary current head, which author does not retain; it is not proof that the actual author/current heads or full models are identical. See C/initialization/{RGBNT201,RGBNT100,MSVR310}_{author,current}.json:12–23 and the construction code above.
- Each M0 records 8 steps, all expected trainable tensors receiving nonzero gradients (155 for all current and author RGBNT201; 159 for author RGBNT100/MSVR310), and reload_max_abs_difference 0.0. M0 training.json:48–52 records these values. The code saves the entire model state and performs a strict reload plus a forward equality check (S/tools/run_foundation_recipe.py:174–186,263–278).
- Formal runs rebuild from the original initialization, never load M0 weights, assert optimizer coverage and finite/non-skipped AMP updates, and check frozen parameters unchanged plus changed visual/camera parameters (S/tools/run_foundation_recipe.py:189–241,259–284). This is executed-source/receipt assurance, not a direct local tensor inspection.

Remote checkpoint/distance hashes agree across intake, result, and accepted-matrix receipts. Neither binary tensor contents nor full optimizer states were locally inspected. Checkpoints save model state rather than optimizer/scaler state; no resumable-training guarantee is made. Training-best and final distance hashes are both recorded, but the code asserts scalar metric parity, not elementwise equality between those two distance files. Different serialized file hashes do not themselves establish a numerical mismatch or equality.

## D. Active metric path and source closure — WARN

The active call chain is concrete:

run_foundation_recipe -> run_clean_clip_joint.control.runner -> run_visual_update_control.runner -> run_correspondence_roles.official_metrics.

The imports are S/tools/run_foundation_recipe.py:18–20, S/tools/run_clean_clip_joint.py:14, and S/tools/run_visual_update_control.py:22. Both per-epoch and strict evaluation call that same official_metrics (S/tools/run_foundation_recipe.py:250,300).

Feature extraction iterates the full ordered protocol records, uses model.eval() and inference_mode(), and asserts the exact query/gallery count and 1536-dimensional output (S/tools/run_correspondence_roles.py:60–76). The author scorer's resolved file path is checked, and its output is compared with the separate local scorer on every call (:79–108). No reranking or test-time optimizer is called. The upstream module imports a reranking function, but F1 calls eval_func/eval_func_msrv directly; the optional wrapper classes and unrelated old evaluation functions are not evidence for F1 results.

**Runtime closure gap:** S/tools/report_foundation_recipe.py:13 imports tools.analyze_correspondence_distances; its compare function is called at :54. That helper is absent from the original 249-source manifest and the original source intake. This was found by inspecting the actual import and catalogs, not inferred from a queue status.

The executor subsequently supplied E/diagnostic_source_closure/tools/analyze_correspondence_distances.py. Its observed SHA256 is 833ebeb47cb5840422710fc00df1a974338cb04078249720d867f18b945f997a. I independently confirmed equality with both the source-file hash and earlier waiter analyzer hash in E/diagnostic_source_closure/prior734_collection.json:1, whose collection timestamp is 17:52:32, before F1. E/diagnostic_source_closure/SOURCE_CLOSURE.json:2–12 also records the current observation, commit comparison, and the unchanged original-manifest omission. The commit comparison there is a receipt, not a Git-history inspection by this reviewer.

The recovered helper reads saved distances, re-scores with the same camera/scene functions, checks score parity, compares the six GT arrays across the pair, and computes paired changes (:28–60). Its bootstrap resamples 2,000 sets of fixed-model identity mean AP changes with RNG42 (:61–75). I checked the reported identity counts, weighted/macro mean arithmetic, and repairs-minus-new-errors identity against the summary. I did not execute that helper, read remote arrays, or recreate its bootstrap distribution.

Therefore no dead/phantom endpoint metric path was found, but the F1 runtime binding of the paired-diagnostic helper remains unproven. The earlier-and-later matching source evidence narrows the uncertainty; it does not retroactively seal F1. Per-query diagnostics and bootstrap intervals retain WARN assurance.

## E. Scope and comparison interpretation — WARN

The actual study is two packages × three datasets × one seed, with 50 epochs for each endpoint. It contains six separately trained endpoints, not six independent seed replicates. Evaluation labels are used every epoch for checkpoint selection, and the report explicitly acknowledges consumed benchmark development (P/REPORT.md:12–17; S/tools/run_foundation_recipe.py:248–255). Gradient training uses only training records, but there is no untouched-test generalization guarantee.

The packages differ in heads, raw versus normalized training features, loss weighting and triplet form, optimizer, LR schedule, batch/identity sampling settings, and augmentation. The active implementation is S/tools/run_foundation_recipe.py:25–53,110–162,213–218:

- Author uses the pinned Signal heads, summed per-head 0.25 smoothed CE plus soft-margin triplet, Adam, its dataset-specific LR grouping and scheduler. RGBNT201/100 use the author's noisy cosine scheduler; MSVR310 uses the 20/40 schedule and classifier LR multiplier. Sources: S/comparators/Signal-cd1b0a6/layers/make_loss.py:13–56; layers/triplet_loss.py:113–135; solver/make_optimizer.py:4–43; solver/scheduler_factory.py:7–30; solver/scheduler.py:67–104; solver/lr_scheduler310.py:43–56.
- Current uses smoothed CE plus margin0.3 triplet, AdamW, visual base LR5e-6 and other trainable base LR3.5e-4, five-epoch warmup/cosine, B64/K8. Author B/K is 64/8, 128/16, and 64/4 respectively. All use workers4 and AMP initial scale256.
- Author augmentation is independently sampled per modality because ImageDataset calls the transform separately on each image (S/comparators/Signal-cd1b0a6/data/datasets/bases.py:98–107). Current shares crop/flip geometry and uses independent erasing (S/modeling/trifusion/aligned_data.py:26–73; S/tools/train_rgbnt100_signal_oof.py:113–123; S/tools/train_msvr310_signal_oof.py:89–99).
- RGBNT100 author's native MAX_EPOCHS30 is extended to50 (S/comparators/Signal-cd1b0a6/configs/RGBNT100/Signal.yml:32–38; S/tools/run_foundation_recipe.py:70–78). Common evaluation loaders and execution settings also differ from the original author entry. Exact author reproduction is not established.
- The current training.json cfg_yaml still contains inherited Adam/NO_MARGIN/author schedule fields, while the current branch directly implements AdamW/margin0.3/current schedule. The source and logged LRs resolve this; cfg_yaml alone is not a complete effective-recipe specification (each full training.json:640; S/tools/run_foundation_recipe.py:148–162,213–218).

Equal epoch budgets are not equal update budgets: the actual counts differ substantially, especially RGBNT100 and MSVR310. The RandomIdentitySampler may repeat short identities and discard incomplete identity groups; 50 logged epochs are not a claim of one unique pass over every training image per epoch (S/comparators/Signal-cd1b0a6/data/datasets/sampler.py:18–68). The report's elapsed training interval includes epoch evaluation and checkpoint overhead; it excludes construction, M0 and final strict evaluation (S/tools/report_foundation_recipe.py:40–51; P/REPORT.md:16). Per-epoch seconds measure the training portion before evaluation; they should not be mistaken for the complete run interval.

The historical clean-public shared-global summary's hash matches its pinned report reference. I compared its actual initial public visual, camera, and current-head witnesses to F1 current and found equality for all three datasets. Relevant historical anchors: S/logs/clean_clip_complete721_20261002/raw/results/clean_clip_joint_complete_20261002/SUMMARY.json:36–48,109–121,182–194; S/tools/report_foundation_recipe.py:59–69. This is a historical adapter control with different capacity and execution history, not a concurrent capacity-matched arm or isolated adapter causal effect.

The progress gate is delta mAP >=0.5 percentage points AND delta Rank-1 >=0 (S/tools/report_foundation_recipe.py:53–58):

| Dataset | Author minus current mAP, percentage points | Gate |
|---|---:|---|
| RGBNT201 | 10.892203303400485 | PASS |
| RGBNT100 | 6.447646493523138 | PASS |
| MSVR310 | 0.482087084596202 | **FAIL** |

Rank-1 differences are positive for all three. The independently count-based paired diagnostic CMC differs from float32 upstream CMC by small rounding amounts within the enforced 1e-5-point tolerance; this is not an inconsistent selected checkpoint. The MSVR310 identity-macro delta is 0.5284354345205875, a different estimand from official query-mean mAP; it must not replace 0.482087084596202 in the gate. Neither rounding nor switching to that macro delta makes the gate pass (P/SUMMARY.json:4447,4483; recovered helper:55–73).

The bootstrap intervals describe these fixed selected models and fixed identities. They do not represent training-seed variation, correct for epoch/method selection, or support unbiased test significance. No claim of a new architecture, isolated normalization cause, robustness across seeds, SOTA, or completion of the overall objective follows.

## F. Evaluation type — PASS: real_gt

All six endpoint retrieval evaluations are real_gt: positives and exclusions use dataset identity and environment fields, and model outputs supply feature distances only (S/tools/run_correspondence_roles.py:85–102; scorer references under A/B).

The temporary constructor helper name _build_signal_teacher does not introduce teacher-generated evaluation targets; its active function only constructs the pinned Signal model (S/tools/build_v12_complete_path_oof_targets.py:244–258). The unrelated generated-target/OOF routines in the imported source module are not called by F1. M0 is an engineering gradient/save-load check, not an accuracy endpoint. Paired bootstrap is post-selection diagnosis, not synthetic GT or an additional experiment.

## Blockers and action items

1. Preserve the diagnostic-helper source-closure WARN. Keep the recovered-source and earlier-hash evidence separate; do not rewrite the completed manifest. For a future campaign, include the report helper in the pre-execution source seal.
2. Retain the remote-tensor limit in any audit claim. Stronger assurance about exact checkpoint contents, saved GT arrays, or image migration bytes requires inspecting those actual assets or a separately preserved complete tensor/asset inspection. This review did not do so; no training/scoring replay is required to state the bounded results.
3. Describe this as a seed42 whole-package comparison on a consumed benchmark. Broader seed/generalization/significance claims remain unsupported. A fixed-identity bootstrap cannot fill those gaps.
4. Keep MSVR310's progress gate FAIL. Use official query-mean mAP, not rounded values or the identity-macro change.
5. In downstream documentation, specify current's effective optimizer/loss/schedule from the active branch and logged LR evidence; do not treat inherited cfg_yaml fields as its complete effective recipe.
6. Keep evidence_check.py marked unavailable and this review same-family/provisional. Do not convert either the source-hash checks or queue's VERIFIED_COMPLETE into external semantic acceptance.

These are assurance and claim ceilings, not a finding that the six recorded endpoint numbers are fabricated. No unqualified all-dataset progress, causal architectural, or unbiased-test claim is approved.

## Bounded supported claims

- Supported: six separately trained public-CLIP/fresh-camera, no-role/no-adapter endpoints completed seed42/full50 with the recorded full protocol galleries and camera/scene filters; primary histories and reports reconcile to 300 epochs and 16,692 formal updates.
- Supported with the stated execution/remote-binary limits: each endpoint reports the four metrics from its one mAP-best checkpoint, with later-tie selection policy and strict reload/re-evaluation evidence.
- Supported: the author package has higher selected mAP and Rank-1 than current on each of these three recorded comparisons. Progress gates pass for RGBNT201/RGBNT100 and fail for MSVR310.
- Supported as a limited witness: initial public visual/camera/current-head hashes agree within each dataset and with the historical shared-global reference; full author/current models are not identical.
- Needs qualifiers: historical shared-global comparisons, per-query paired diagnostics and fixed-identity bootstrap.
- Unsupported: isolated normalization/head/optimizer causality, new CNN/Transformer/Mamba contribution, exact author reproduction, seed robustness, untouched-test generalization, SOTA, or overall-goal completion.

Audit artifacts: this report and EXPERIMENT_AUDIT.json, written only in E/reviewer_foundation739/.

