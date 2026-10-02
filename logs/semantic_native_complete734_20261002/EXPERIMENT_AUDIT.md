# Experiment Audit Report

Date: 2026-10-02. Auditor: gpt-6-astra, reasoning max; fresh delegated read-only reviewer `/root/audit_ev1_complete734`. Review independence: **same-family**. Acceptance status: **provisional**.

Project: TriFusion EV1 semantic/native evidence, six completed endpoints.

**Overall verdict: WARN. Integrity status: warn.** No supported instance of model-generated GT, prediction-based metric inflation, fabricated reported numbers, or a claimed metric with an uncalled implementation was found in the inspected evidence. This is a qualified source/text audit, not a fresh verification of the remote image, checkpoint, or distance bytes. The scientific outcome is separately **EV-A FAIL, EV-B FAIL**; an integrity audit does not turn those failures into a positive result.

All `R/` references below resolve to `C:/Users/gb/.codex_tmp/semantic_native_complete734_20261002/raw/`. This mirrors `/data/gaob/Re-ID/Trifusion/` without changing paths in the evidence. `T/` resolves to this audit's `.aris/traces/experiment-audit/2026-10-02_run01/`. Line numbers refer to the original files, not reserialized JSON.

| Check | Status | Conclusion |
|---|---|---|
| A. Ground-truth provenance | PASS | Dataset-record identity/environment labels; no prediction-derived reference in the called path. |
| B. Score normalization | PASS | AP/CMC use GT/rank/query denominators; feature normalization is not metric inflation. |
| C. Files, numbers, status | WARN | All available text/hash/arithmetic checks pass; old tracker is explicitly historical and raw binaries are not local. |
| D. Called metrics/dead code | PASS | Claimed scores trace through invoked epoch/final/CPU-report paths and actual stdout. |
| E. Scope, selection, resources | WARN | One seed, official-set selection, added capacity, fixed historical controls; broader causal/generalization claims remain unsupported. |
| F. Evaluation classification | PASS | Formal retrieval is `real_gt`; fixed-model diagnosis and engineering probes have narrower roles. |

## Evidence actually checked

The reviewer used only file reads, hashes, JSON parsing and independent arithmetic in the Python standard library. No project module was imported, and no model, scorer, evaluation, report generator, training, GPU, or remote command was invoked. Only this audit and its trace were written.

- **337/337** collected files, **14,247,648 bytes**, match `collection.json:1`; **243/243** manifest source/config/protocol hashes match `R/logs/semantic_native_evidence_20261002_v1/manifest.json:54`. Hash agreement binds the mirror to the supplied collection; it is not independent attestation of remote execution.
- **60/60 available inputs** in the report's 75-file hash map match; the other **15** are saved distance tensors, absent locally. The PNG/SVG figure hashes also match. Input map: `R/results/semantic_native_evidence_complete_20261002/SUMMARY.json:4367`.
- All six accepted rows match their child verification, parent result, initialization binding, final metrics and source/protocol hashes. All **300** epoch records, **20,416** finite training-step records, batch indices, epoch step counts and mean losses agree. All **300 stdout epoch rows** and all six final stdout metric objects agree with the receipts. Six M0 receipts contain **8 batches each**, totaling **48**; their reported gradient coverage and reload thresholds pass.
- All **12** comparisons agree with available endpoint receipts. Delta arithmetic, net Rank-1 repair counts, identity query counts, weighted identity AP differences, macro means and gate predicates agree. This does **not** independently establish the per-query distances, individual repair memberships or bootstrap interval endpoints.

Reproducible audit checks and complete results are `T/verify_evidence.py`, `T/deterministic_checks.json`, `T/verify_execution_logs.py`, and `T/execution_log_checks.json`.

## A. Ground-truth provenance — PASS

The active path loads protocol records and preserves dataset identity labels for query/gallery. Training alone uses the relabeled training IDs: `R/tools/official_three_dataset_data.py:8–18`. `R/tools/run_correspondence_roles.py:85–99` takes query/gallery IDs, cameras and scenes directly from those records; model output supplies distances only. No model output is used as the reference.

The three pinned author parsers derive identity/environment from dataset filenames: `R/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:61–85`, `RGBNT100.py:63–84`, and `msvr310.py:67–87`. Independently parsing **27,266 protocol records** by these rules reproduces their identity/camera/scene/view values and training label maps. Training IDs have no intersection with query/gallery IDs. RGBNT `scene` duplicates camera but is not the scoring environment; MSVR uses the `s###` scene/time label.

| Dataset | Train records / IDs | Query records / IDs | Gallery records / IDs | Exclusion |
|---|---:|---:|---:|---|
| RGBNT201 | 3,951 / 171 | 836 / 30 | 836 / 30 | Same identity and camera |
| RGBNT100 | 8,675 / 50 | 1,715 / 50 | 8,575 / 50 | Same identity and camera |
| MSVR310 | 1,032 / 155 | 591 / 52 | 1,055 / 155 | Same identity and scene/time |

Counts are recorded at `R/logs/official_three_dataset_protocols_20260923/RGBNT201.json:78309`, `RGBNT100.json:218978`, and `MSVR310.json:38538`. Query records begin at lines 51550, 95491 and 13587 respectively. Every query retains a valid GT positive after filtering; independently counted minima are 7, 50 and 1. The saved `query_rows` positive/exclusion counts match these independent counts exactly. MSVR's gallery includes identities without a query; this audit does not mistake its 155 gallery IDs for 155 evaluated query IDs.

Full split extraction checks `(protocol count, 1536)` and uses no query cap: `R/tools/run_correspondence_roles.py:66–76`. RGBNT201's temporary loader template duplicates records internally, but the actually returned loader wraps the original records once (`R/tools/train_signal_preserving_v18.py:33–41` versus `R/tools/build_v12_complete_path_oof_targets.py:362–381`), so that template does not double the evaluation set.

Actual metric calls use the pinned author `eval_func`/`eval_func_msrv` plus the independent camera/scene implementation and enforce agreement below `1e-5`: `R/tools/run_correspondence_roles.py:79–107`. The author filters are `R/comparators/Signal-cd1b0a6/utils/metrics.py:68` and `:137`.

Limit: raw dataset images, the original inventory behind `inventory_sha256`, protocol-construction execution and the author's historical complete file lists are not locally available. Filename/protocol consistency supports `real_gt` provenance; it does not certify those unavailable bytes or historical benchmark membership.

## B. Score normalization — PASS

The author AP denominator is the number of GT-relevant gallery items, and the CMC denominator is valid query count: `R/comparators/Signal-cd1b0a6/utils/metrics.py:93–106` and `:153–168`. The independent scorers average precision at relevant ranks and average query Rank-k hits: `R/tools/train_rgbnt100_signal_oof.py:253–268` and `R/tools/train_msvr310_signal_oof.py:223–238`. Neither divides a reported accuracy by the model's maximum, minimum or mean prediction.

`R/tools/run_official_three_dataset_roles.py:230–238` L2-normalizes embeddings before squared Euclidean distance. Attention softmax in `R/modeling/trifusion/semantic_native_evidence.py:36–44` and embedding normalization in `R/modeling/trifusion/native_detail_roles.py:90–100` belong to the representation, not the final score denominator. Author fractions are converted to percent by a fixed factor of 100 (`R/tools/run_correspondence_roles.py:100–102`). No reranking is called; final receipts state `reranking:false`, e.g. `R/trained-model/semantic_native_evidence_20261002_v1_clean_clip_combined_RGBNT201_seed42_full/official_metrics.json:31`.

The independent scorer's Rank-k uses float64 means while author CMC is accumulated as float32. Their tiny differences are explicitly tolerated by `1e-5`; the paired deltas are from the former. This is not a normalization trick or a mismatch at the report's displayed precision.

## C. Result existence, numbers and status — WARN

**Available evidence passes.** `R/results/semantic_native_evidence_complete_20261002/REPORT.md:5–10` agrees with the six original metric receipts and selected epoch histories:

| Dataset | Arm | Best epoch | mAP (%) | Rank-1 (%) | Formal steps |
|---|---|---:|---:|---:|---:|
| RGBNT201 | semantic | 10 | 69.859597835 | 71.650719643 | 2,649 |
| RGBNT201 | combined | 10 | 69.202887329 | 70.574164391 | 2,649 |
| RGBNT100 | semantic | 12 | 79.143114009 | 94.985425472 | 6,559 |
| RGBNT100 | combined | 30 | 78.844603194 | 95.160347223 | 6,559 |
| MSVR310 | semantic | 24 | 52.687040296 | 71.912014484 | 1,000 |
| MSVR310 | combined | 24 | 51.713198240 | 70.896786451 | 1,000 |

Exact endpoint objects begin at `SUMMARY.json:16`, `:92`, `:168`, `:244`, `:320`, `:396`. All six `R/trained-model/semantic_native_evidence_20261002_v1_clean_clip_{arm}_{dataset}_seed42_full/official_metrics.json:19–32` contain selected epoch, metrics and parity/rerank flags. The six matching `R/logs/semantic_native_evidence_20261002_v1/semantic_native_evidence_20261002_v1_clean_clip_{arm}_{dataset}/evaluate.log:16` contain the same parsed final JSON. Their `train.log:16–65` matches every epoch row.

The terminal parent is COMPLETE at `R/logs/semantic_native_evidence_20261002_v1/campaign.json:2`, with six exit-0 jobs at lines 95, 186, 277, 368, 459 and 550; its completed time is **17:42:53.657392 +08:00** at line 555. The original waiter records **one** report invocation, completed **17:47:32.954179 +08:00**, exit 0 (`R/logs/semantic_native_report_waiter_20261002.json:2–22`). Source code enforces output/receipt nonexistence and has no retry loop around the report (`R/tools/wait_semantic_native_evidence_analysis.py:22–53`). These receipts support one recorded invocation; they do not independently prove that no unrecorded command ever ran remotely.

The tracker is **not a current COMPLETE tracker**. It explicitly identifies itself as the **15:20:02** snapshot and contains RUNNING/PENDING rows plus zero report invocations (`R/refine-logs/semantic_native_evidence_v1/EXPERIMENT_TRACKER.md:3–14`). This precedes the terminal receipts. Likewise the manifest's PENDING rows are the immutable launch plan, not the terminal campaign. Treating either as current would be an error; their historical statuses do not negate the later receipt chain. Preserve the sealed historical file and use a separate terminal update for current status.

The prior clean summary matches fixed SHA `784e3938a0203ea043543804c6e2933b43877405d7ca624116d59e59d35538fd`; native-failure summary matches `819ed74b86864786bfc25fac439a1bc8022cdf473e3ab26f021b46648ef04458`. Their embedded rows and all nine historical metric receipts match. Original MSVR-low metrics remain null (`R/results/semantic_native_evidence_complete_20261002/SUMMARY.json:1004`), so there is no fabricated repair of the predecessor failure.

**Local evidence gap:** the 15 saved `official_distances.pt` files in the report input map, the six new `best_map.pth` files, six M0 reload probes, and public CLIP file are absent. The audit verifies their hash *references* across receipts, not their bytes or a fresh strict load. Old native manifest/campaign and native initialization witness files referenced by queue/preparation also are not in the supplied mirror. Therefore full binary provenance, saved-array GT equality, checkpoint state completeness, per-query errors/AP and exact bootstrap intervals are not freshly verified here. This is a WARN, not evidence that the files are nonexistent on the server.

## D. Metric call paths and dead code — PASS

The source and execution evidence support this actual path:

1. `R/tools/queue_semantic_native_evidence.py:38–44,59–64,125–127` installs the EV1 entry and delegates each worker. `R/tools/queue_clean_clip_joint.py:138–164` runs M0, training and final evaluation once, returns on a failed subprocess, then records verification. The two RGBNT201 M0s are reused from preflight at lines 135–146, not run twice.
2. `R/tools/run_semantic_native_evidence.py:51–59` replaces the reused builder/condition and invokes `clean.train` or `control.evaluate`. `R/tools/run_clean_clip_joint.py:136–146` invokes `control.train`; `R/tools/run_visual_update_control.py:188–197` actually calls `runner.official_metrics` every epoch and stores its output.
3. Final evaluation loads the full state with `strict=True` (`R/tools/run_visual_update_control.py:102–109`), checks the selected epoch/metrics, calls the same scorer, saves distances, checks the unmodified `1e-5` gate and writes the receipt (`:227–249`). The scorer invokes both pinned author and independent implementations, as described in A.
4. The CPU report calls `compare` for four controls on each of three datasets (`R/tools/report_semantic_native_evidence_complete.py:83–106`). `R/tools/analyze_correspondence_distances.py:28–75` reads saved distances, checks their hashes, calls camera/scene scores, verifies array equality and computes the reported diagnostic fields. Report source writes these fields at `R/tools/report_semantic_native_evidence_complete.py:126–165`.

No claimed metric is merely a function definition without a call/output path. Imported legacy OOF training/evaluation entry points and author `R1_mAP`/`R1_mAP_eval` convenience wrappers are not this campaign's invoked path; their presence does not demonstrate another experiment or reranking. The active pipeline calls the underlying author functions directly. This PASS is scoped to the claimed outputs, not a certification that the full 243-file inherited tree contains no unused code.

One report dependency, `tools/report_clean_clip_joint_complete.py:elapsed`, is not in the mirror or the report input hash map. The separately supplied current repository file `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_clean_clip_joint_complete.py:20–21` implements plain ISO timestamp subtraction. Its SHA is `25250470077bcda629633fba47c94f91bb686910ed708310a6afd46ec3709e86`; repository HEAD at inspection is `3febb072794e2c703eff362e0070497a5666b5f2`. This extra source supports the timing semantics, not the identity of the remote executed helper. Independent timestamp arithmetic already agrees with the reported elapsed values.

## E. Scope, selection, comparisons and resources — WARN

**Actual scope.** Three datasets, two new arms each, **one seed (42)**, **50 epochs each**, **300 epochs / 20,416 formal batches total**, plus six eight-batch engineering M0s. This is fixed in `R/tools/run_semantic_native_evidence.py:41–47`, `R/tools/queue_semantic_native_evidence.py:81–98`, and `R/results/semantic_native_evidence_complete_20261002/SUMMARY.json:6–13`. The six historical clean controls and three native-high records are reused evidence, not nine newly trained EV1 arms (`R/tools/report_semantic_native_evidence_complete.py:36–51,82–99`). N2/N3 are not tested (`REPORT.md:29`).

**Initialization and intervention.** The active constructor loads the public CLIP file directly, disables Signal A/B modules, verifies 152 public visual tensors, and creates fresh camera/head/module state (`R/tools/run_clean_clip_joint.py:37–107`; pinned `modeling/meta_arch.py:68–86`). The generic legacy ReID-checkpoint builder is bypassed by the installed EV1 builder. There is no external text, SAM or DINO path in this intervention.

Semantic and combined have the same common initialization hashes. Preparation compares each actual tensor before writing witnesses (`R/tools/prepare_semantic_native_evidence.py:37–68`); the six preflight witnesses report those comparisons (`R/logs/semantic_native_preflight_20261002_v1/preflight.json:249–496`). This audit checks witness/source/receipt consistency, not the absent initialization tensors afresh. Both arms retain semantic keys/values with **512** CNN candidates; combined adds the stride-8 image stem and **93,248 actual trainable parameters** (six parameter tensors), then sums semantic and native values before the same value projection (`R/modeling/trifusion/semantic_native_evidence.py:17–44`). The existing 1536-dimensional readout is retained (`R/modeling/trifusion/native_detail_roles.py:90–100`; extraction width asserted at `R/tools/run_correspondence_roles.py:75`).

Consequently EV-A is an **added-detail-plus-capacity** comparison, not proof of the isolated information source. EV-B also changes the old clean roles' 128 candidates to 512. Historical native-high shares checked initial-state structure, but is not a concurrent new arm or proof of matched training randomness. Step logs record scalar losses/LRs, not actual minibatch identity/environment tuples. The report's qualifiers at `REPORT.md:30–36` are warranted.

**Recipe and selection.** The live loop uses new-module LR `3.5e-4`, visual LR `5e-6`, AdamW weight decay `1e-4`, warmup 5/cosine 50, CE smoothing `0.1` and triplet margin `0.3` (`R/tools/run_visual_update_control.py:25,120–141,150–184`; dataset config optimization lines 40–43 or 51–52). B64/K8 is on the called data-loader path (`R/tools/official_three_dataset_data.py:21–39`; `R/tools/train_rgbnt100_signal_oof.py:113–123`; `R/tools/train_msvr310_signal_oof.py:89–99`). The source keeps visual parameter storage FP32 but uses autocast/GradScaler in the training loop; a claim of entirely FP32 training would overstate it.

Selection is maximum **official fused mAP**, with a later epoch winning an exact tie, and all CMC metrics follow that same checkpoint (`R/tools/run_visual_update_control.py:189–193,235–241`; `R/tools/queue_clean_clip_joint.py:97–106`). All six text histories satisfy that policy. The official evaluation set is used during development and for 50-way epoch selection, so these are post-selection benchmark scores rather than unbiased holdout estimates (`REPORT.md:32–33`). No multiple-seed stability follows from this evidence.

**Registered outcomes.** EV-A compares combined against semantic; EV-B compares combined against fixed clean roles. Both require positive mAP, nonnegative Rank-1, and at least +0.5 mAP on RGBNT201/MSVR310 (`R/refine-logs/semantic_native_evidence_v1/EXPERIMENT_PLAN.md:24–30`; `R/tools/report_semantic_native_evidence_complete.py:100–114`). Independent arithmetic confirms:

| Dataset | EV-A delta mAP / Rank-1 (points) | EV-A | EV-B delta mAP / Rank-1 (points) | EV-B |
|---|---:|---|---:|---|
| RGBNT201 | -0.656711 / -1.076555 | FAIL | -0.673508 / -0.837321 | FAIL |
| RGBNT100 | -0.298511 / +0.174927 | FAIL | -2.078457 / +0.874636 | FAIL |
| MSVR310 | -0.973842 / -1.015228 | FAIL | -0.291004 / +2.199662 | FAIL |

Evidence: `REPORT.md:14–25`, `SUMMARY.json:1043,1254,1883,2194,3123,3444`. Every dataset fails both gates because its mAP delta is negative. Improvements in selected R1 or descriptive historical comparisons cannot overturn the conjunctive gates.

**Bootstrap scope.** `R/tools/analyze_correspondence_distances.py:57–74` resamples **identity mean AP changes**, 2,000 draws with RNG seed42, over **30/50/52 query identities**. Its macro mean weights identities equally and differs conceptually from query-weighted benchmark mAP. It describes fixed selected models; it is neither training-seed uncertainty nor unbiased statistical significance. JSON identity counts/means are arithmetically consistent, but tensor-derived memberships and exact interval endpoints were not recomputed.

**Resource scope.** The six training receipt intervals sum to **22,923.145167 seconds (6.367540 hours)**; parent observed wall interval is **10,567.303627 seconds (2.935362 hours)** (`SUMMARY.json:4444–4445`). Independent timestamp subtraction agrees. The former includes epoch evaluation and end-of-loop bookkeeping but begins after model/loader construction (`R/tools/run_visual_update_control.py:116–148`) and excludes M0, final reload/evaluation and prior campaigns. It must not be described as total GPU cost. The record also reports **4,425,957,536 logical output bytes** and **53,340,336,128 free disk bytes** (`SUMMARY.json:4446–4447`); absent remote binary files and filesystem state prevent independently verifying those physical-resource observations. Peak allocated bytes are receipt-reported PyTorch allocations, not total device reservation or power usage.

## F. Evaluation type — PASS

| Evidence | Classification | Permitted interpretation |
|---|---|---|
| Six final EV1 retrieval endpoints | `real_gt` | Selected single-seed scores using dataset identity/environment labels. |
| Reused clean/global/roles/native-high retrieval records | `real_gt` historical records | Fixed prior comparisons, not fresh EV1 training runs. |
| Paired repairs, AP changes, identity bootstrap | `real_gt` post-selection diagnostic | Analysis of saved model distances; no new inference or independent test set. |
| Construction parity and M0 reload/gradient probes | Engineering checks; GT-supervised M0 training, not a retrieval evaluation | Code execution/gradient/reload feasibility only. |
| Original MSVR-low failed formal result | Unavailable, preserved null | No accepted score or valid high-minus-low result may be inferred. |

Classification follows the record-to-label path in A, the analyzer call at `R/tools/analyze_correspondence_distances.py:37–48`, and M0 handling at `R/tools/run_visual_update_control.py:147–159,204–223`. None is synthetic/model-generated GT, simulation, human evaluation, or self-supervised proxy retrieval.

## Claim boundaries and minimal follow-up

**Supported within available evidence:** six recorded EV1 endpoints completed the fixed full50 procedure; the published numeric table, steps and selected epochs match the original text receipts; combined's mAP is lower than semantic on every tested dataset; all six component predicates and both joint EV gates fail; no rescue of the original N1 MSVR-low failure occurred in these reported results.

**Needs explicit qualification:** original strict-reload success and distance/GT parity are code-and-receipt supported, not freshly checked binary facts in this local audit; diagnostic repairs and bootstrap intervals are fixed-model, post-selection receipt-backed analyses; common initialization is source/witness supported; timings cover the named intervals only.

**Unsupported:** positive transferable native-detail gain, isolated detail-source causality, role necessity, unbiased or multi-seed robustness, N2/N3 efficacy, a unique reason for the negative result, three-dataset baseline/SOTA achievement, novelty, total GPU cost, or full independent reproduction of image/checkpoint/distance provenance.

No blocking integrity defect or experimental code patch is established. Keep the negative outcome and qualifiers. For current tracking, append a separate terminal snapshot while retaining the sealed 15:20 tracker. A future stronger binary audit would need the existing saved tensors/checkpoints, dataset inventory and their original hashes; the present report must not be relabeled as having inspected them. The unmirrored elapsed helper also needs an execution-time hash if a future claim requires complete dependency provenance.
