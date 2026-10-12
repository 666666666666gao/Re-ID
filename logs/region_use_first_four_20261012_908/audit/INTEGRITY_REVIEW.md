# Four original endpoints: integrity review908

**Overall: WARN; no blocking code or saved-result arithmetic defect found.** All 261 deterministic checks pass. The four endpoint reports are supported provisionally as saved producer/text evidence. The standalone claims draft now includes the explicit official-benchmark selection/development qualification; three narrow document checks close W1 without rerunning the original checks.

Fresh native Codex reviewer: `/root/audit_region_use_four908`. Requested Astra/max; actual serving model and effort **UNATTESTED**. Semantic review is **same-family / provisional**. Deterministic acceptance applies only to the documented exact checks.

The actual received task body is preserved in `REVIEW_REQUEST_ACTUAL.txt`; it does not purport to expose invisible parent tool arguments or timestamps. All 79 primary/source/prior-review/claim snapshots are retained with manifests. No production source, checkpoint or distance was changed; no SSH, GPU, model, optimizer or production report call was performed.

## Verified endpoint values

| Dataset | Condition | Selected epoch | mAP | R1 | R5 | R10 | Last mAP | Recorded steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RGBNT201 | uniform | 8 | 74.788049 | 78.947371 | 88.636363 | 92.224878 | 71.636187 | 2649 |
| RGBNT201 | ambiguity | 8 | 74.629901 | 79.066986 | 88.277513 | 91.866028 | 71.960387 | 2649 |
| MSVR310 | uniform | 38 | 50.541777 | 67.851102 | 80.541456 | 85.448390 | 50.490284 | 706 |
| MSVR310 | ambiguity | 38 | 50.545760 | 67.851102 | 80.541456 | 85.448390 | 50.491665 | 706 |

Every row is one seed42 run, 50 epochs, the latest maximum-mAP epoch under the existing tie rule, and the same checkpoint for all CMC metrics. There is no epoch/seed averaging. All 30 intake digests match; each original job has exactly prepare → m0 → train → evaluate, exit0/COMPLETE. Saved initialization/M0 acceptance summaries agree with the formal initializer and record eight M0 updates/activity. The absent original M0 payloads were not independently replayed.

At the saved 2026-10-12 22:03:02 +08:00 observation the campaign remained RUNNING: RGBNT100 uniform had 3 completed epochs and an active train stage; ambiguity was unattempted. This review does not accept six-end completion or the original all-three goal.

## A–F findings

### A gt provenance — PASS

Retrieval identities/environments come from sealed dataset protocol records. All path identity/environment fields, training/evaluation identity disjointness, and legal-positive counts were recomputed. All saved batch paths/labels/cameras map to training records. Global model margins only weight the auxiliary training objective; they do not create retrieval GT. Physical dataset image bytes were not inspected.

Evidence: `evidence_snapshot/production_sources/tools/official_three_dataset_data.py:8`; `evidence_snapshot/production_sources/tools/run_correspondence_roles.py:86`; `evidence_snapshot/production_sources/tools/run_region_use_probe.py:108`; `evidence_snapshot/production_sources/modeling/trifusion/region_use_objective.py:21`.

### B score normalization — PASS

Complete sorted legal gallery enters AP; only same identity AND same camera/scene is removed. Different-identity distractors in the query environment remain. AP averages precision over legal positives, and mAP/CMC average all valid queries; every protocol query has positives. No retrieval metric is normalized by model-output extrema or means. L2 feature normalization and normalized auxiliary loss weights are not score inflation. Upstream float32 CMC versus per-query integer arithmetic is reconciled with the existing <1e-5 percentage-point tolerance.

Evidence: `evidence_snapshot/production_sources/tools/train_rgbnt100_signal_oof.py:253`; `evidence_snapshot/production_sources/tools/train_msvr310_signal_oof.py:223`; `evidence_snapshot/production_sources/comparators/Signal-cd1b0a6/utils/metrics.py:68`; `evidence_snapshot/production_sources/comparators/Signal-cd1b0a6/utils/metrics.py:137`.

### C result existence and original completion — WARN

All 30 intake files have exact expected SHA256; 16 stage receipts are ordered prepare/m0/train/evaluate, COMPLETE/exit0. Four receipts equal original job results and bind seed42, full50, same selected-best epoch, checkpoint/distance hash strings. Histories, logs, candidate summaries and rounded draft table agree. Binary hashes are consistent producer claims, not independently recomputed from absent bytes. L1 and external historical-reference boundary L2 apply.

Evidence: `evidence_snapshot/intake/INTAKE_MANIFEST_908.json:3`; `evidence_snapshot/intake/logs/region_use_resume906/CPU_REPORTS_908.json:2`; `evidence_snapshot/production_sources/tools/report_region_use_endpoint.py:30`; `evidence_snapshot/production_sources/tools/run_foundation_recipe.py:295`.

### D live call path — PASS

Direct source reads trace the actual mode dispatcher through configured foundation train/evaluate and official scorer, and reporter main through paired()/camera_scores()/scene_scores(). Four saved commands call the reviewed reporter SHA 807746c6ab295fc39de40bd3a91028c57dc7d346f89e1351d803d5ac401a314d; four exit0 receipts and stdout lines agree with four distinct reports. These are saved execution evidence, not a new reviewer invocation. Unrelated OOF training/evaluation entry points and author visualization/reranking classes are not used by this pipeline.

Evidence: `evidence_snapshot/production_sources/tools/run_region_use_probe.py:80`; `evidence_snapshot/production_sources/tools/run_region_use_probe.py:122`; `evidence_snapshot/production_sources/tools/run_independent_native_evidence.py:168`; `evidence_snapshot/production_sources/tools/run_foundation_recipe.py:300`; `evidence_snapshot/production_sources/tools/run_correspondence_roles.py:93`; `evidence_snapshot/production_sources/tools/report_region_use_endpoint.py:71`; `evidence_snapshot/production_sources/tools/report_prepool_dense_correspondence.py:20`; `evidence_snapshot/intake/logs/region_use_resume906/CPU_REPORTS_908.json:3`.

### E coverage scope and claims — WARN

All 6710 step/batch records cover 200 endpoint-epochs with continuous epoch batch IDs and exact sealed historical raw-semantic order digests. Auxiliary coverage is independently recomputed from train GT for every batch. Eight full-query comparisons reproduce AP/CMC, identity counts/macros, all four transition contributions, repairs-minus-new-errors R1, and ceil(5% N) signed tails. Scope is two datasets, two conditions and one training seed, with official split selection/development consumption. No seed/epoch average, own-global/regional decomposition, local efficacy or unique causal explanation is supported. W1 was closed by an append-only standalone draft wording clarification; no original numerical checks were rerun.

Evidence: `evidence_snapshot/production_sources/tools/report_region_use_endpoint.py:46`; `evidence_snapshot/production_sources/tools/report_region_use_endpoint.py:59`; `evidence_snapshot/production_sources/tools/report_prepool_dense_correspondence.py:41`; `evidence_snapshot/production_sources/tools/report_region_use_endpoint.py:73`; `evidence_snapshot/claims/CLAIMS_908_DRAFT.md:14`.

### F evaluation type — PASS

Retrieval evaluation is real_gt according to directly audited metadata/source and saved text evidence. Auxiliary loss/support are dataset-ID-supervised training diagnostics with model-dependent weighting, not semantic-part ground truth or a local retrieval evaluation. Prior CPU fixtures were source/simulation checks and confer no neural runtime acceptance.

Evidence: `evidence_snapshot/production_sources/tools/run_correspondence_roles.py:88`; `evidence_snapshot/production_sources/modeling/trifusion/region_use_objective.py:16`; `evidence_snapshot/production_sources/tools/report_region_use_endpoint.py:95`.

## Coverage, transitions and tail checks

RGBNT201 uses all 836 queries, 836 gallery entries and 30 query identities; each query has 7–21 legal positives. MSVR310 uses all 591 queries, 1,055 gallery entries, 52 query identities and 155 gallery identities; each query has 1–31 legal positives. Physical image/tensor completeness is a producer-evidence boundary, while the local metadata counts, identity order and every legal-positive count were checked directly.

All 6,710 formal step/batch pairs were read. RGBNT201 has 57,416 / 169,536 eligible query exposures (33.8666%), 11,089,680 legal triplets and 92 zero-support batches per condition. MSVR310 has 21,708 / 45,184 (48.0436%), 2,793,000 legal triplets and no zero-support batch per condition. Counts are exposures, not unique samples. Every batch maps to training GT; no query/gallery identity enters the training batches. Lower auxiliary loss describes training behavior and does not prove causal efficacy.

| Dataset / condition | Control | ΔmAP (pp) | ΔR1 (pp) | Repairs / new errors | Worst 5% N | Tail mean ΔAP (pp) |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 / uniform | raw_semantic | +0.413187 | +0.119617 | 7 / 6 | 42 | -5.299494 |
| RGBNT201 / uniform | independent_global | +0.491385 | +0.000000 | 6 / 6 | 42 | -4.940888 |
| RGBNT201 / ambiguity | raw_semantic | +0.255039 | +0.239234 | 8 / 6 | 42 | -6.319871 |
| RGBNT201 / ambiguity | independent_global | +0.333237 | +0.119617 | 8 / 7 | 42 | -6.046006 |
| MSVR310 / uniform | raw_semantic | -0.000448 | +0.000000 | 0 / 0 | 30 | -0.351230 |
| MSVR310 / uniform | independent_global | -0.000351 | -0.169205 | 0 / 1 | 30 | -0.559569 |
| MSVR310 / ambiguity | raw_semantic | +0.003535 | +0.000000 | 0 / 0 | 30 | -0.290246 |
| MSVR310 / ambiguity | independent_global | +0.003631 | -0.169205 | 0 / 1 | 30 | -0.539600 |

Each of the eight comparisons passes all-query AP/CMC recomputation, identity counts and macro averages, improved/worsened counts, four disjoint transition groups on the full-query denominator, sum-of-contributions = ΔmAP, and `(repairs − new errors) × 100 / N = ΔR1`. The tail is exactly the lowest signed `ceil(0.05 × N)` query AP changes: 42 or 30, without selecting only damaged queries. Candidate per-query values agree across both control comparisons; each control agrees across conditions. These are arithmetic checks of saved AP/rank values, not rescoring absent distance matrices.

For RGBNT201 uniform versus independent global, 654 first-still-correct queries contribute +0.421672 pp of the total +0.491385 pp. Repairs and new errors are 6/6. Thus the mAP gain cannot be described solely as first-rank repair. Both MSVR conditions have 0 repairs / 1 new error against independent global. The small mAP differences do not establish stability, significance, equivalence or an independent local gain.

## Timing and resource boundaries

| Endpoint | Producer train-stage wall (s) | Training + epoch evaluation/save interval (s) | Epoch loop sum (s) | Strict evaluator internal (s) | CPU report wall (s) |
|---|---:|---:|---:|---:|---:|
| RGBNT201 uniform | 2428.699669 | 2402.528419 | 1646.581627 | 39.604387 | 4.665100 |
| RGBNT201 ambiguity | 2422.959254 | 2395.080174 | 1645.164808 | 41.394468 | 4.411200 |
| MSVR310 uniform | 1433.092709 | 1405.454779 | 516.258550 | 43.796942 | 3.799470 |
| MSVR310 ambiguity | 1430.653969 | 1403.034922 | 515.309923 | 43.593362 | 3.792572 |

All intervals reproduce the original timestamps/epoch loop sums. Producer train-stage wall starts just after Popen and ends after wait; it includes construction and train/eval/save, but excludes pre-stage GPU admission wait, prepare/M0 and the separate strict evaluator. The training JSON interval starts after construction/optimizer/loader and ends before final receipt/wrapper writes. Per-epoch `seconds` is captured before evaluation/save. Their difference is not a pure measured evaluation-only timer. Strict evaluator time separately includes fresh construction, strict reload and scoring. See `tools/queue_native_research.py:63`, `tools/run_foundation_recipe.py:200`, `:246`, `:283`, `:287`, `:309`.

Both partition allocated/reserved counters are retained and agree with training receipts. `tools/run_native_partitioned.py:37` resets both before optimization; the foundation loop resets the default GPU again at `run_foundation_recipe.py:209`. GPU1 can therefore include optimizer setup. These are producer counters with asymmetric reset boundaries, not symmetric whole-process maxima. No power/temperature/current GPU query was made.

## Minimal actions and claim ceiling

- **W1 — CLOSED after document clarification; original WARN preserved.** The original draft said same-mAP-best but did not explicitly say official query/gallery metrics select the checkpoint across 50 epochs. The clarified draft now appends this sentence explicitly at line 22. The endpoint reports do already disclose official-benchmark development consumption. The final strict pass reloads the selected checkpoint; it is not an independent untouched test split. The original draft matches the author-preserved predraft byte for byte; the only change appends the required scope sentence. `CLAIM_SCOPE_CLARIFICATION_CHECK.json` passes all three narrow checks. No producer, result or original 261-check rerun occurred.
- **L1 — Saved producer evidence is not independent binary or runtime replay.** The local intake contains text/metadata/report files only. Actual checkpoint states, official distance tensors, old control tensors, image bytes, and full original M0 tensor/step payloads remain outside this review. Saved strict evaluator receipts, stage exit codes and M0 acceptance summaries agree, but the reviewer did not independently load those binaries, reconstruct AP from distances, or witness the remote process. Minimal action: Retain the producer-evidence and no-independent-replay qualification. This does not authorize a replay, new parity/kernel check or broader campaign gate.
- **L2 — Four endpoints and one seed do not establish campaign completion or stable efficacy.** The 22:03:02 +08:00 witness has four COMPLETE jobs, RGBNT100 uniform train RUNNING with 3 completed epochs and its ambiguity job PENDING/unattempted. Historical Signal numbers in the draft are explicitly external project references, not recertified primary results in this intake. Minimal action: Keep the exact four-endpoint/seed42 scope and all-three goal unmet. Do not call historical Signal references independently certified or use this audit as an all-three completion audit.

The reviewed draft table, repair/error counts, contribution, support/loss/drop figures and timing numbers agree with the saved evidence. It explicitly rejects stronger-teacher, own-global/regional decomposition, unique-cause, stable multi-seed, significance and all-three-success claims. Historical Signal figures remain externally referenced and unrecertified in this review. No new production code fix is required by these artifacts.

The previous R1/R2/supplement source reviews remain intact. Their source-only PASS was not substituted for this result audit. The newly pinned author scorer has exact sealed SHA `91604acb7d978462c16904910a645d7d1697c15bf150da529099004c8b5eecb2`. All directly reviewed sealed source snapshots match; the new reporter is separately self-hashed in all four summaries. The entire inherited 463-file source inventory was not re-audited.

Reproduction of local checks: run `deterministic_checks.py`, then `supplemental_checks.py` using the installed standard-library Python. Neither script imports project, NumPy or torch modules or accesses remote payloads. `DETERMINISTIC_CHECKS.json` preserves every exact check, and `INTEGRITY_REVIEW.json` preserves structured findings, per-endpoint values and claim impacts.

## Document clarification closeout

W1 is closed. `initial_review/INTEGRITY_REVIEW.md` and `.json` preserve the original WARN verbatim. `evidence_snapshot/claims/CLAIMS_908_SCOPE_CLARIFIED.md:22` explicitly states per-epoch official evaluation, mAP-best selection, method development, and no untouched-test or full-training-seed confirmation. The overall WARN remains only for the disclosed scope/provenance limitations; no blocking defect or open source/document repair remains.
