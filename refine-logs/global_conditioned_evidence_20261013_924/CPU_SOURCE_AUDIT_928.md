# Endpoint report 928 — cold source audit

**Overall: WARN. Source checks: PASS. No concrete blocking source defect found.**
`source_ready_for_cpu_report=true`; `unresolved_blocking_source_defects=[]`.

This conclusion applies only to the new single-endpoint CPU report after its original full50/first-strict endpoint is accepted. It neither approves training nor certifies any runtime result, retrieval gain, or completed two-arm experiment.

Requested reviewer: gpt-6-astra / max. Actual model and effort: **UNATTESTED**. Review independence: **same-family**; acceptance: **provisional**.

## Audited scope and context

- Draft: `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/report_evidence_formation_endpoint.py`; SHA256 `0884d747e4d4719a87894bf5e0091a214995f2ddc04a9c551e50914d53e2accb`.
- Producer: `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_full_controller926.py`; SHA256 `ce75b9987862b189eddedf930762f719a25f9bc9c3bc3baaa2db09f54abab5b0`; identical to the published controller copy.
- Actual delegated training/evaluation chain, report helper/scorer, model activity-field producer, two old-control seals, public plan/tracker and static launch/source provenance were inspected. The new CPU launch/capture wrapper was subsequently inspected and its one-line correction verified before execution.
- Mandatory SOUL.md, USER.md and today/yesterday notes exposed prior session/TriFusion conclusions; note output was truncated. Fresh-agent does not mean strictly context-blind here. MEMORY.md was not loaded in this delegated session; BOOTSTRAP.md was absent.
- No SSH/network/GPU query, torch import, model/report execution, training approval, transient-result inspection or production edit occurred. Output is confined to this new private review directory.

References below use the private draft/controller basenames; all `tools/`, `modeling/` and `refine-logs/` references are relative to `C:/Users/gb/.trifusion_github_publish_22c3bee`. Full absolute paths and byte hashes are in SOURCE_AUDIT.json.

## Findings

### A_ground_truth_provenance: PASS

Source wiring loads full protocol query/gallery records; identity, camera and scene arrays originate in dataset protocol metadata. Saved distances contain these labels. No model output becomes retrieval ground truth. Exposure sensitivity/support is separately labeled model-derived activity, not region ground truth. Source classification only; remote data and saved binaries were not loaded.

Evidence: `report_evidence_formation_endpoint.py:75-76`; `report_evidence_formation_endpoint.py:104`; `tools/run_correspondence_roles.py:66-108`; `tools/official_three_dataset_data.py:8-18`; `tools/run_official_three_dataset_roles.py:58-68`; `tools/build_official_three_dataset_protocols.py:32-80`; `modeling/trifusion/global_relation_inputs_diagnostic.py:35-57`.

### B_score_normalization: PASS

camera_scores uses all legal positives and excludes only same-identity/same-camera gallery entries. AP is the mean of precision at all retained positive ranks; mAP and rank metrics average over the full query set. No metric divides by model prediction extrema. Activity gradient means are proxy summaries, not accuracy normalization.

Evidence: `tools/train_rgbnt100_signal_oof.py:253-268`; `tools/report_prepool_dense_correspondence.py:18-55`; `report_evidence_formation_endpoint.py:59-71`; `report_evidence_formation_endpoint.py:78-86`.

### C_result_existence_and_completion: WARN

This is an unexecuted report draft, with no completed result or numerical gain claimed. It requires an accepted endpoint plus exactly its successful train/evaluate stages, 50 history epochs, matching best epoch/metrics and checkpoint/distance hashes. These producer keys exist in the actual source. The remote artifacts were intentionally not inspected, so future runtime existence/completion is not certified.

Evidence: `report_evidence_formation_endpoint.py:18-38`; `evidence_full_controller926.py:53-91`; `tools/run_foundation_recipe.py:243-257`; `tools/run_foundation_recipe.py:279-312`; `refine-logs/global_conditioned_evidence_20261013_924/EXPERIMENT_TRACKER.md:7-12`.

### D_live_wiring_and_schema: PASS

The actual dispatch is run_global_conditioned_evidence -> run_global_task_role -> run_role_input_detach -> run_native_research -> run_native_partitioned -> run_independent_native_evidence -> run_foundation_recipe. configure rewires foundation schema, build, loss and train wrapper. Formal loss components preserve global_loss/fused_role_loss/evidence_exposure; the actual loop writes them to training_steps.jsonl and writes official_fused history. The partition wrapper adds partition_peak_memory. The report calls paired/descriptor rather than their unrelated main function.

Evidence: `tools/run_global_conditioned_evidence.py:16-20,54-81,84-107`; `tools/run_global_task_role.py:25-49`; `tools/run_role_input_detach.py:7-13,22-41`; `tools/run_native_research.py:8-25`; `tools/run_native_partitioned.py:43-60`; `tools/run_independent_native_evidence.py:95-97,167-184`; `tools/run_foundation_recipe.py:229-257`; `report_evidence_formation_endpoint.py:47-71,75-87,102`.

### E_scope_and_claim_ceiling: WARN

A single accepted original seed42 full50/first-strict endpoint permits a saved-ranking descriptive diagnosis only. The report explicitly sets same_model_global_decomposition_available=false and names the old global history independent_global_same_epoch_metrics. It cannot close the registered gradient-random comparison, same-model global/correction/fused decomposition, causal mechanism, seed uncertainty, or Signal superiority. These are scope limits, not blockers to this endpoint report.

Evidence: `report_evidence_formation_endpoint.py:92-104`; `refine-logs/global_conditioned_evidence_20261013_924/EXPERIMENT_PLAN.md:11-14,30-38,46-50`.

### F_evaluation_type: PASS

No synthetic or self-referential retrieval ground truth is introduced. Role-exposure activity is explicitly not certified region importance.

Evidence: `tools/run_correspondence_roles.py:88-107`; `report_evidence_formation_endpoint.py:104`.

### activity_means_and_denominators: PASS

Producer norms are B x 3 x 128; available_sample_modalities is B x 3, active is nonzero norm sum, excluded is 32 per active sample/modality, and norm mean includes zero/inactive entries. Formal B=64 is enforced by the loop and draft checks 192 available modalities per step. Consequently the unweighted per-step norm mean is the equal-element epoch mean across 192*128 positions per step. Loss means are step means; summed available modalities, legal/supported relations and step counts preserve denominators. supported_active_steps is a conjunction, not an assertion that every supported step is active.

Evidence: `modeling/trifusion/global_conditioned_evidence.py:59-84`; `modeling/trifusion/global_relation_inputs_diagnostic.py:45-57`; `tools/run_foundation_recipe.py:224-225`; `report_evidence_formation_endpoint.py:47-71`.

### query_group_ap_decomposition: PASS

Four mutually exclusive and exhaustive Rank-1 transition groups contribute sum(delta AP in percentage points)/all queries, and their total is asserted equal to the complete mAP delta. This is a contribution decomposition, not a within-group mean. paired checks exact query/gallery label order and shape, rescoring both full matrices, and verifies Rank-1 delta=(repairs-new_errors)*100/N. Worst five percent means the lowest ceil(0.05*N) query AP deltas, not an identity statistic or confidence bound.

Evidence: `tools/report_prepool_dense_correspondence.py:18-61`; `report_evidence_formation_endpoint.py:75-87`.

### required_sha_pins: PASS

Hardcoded region seal hash is byte-exact. The nested control seal key, raw semantic batch-order path and independent-global training.json path exist with matching hashes in the seals; both control distance row hashes equal seal entries. New official metrics, checkpoint and distances are hash-checked. The 469-entry runtime map and M0 map are rechecked at report execution; directly inspected runtime producer/report helper files are byte-exact to the static source synchronization map. Remote byte verification is deliberately outside this audit.

Evidence: `report_evidence_formation_endpoint.py:18-21,29-43,51-53,75-76,88-89`; `tools/report_prepool_dense_correspondence.py:23-29,58-61`; `refine-logs/region_use_fixed_best_diagnosis918/INPUT_SEAL.json:473-475,488-493,517`; `refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json:531-594,959-1020`; `launch_evidence_full926.py:6-16`.

### phase_ownership_and_cpu_only: PASS

The report is independent of the producer and only reads accepted artifacts; its only mutation creates a new report directory/SUMMARY.json. It does not launch jobs, restart training, alter campaign state, call model forward/backward, use CUDA, load model checkpoint tensors, or run an optimizer. paired loads saved distance bundles onto CPU; set_num_threads(1) is the sole report torch configuration call. Static analysis of 13 local report-import modules found only definitions, constants, path setup and guarded mains at import, with no neural/queue execution. The existing sequential full50 controller remains producer-owned.

Evidence: `report_evidence_formation_endpoint.py:14-25,37-38,72-76,102-109`; `tools/report_prepool_dense_correspondence.py:23-29,135-136`; `evidence_full_controller926.py:43-75,89-95`; `tools/queue_region_use_probe.py:133-136`; `tools/collect_correspondence_roles.py:24-29`.

## Deterministic checks and limits

- Stdlib AST parsing passed for 33 Python files. The statically followed report import graph contains 13 local modules; audited modules were never imported or executed.
- Source synchronization manifest: 469 entries; 335 locally byte-exact, 107 absent comparator/protocol entries, 27 differing only by CRLF versus LF, zero other mismatches. The directly used report helpers and inspected runtime producer chain are byte-exact. A complete remote byte audit was not attempted; report execution retains exact runtime-map checks.
- Region seal SHA256 is `8e5c37e57c088a8ab53344266195bbc7183ccd27a94d0f0910ff9c20998b4b41`; nested control seal SHA256 is `9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e`. Raw semantic distance/order and independent-global distance/history paths are present in both seals with matching values.
- The intended runtime is the existing Linux repository with the seals' absolute paths. The partial Windows public checkout is not represented as a runnable complete deployment.
- The initial default python launcher failed before running the snippet (`No pyvenv.cfg`); existing `E:/python.exe` subsequently performed stdlib-only checks. No installation or environment edit occurred.
- Dataset files, complete protocol bytes, upstream comparator evaluator and saved distance binaries were not loaded. Source-ready does not replace future artifact/runtime verification.

## Required action

No source change is required for this bounded endpoint report. Preserve the explicit unavailable same-model decomposition and deferred final gradient/random comparison. Do not promote this source audit into a training or performance acceptance.

Full audited hashes, checklist details and context caveat are recorded in SOURCE_AUDIT.json. Parent owns capture of the full native reviewer transcript and verbatim final response.

## First wrapper correction — historical source check, superseded by the key fix below

`run_evidence_gradient_cpu_report928.py` had SHA256 `35a39271436a92a9e7bee011a817f8108f835751f0eb231d1443143b83aa8a0b` at the first SHA-binding review. Its readiness gate and exact reviewed-report SHA assertion are at lines 8–10. The exact audit lookup key is `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/report_evidence_formation_endpoint.py`; expected report SHA256 remains `0884d747e4d4719a87894bf5e0091a214995f2ddc04a9c551e50914d53e2accb`.

The original wrapper, SHA256 `f7f2c64c852e66f3fd36e2e06f235a217be14d9e7009d2dc679588a7fcfc2fbb`, had one blocking omission: readiness was checked but the source digest was not bound to the audit (original lines 8–11). The author added exactly one assertion before source embedding; deleting that added line reconstructs the original SHA. This finding is retained as corrected before execution.

The current wrapper creates a fresh administration directory, writes its private report copy there, and launches only that report with empty `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` (lines 17–30). It does not replace production source or launch a model/training command. It waits for the child and writes the remote child EXIT receipt/log, then captures local stdout/stderr and transport exit before parsing; received child receipt/log are retained before asserting success (lines 31–44). Both outer wrapper and embedded remote code pass AST parsing. No wrapper execution, SSH or neural call was performed by the reviewer.

Final gate remains `source_ready_for_cpu_report=true`, `unresolved_blocking_source_defects=[]`. Overall WARN reflects the disclosed source-only and same-family/provisional limits; no retrieval claim is accepted.

## Latest narrow follow-up — explicit manifest key, PASS

The prior wrapper's first local invocation is recorded as `native410cd6`, EXIT1 at line 10 before SSH or CPU-report execution in `EVIDENCE_CPU_LAUNCHER_KEY_FAILURE_928.txt:1-4` (SHA256 `701a94cf3a1fb1dd21790fe651428ea23a6ef08b9460670de185199049646ee6`). `Path.resolve()` yields `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/report_evidence_formation_endpoint.py`, but the audit map stores the supplied C alias. The reviewer independently confirmed that exact resolved path, absent D manifest key, existing C key, and identical alias/resolved report bytes. The preserved failure file is explanatory text; the original native tool transcript was not independently consumed by this reviewer. Statement order places the lookup before the SSH call at line 37.

Only line 10 changed to use the explicit existing audit key `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/report_evidence_formation_endpoint.py`. It still requires SHA256 equality. Replacing this literal lookup with the prior `source.as_posix()` expression reconstructs previous wrapper SHA256 `35a39271436a92a9e7bee011a817f8108f835751f0eb231d1443143b83aa8a0b`. No fallback, report edit or remote action was introduced.

Current wrapper SHA256: `ed18ea91dd0c5b948861dc72a64ccb625a85704622e567a8158c6579ea7f3503`. Report SHA256 remains `0884d747e4d4719a87894bf5e0091a214995f2ddc04a9c551e50914d53e2accb`. Wrapper and embedded remote program pass stdlib AST parsing. Existing CPU environment, fresh admin directory and failure-capture logic are unchanged. This reviewer did not execute either wrapper revision, connect remotely, or run the report.

The prior missing-binding and resolved-key findings are preserved separately in JSON history. Final source gate: `source_ready_for_cpu_report=true`; `unresolved_blocking_source_defects=[]`. Overall WARN and same-family/provisional, actual model/effort UNATTESTED remain unchanged. This review establishes no training approval or retrieval result.
