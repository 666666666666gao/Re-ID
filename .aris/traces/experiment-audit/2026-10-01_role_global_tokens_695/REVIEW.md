# Experiment integrity audit

**Overall verdict: WARN — the completed panel is supported; advancement is not.**

Auditor: **gpt-6-astra**, reasoning **max**. Fresh, read-only review; `review_independence: same-family`; `acceptance_status: provisional`.

Audited at **2026-10-01 02:52:39 UTC**. Remote paths below are relative to `/data/gaob/Re-ID/Trifusion`.

| Check | Status | Finding |
|---|---|---|
| A. Ground truth | PASS | Dataset filename labels, complete query/gallery populations, and original exclusion rules agree. |
| B. Metric normalization | PASS | Raw percentage mAP/CMC; no normalization by prediction maxima. |
| C. Results and provenance | PASS | Nine terminal endpoints, 27 child exits, full50 histories, selected checkpoints, arrays, report numbers, and source bindings agree. |
| D. Evaluator execution path | PASS | The actual entry point calls the original scoring functions; reported diagnostics trace to called analysis functions. |
| E. Scope and interpretation | WARN | One seed and official-best selection support a descriptive comparison. Training trajectories are not bitwise reproducible. All six advancement comparisons fail. |
| F. Evaluation classification | PASS | Retrieval and label-based diagnostics are `real_gt`; M0 and the structural fixture remain engineering evidence. |

## A. Ground-truth provenance: PASS

I independently parsed labels from every referenced filename, checked every modality path, compared complete directory membership with each protocol, checked train/test identity disjointness, and reconstructed every query’s valid-positive/exclusion count.

| Dataset | Train | Query | Gallery | Query/gallery identities | Gallery-only distractors |
|---|---:|---:|---:|---:|---:|
| RGBNT201 | 3,951 | 836 | 836 | 30 / 30 | 0 |
| RGBNT100 | 8,675 | 1,715 | 8,575 | 50 / 50 | 0 |
| MSVR310 | 1,032 | 591 | 1,055 | 52 / 155 | **103 identities; 464 records** |

All **27,266 protocol records** and **3,142 query masks** passed. Every query has a valid positive. MSVR310 excludes the same identity **and scene**, while RGBNT201/RGBNT100 exclude the same identity **and camera**. Other identities remain in the gallery.

Evidence:

- `comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:61–85`
- `comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py:63–84`
- `comparators/Signal-cd1b0a6/data/datasets/msvr310.py:67–87`
- `tools/build_official_three_dataset_protocols.py:32–80`
- `tools/official_three_dataset_data.py:8–18`
- `comparators/Signal-cd1b0a6/utils/metrics.py:68–108,125–170`
- `tools/run_correspondence_context_identity.py:207–224`

These are dataset labels, not labels inferred from model predictions. I verified the available dataset files and labels; I did not rehash all image contents or independently reconstruct the upstream checkpoints’ original training history.

## B. Metric normalization: PASS

The distance function L2-normalizes embeddings and computes squared Euclidean distances. The evaluators calculate AP from relevant-gallery ranks and CMC from valid-query counts. Multiplication by 100 produces ordinary percentage metrics.

I independently ranked all **27 saved matrices**, reproducing all **108 aggregate metrics** within **2.8518968804291944e-6 percentage points**. All **28,278 per-query/path AP and first-match-rank entries** matched exactly.

The small aggregate CMC differences are consistent with the author evaluator’s float32 aggregation versus the independent count-based calculation. They are far below the registered `1e-5` percentage-point tolerance.

Evidence:

- `tools/run_official_three_dataset_roles.py:230–238`
- `comparators/Signal-cd1b0a6/utils/metrics.py:93–108,153–170`
- `tools/train_msvr310_signal_oof.py:223–238`
- `tools/train_rgbnt100_signal_oof.py:253–268`
- `tools/run_correspondence_context_identity.py:214–228`

No reported performance metric divides by the model’s own maximum, minimum, or mean prediction.

## C. Result existence, arithmetic, and provenance: PASS

The following checks succeeded:

- Parent campaign: **9/9 COMPLETE**, nine recorded worker exit codes zero.
- Child campaigns: **27/27 COMPLETE**, with actual `wait()`-derived exit codes zero.
- All 27 complete subprocess logs agree with their raw receipts and the 10:30 snapshot.
- Nine full histories contain epochs **1–50**. All **30,624 formal steps** and **72 M0 steps** were parsed.
- Formal step counts per condition: RGBNT201 **2,649**, RGBNT100 **6,559**, MSVR310 **1,000**.
- Every epoch mean loss matches its raw steps exactly. Maximum step loss-reconstruction error: **2.384185791015625e-7**.
- All nine selected checkpoints are the last exact fused-mAP maximum: epoch **2 / 1 / 10** for RGBNT201/RGBNT100/MSVR310 respectively.
- All nine best checkpoints, nine M0 checkpoints, nine distance files, and three baseline checkpoint files were loaded or hashed on CPU.
- Each role checkpoint contains **146 state entries**. Applying the source-defined frozen-parameter/buffer rules yields **128 trainable tensors** and the reported parameter count.
- All **215 frozen source hashes**, **57 report-bound raw-artifact hashes**, five reused-analysis source hashes, and both figure hashes match.
- Every Markdown table value, reported cost, paired statistic, repair/new-error count, identity aggregate, bootstrap interval, and gate agrees with the underlying artifacts.
- All **900 SVG trajectory points** match the corresponding raw histories under their common axis transformations.

Parameter counts are equal across the three conditions within each dataset: RGBNT201 **2,871,242**, RGBNT100 **2,685,386**, MSVR310 **2,846,666**. Full-initial-state hashes also agree within each dataset.

The measured campaign wall time is **9,385.471587 seconds**; summed endpoint wall time is **30,026.496473 seconds**, or **8.340693 hours**. These intervals exclude upstream ReID training and are not a measurement of total GPU utilization or equal computational cost.

Evidence:

- `tools/queue_role_global_tokens.py:91–129`
- `tools/queue_correspondence_refinement.py:135–181`
- `logs/role_global_tokens_20261001_v1/campaign.json:2–639`
- `logs/role_global_tokens_20261001_v1/accepted_matrix.json:5–8`
- All nine `..._seed42_full/training.json:53–657`
- All nine `..._seed42_m0/training.json:53–67`
- `tools/run_correspondence_context_identity.py:95–171`
- `tools/run_role_global_tokens.py:52–68`
- `tools/report_role_global_tokens_complete.py:54–105,109–156`
- `results/role_global_tokens_complete_20261001/SUMMARY.json:138325–138405`
- `results/role_global_tokens_complete_20261001/REPORT.md:7–34`

**Reload limitation:** I verified checkpoint contents, metadata, keys, shapes, source-defined strict loading, and historical successful receipts/logs. I did not instantiate the production model or repeat its forward pass. Likewise, equality of fresh initial states is supported by recorded full-state hashes and the construction path; no preserved initial-state tensor snapshot was independently regenerated during this audit.

## D. Actual evaluator path: PASS

The active chain is:

`run_role_global_tokens.main` → patched context runner → `run_correspondence_context_identity.train/evaluate`.

Training calls `runner.official_metrics`; final evaluation calls the original `eval_func` or `eval_func_msrv`, alongside the independent camera/scene scorer. The global-token wrapper then adds metadata to the completed retrieval receipt.

The report calls `audit`, `analyze`, and `compare`; their outputs appear in the summary and independently reproduce from raw steps and arrays.

Evidence:

- `tools/run_role_global_tokens.py:71–92`
- `tools/run_correspondence_context_identity.py:24–29,138–142,175–240,244–254`
- `tools/run_correspondence_roles.py:79–108`
- `tools/report_role_global_tokens_complete.py:15–17,64–65,92`
- `logs/role_global_tokens_20261001_v1/analysis_waiter_status.json:9–24`
- `logs/role_global_tokens_complete_analysis_20261001.log:1`

Old evidence-readout entry points, author visualization utilities, and unused metric wrapper classes are not the execution path for these results. No claim depends on those inactive functions.

I did **not** invoke the author MSVR scorer during auditing because it writes `re.txt`; the CPU verification independently implemented its ranking equation.

## E. Scope and registered decision: WARN

The completed scope is exactly **three datasets × three conditions × seed42**, with fresh full50 training. The report appropriately states the differences in effective capacity, direct-path computation, and shared projection/readout use.

The registered rule remains unchanged:

- Token must exceed both controls in mAP on all datasets.
- Rank-1 must not decrease.
- RGBNT201 and MSVR310 must gain at least **0.5 mAP** over each control.

Independent paired results:

| Dataset | Control → token | ΔmAP, pp | ΔRank-1, pp | Repairs / new errors | Gate |
|---|---|---:|---:|---:|---|
| RGBNT201 | static | +0.031294 | −0.119617 | 1 / 2 | FAIL |
| RGBNT201 | direct | +0.040080 | 0 | 2 / 2 | FAIL |
| RGBNT100 | static | +0.014255 | −0.058309 | 0 / 1 | FAIL |
| RGBNT100 | direct | +0.015800 | −0.116618 | 0 / 2 | FAIL |
| MSVR310 | static | +0.002776 | +0.169205 | 8 / 7 | FAIL |
| MSVR310 | direct | +0.053516 | 0 | 7 / 7 | FAIL |

All six identity-bootstrap intervals include zero. They describe resampling identities for selected fixed models; they do not quantify training-seed variance or untouched-test significance.

**Observed reproducibility limitation:** A supplementary exact-equality check of each M0 run’s first eight scalar steps against its corresponding fresh full run failed for **all nine endpoints**. The first batch is exactly equal in every case; subsequent differences reach **0.00014925003051757812 loss units**. Exact trajectory equality was not a registered acceptance condition, so this does not invalidate M0 or the stored retrieval results. It does prevent a claim of bitwise training reproducibility. The cause was not localized.

Evidence:

- `refine-logs/role_global_tokens_v1/EXPERIMENT_PLAN.md:13–39`
- `modeling/trifusion/role_global_tokens.py:45–58,79–94`
- `tools/report_role_global_tokens_complete.py:88–102,138–144`
- `results/role_global_tokens_complete_20261001/REPORT.md:19–33`
- Corresponding full/M0 `training_steps.jsonl:1–8`
- `tools/run_signal_preserving_v5.py:257–267`

The small positive mAP differences are valid descriptive observations. Stable improvement, a causal interaction advantage, training robustness, three-role necessity, novelty, and SOTA remain unsupported.

## F. Evaluation classification: PASS

| Evidence | Classification | Permitted interpretation |
|---|---|---|
| Fused official retrieval | `real_gt` | Selected-checkpoint performance under the documented protocol |
| Global/correction retrieval, repairs, AP, identity bootstrap | `real_gt`, post-selection diagnostics | Descriptive within-checkpoint or fixed-model comparisons |
| Eight-batch M0 | Real-label training engineering check | Cumulative gradient support, finite training, frozen-state and reload checks |
| CPU linear-Mamba fixture | `simulation_only` structural fixture | Shapes, initial equality, and wiring sensitivity |
| Loss and timing summaries | Operational diagnostics | Scalar task accounting and measured process intervals |

M0’s **128/128** figure means cumulative nonzero-gradient support across eight batches—not every tensor having a nonzero gradient on every batch.

`joint_local` contains the total correction, including global conditioning/projection in the relevant modes. `shared_global` is jointly adapted and is not a separately trained global-only control.

Evidence:

- `tools/check_role_global_tokens_cpu.py:19–66`
- `refine-logs/role_global_tokens_v1/PREFLIGHT_20261001.json:18–75`
- `tools/run_correspondence_context_identity.py:122–166`
- `modeling/trifusion/role_global_tokens.py:79–94`
- `results/role_global_tokens_complete_20261001/REPORT.md:29–33`

## Actions and claim impacts

1. Archive the panel as **complete, advancement gate FAIL**, retaining all nine endpoints and all six comparisons. No numerical or scoring correction is indicated.
2. Preserve the M0/full scalar discrepancy in the audit record. Describe initialization equality as hash-witnessed; do not claim identical training trajectories.
3. Keep official-best, one-seed, effective-capacity, upstream-cost, and diagnostic-output qualifications attached to any derived claims.

Supported: completed full50 panel; reported selected-checkpoint metrics; descriptive tiny mAP differences; complete negative advancement decision.

Unsupported: stable/causal interaction improvement, significance, pure-local identity evidence, independent global-only comparison, or SOTA.

## Actual operations and errors

All remote operations used the authorized connection:

```text
ssh -i C:/Users/gb/.ssh/id_ed25519 -p 2026
    -o ProxyCommand=none -o BatchMode=yes -o ConnectTimeout=10
    gaob@172.19.12.138
    /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -
```

Read-only stdin scripts performed source reads, full JSON/JSONL parsing, SHA-256 verification, directory/filename checks, CPU checkpoint loading, independent NumPy ranking, and SVG inspection. Third-party imports used `sys.dont_write_bytecode=True`; no project analysis entry point was executed.

Recorded check results:

- Artifact/scalar stage: **640/649 passed**; nine supplementary exact M0/full scalar-equality checks failed as documented.
- Dataset/label stage: **84,958/84,958 passed**.
- Binary/ranking stage: **430/430 passed**.
- Additional report checks: **68/68 passed**.
- All 27 full stage logs matched; all 900 plotted points matched.

One diagnostic stdin script initially exited with a **SyntaxError at line 8** due to a missing dictionary brace. It executed no checks; the corrected read-only command completed successfully. Some early large text displays were truncated; affected source sections were reread and large raw artifacts were subsequently checked through complete machine parsing. The local PNG was absent; the remote PNG was hash/header checked and the remote SVG’s data points were verified. No raster visual inspection is claimed.

No files were saved or edited by this reviewer. No training, GPU/model forward, package installation, report rerun, or other server access occurred.

