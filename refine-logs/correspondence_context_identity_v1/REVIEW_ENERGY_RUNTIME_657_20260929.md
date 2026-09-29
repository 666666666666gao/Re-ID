# Energy diagnostic runtime artifact review — 657

Date: 2026-09-29. Reviewer: continuation of `/root/audit_context_energy_657`, native Codex, requested `gpt-6-astra` / `max`; same-family, provisional. The caller's invocation transcription records those arguments and the returned task name, while explicitly excluding independent backend attestation (`logs/context_identity_review_invocations_657_20260929.json:23`).

**Overall: WARN. Local artifact checks: PASS. Reported runtime numerical gates: PASS. Independent binary replay: NOT_PERFORMED. OS exit codes: UNKNOWN.** No contradictory numerical result or new code blocker was found. Both datasets have complete five-condition JSON receipts. The remaining warnings concern evidence coverage and process termination, not a failed recorded parity gate.

This review inspected local artifacts and recomputed local SHA256/JSON/arithmetic checks. It performed no SSH, GPU inference, training, or code changes. The original source-review Markdown and JSON remain byte-identical. References are relative to `C:/Users/gb/.trifusion_github_publish_22c3bee`, except the explicitly absolute launcher path.

## A–F checks

| Check | Status | Evidence and finding |
|---|---|---|
| A. Ground truth provenance | PASS | The runtime output contains activation statistics and same-checkpoint distance consistency, not a new identity-accuracy score. The program derives these values from the normal model output (`tools/diagnose_correspondence_context_energy.py:83`, `:93`, `:111`). Its boundary explicitly calls this official post-selection mechanism diagnosis (`logs/context_energy_RGBNT201_657_20260929/summary.json:805`; same line in the MSVR310 summary). Dataset GT remains the underlying accepted retrieval protocol, not the saved predictions. |
| B. Normalization and arithmetic | PASS, within available summaries | All 120 split/statistic distributions have the expected count, finite means/quantiles, ordered quantiles, and means within extrema. Cosine/norm ranges and 20 ratio-extrema bounds are consistent. The reported ratio is mean per-sample `abs(gain)*norm(correction)/norm(global)`, not the ratio of reported means (`tools/diagnose_correspondence_context_energy.py:23`, `:94`). Cosine/L2 consistency is addressed below. Exact summary recomputation from per-sample tensors was unavailable. |
| C. Files, bindings, and execution evidence | WARN | All 12 JSON files match the archive's recorded SHA256 values (`logs/context_energy_archive_657_20260929.json:7`, `:19`). All 10 per-condition objects equal their summary rows, and all 60 checked dataset/variant/epoch/checkpoint/receipt/distance fields match accepted_667. However, the raw `.pt` files remain remote and no supplied artifact records OS exit codes or measured per-process elapsed times (`logs/context_energy_RGBNT201_657_20260929/static_local.json:155`; `C:/Users/gb/.codex_tmp/launch_context_energy_657.py:38`). |
| D. Reachable computation and reported parity | PASS at receipt level | The code computes the reconstruction maximum and three full saved-distance differences before saving each result (`tools/diagnose_correspondence_context_energy.py:89`, `:108`, `:111`, `:118`). All 10 reconstruction maxima and all 30 distance comparisons are present. Nine checkpoints report zero for all four maxima; RGBNT201/static_local reports two small nonzero distance differences (`logs/context_energy_RGBNT201_657_20260929/static_local.json:11`). These are numeric runtime receipts, stronger evidence than an unevaluated assertion, but not an independent tensor replay. |
| E. Scope and selection | PASS | Both summaries contain the exact five predeclared conditions and bind the same reviewed source and accepted matrix (`logs/context_energy_RGBNT201_657_20260929/summary.json:2`, `:4`, `:23`, `:180`, `:337`, `:494`, `:651`; same structure in the MSVR310 summary). Every statistic has 836 query / 836 gallery samples for RGBNT201 or 591 / 1055 for MSVR310, for each checkpoint. The same ten previously selected best epochs are retained; no RGBNT100 energy result, new epoch search, weighting selection, or new retrieval metric appears. |
| F. Evaluation type and claim ceiling | PASS | The new evidence is a descriptive activation/reproducibility diagnostic (`self_supervised_proxy` taxonomy); the underlying accepted retrieval uses `real_gt`. It supports qualified observations about the recorded embeddings, not causal failure attribution, optimizer-update shares, independent role contributions, or a new retrieval-performance claim (`tools/diagnose_correspondence_context_energy.py:136`; `refine-logs/correspondence_context_identity_v1/ENERGY_DIAGNOSTIC_PLAN_20260929.md:15`). |

## Deterministic local checks

- Recomputed and matched **12/12 archived JSON hashes**, including both summary files. The per-condition JSON content matches the corresponding summary object in **10/10 cases**.
- Checked **60/60 accepted binding fields**: dataset, variant, best epoch, checkpoint SHA, official receipt SHA, and saved-distance SHA. The fixed matrix is `beaeaa1db722df30832ca3d58de2dde1319e61949b356ebe105bd1eff982dae9`; the diagnostic is `dba46b8f73460191318e7ba5d42a4bb191409b3ed639d5bf6eb1f7aff3ffb8e6` (`logs/context_energy_launch_657_20260929.json:4`; both summaries `:4`).
- Both summary source maps match the launch receipt, the archived manifest, and all **9/9 local source bytes**. The reviewed diagnostic and plan also retain their original hashes. The checkpoint protocol/baseline/condition binding continues to use the previously reviewed strict loader (`tools/run_correspondence_context_identity.py:53`); the raw remote checkpoint/protocol bytes were not independently reread by this reviewer.
- Checked **120 unique distributions**: 10 checkpoints × 2 splits × 6 statistics. Their counts imply 8,360 RGBNT201 and 8,230 MSVR310 record-forward instances, **16,590 total across conditions and splits**, not 16,590 unique images.
- For all 20 checkpoint/split combinations, ratio extrema lie inside `abs(gain)*[min(correction)/max(global), max(correction)/min(global)]`, and the mean L2 obeys the second-moment bound from mean cosine. Across the 100 reversed cosine/L2 quantile pairs, the largest residual in `cosine = 1 - L2^2/2` is **1.392440256431371e-7**. This is an aggregate consistency check; quantile interpolation and float arithmetic prevent treating it as exact per-sample reconstruction.

The original source-review hashes remain:

- Markdown: `55bc5128ab7931559d0377276def12395fe6934f9ed13a3e70bb9fb3efed7c70`.
- JSON: `e73e0b25ec1944ececc77f44e9e3436ca6d78fe5c830b9459827e27cd99f973e`.

## What the parity receipts establish

The reported maximum forward reconstruction error is **0.0 for all ten checkpoints**. The largest reported full saved-distance difference is **5.960464477539062e-7**, for RGBNT201/static_local joint_local. That checkpoint's fused difference is **2.384185791015625e-7** and shared-global difference is zero (`logs/context_energy_RGBNT201_657_20260929/static_local.json:11`). All remaining saved-distance maxima are zero. Every reported value is below the predeclared strict `1e-5` gate (`tools/diagnose_correspondence_context_energy.py:108`, `:114`).

The corresponding code subtracts actual recomputed matrices from the saved arrays and records the maximum absolute error; these values are not hardcoded PASS flags. The completion summaries are written only after all five per-condition objects have been produced (`tools/diagnose_correspondence_context_energy.py:128`, `:132`). Thus “the archived runtime receipts report successful full-scope numerical gates” is supported. “An independent reviewer replayed those tensors” is unsupported: the ten remote per-sample `.pt` paths and hashes are recorded, but their bytes are not local, and the original checkpoint/distance tensors were not loaded in this review.

The remote per-sample file contains scalar arrays and protocol records, not the fresh three embedding matrices (`tools/diagnose_correspondence_context_energy.py:115`). Even that scalar file alone would permit recomputing means/quantiles but would not independently reproduce the saved-distance comparisons. No rerun is requested.

## Descriptive numerical findings

Below are query-set means directly checked against each per-condition JSON. In each file, the relative-norm mean is at line 43 and the global/fused cosine mean at line 65. Ratios are amplitudes expressed as a percentage of the global norm, not squared-energy or gradient shares.

| Dataset | Condition | Scaled correction / global norm | Global/fused cosine |
|---|---|---:|---:|
| RGBNT201 | static_none | 6.04705% | 0.998166856 |
| RGBNT201 | context_none | 6.07435% | 0.998148921 |
| RGBNT201 | static_local | 1.10847% | 0.999938245 |
| RGBNT201 | context_local | 1.11009% | 0.999938064 |
| RGBNT201 | context_global | 6.20915% | 0.998063373 |
| MSVR310 | static_none | 47.83138% | 0.878280823 |
| MSVR310 | context_none | 48.96495% | 0.872328934 |
| MSVR310 | static_local | 3.70858% | 0.999310245 |
| MSVR310 | context_local | 3.58760% | 0.999354802 |
| MSVR310 | context_global | 37.14674% | 0.929343581 |

The two local-supervised conditions have smaller correction amplitudes and a fused direction closer to their own shared-global output than the corresponding none conditions at these retained checkpoints. The query amplitude reductions are 81.6692% / 81.7250% on RGBNT201 and 92.2466% / 92.6731% on MSVR310, for static / context comparisons. Those percentages are arithmetic over the recorded means, not newly evaluated retrieval metrics. Checkpoints were selected separately; MSVR310 best epochs also differ. These observations do not isolate the cause of retrieval changes or justify selecting a new gain. All recorded gains are positive, so the separately reported unscaled-correction cosine retains its directional sign here.

## Launch evidence and remaining limits

The additional prelaunch snapshot records **14 COMPLETE, 1 RUNNING, 0 PENDING/FAILED** across the original 15 jobs. The remaining job is RGBNT100/context_global on GPU0 (`logs/context_identity_progress_668_20260929.json:660`); the old M3 campaign is `COMPLETE`, its active list is empty, and GPU1/GPU2 show 23/15 MiB (`:760`). The snapshot's manifest hash matches the archived manifest (`:7`). The supplied launcher independently rechecks allowed queue states, M3 completion, source hashes, and GPU1/GPU2 memory below 500 MiB before spawning (`C:/Users/gb/.codex_tmp/launch_context_energy_657.py:14`, `:18`, `:22`). The launch receipt records GPU1/RGBNT201 PID 3004039 and GPU2/MSVR310 PID 3004040 with exact commands (`logs/context_energy_launch_657_20260929.json:25`, `:43`). These artifacts support the planned nonpreemptive launch conditions; no launch blocker is demonstrated.

**The required OS exit-code/elapsed-time record is missing.** The launcher detaches children with `Popen` and records start metadata, without waiting for or persisting their return codes (`C:/Users/gb/.codex_tmp/launch_context_energy_657.py:38`, `:42`). The supplied summaries and archive contain no terminal process receipt. The launch-receipt-to-summary timestamp spans are 117.983388 s and 122.617075 s for RGBNT201 and MSVR310; they are not measured process runtimes. A completion summary, stdout message, or vanished PID cannot supply the missing OS exit code. Keep it `UNKNOWN`; if an existing independent terminal receipt exists, link it without rerunning the diagnostic.

The plan's “not yet run” line is the preserved preregistration state (`refine-logs/correspondence_context_identity_v1/ENERGY_DIAGNOSTIC_PLAN_20260929.md:3`); current completion evidence is in the new runtime summaries. Preserve both. Same-family/provisional review status and the local-JSON/remote-binary boundary remain. There is no basis here for claiming full three-dataset completion or promotion of a scientific mechanism.
