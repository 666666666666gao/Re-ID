# Fresh full-panel experiment audit — 672 / 2026-09-30

**Overall verdict: WARN.** The nine registered endpoints, all 27 saved full-gallery matrices and the reported table verify. No numerical or registered-artifact mismatch was found. The scientific claim remains a qualified one-seed development result; consistent persistence benefit is not supported.

**Assurance:** requested reviewer `gpt-6-astra`, reasoning `max`, fresh native Codex agent; `review_independence: same-family`; `acceptance_status: provisional`. Actual backend model/version and reasoning attestation are unavailable. The parent preserves the full native prompt/response trace at `C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-30_full_panel_672/`. The native spawn API exposed the canonical task name only; opaque agent ID and actual backend attestation are unavailable.

Evidence roots used below:

- `REPO` = `C:\Users\gb\.trifusion_github_publish_22c3bee`
- `SRC` = `C:\Users\gb\.codex_tmp\cross_depth_source_evidence_668_20260929`
- `AUDIT` = `C:\Users\gb\.codex_tmp\cross_depth_fresh_full_audit_672_20260930`
- Remote primary root = `/data/gaob/Re-ID/Trifusion`.
- All 88 requested primary inputs exist. The JSON report records input and remote-artifact SHA256 values.

## Independent execution

The reviewer wrote `AUDIT/auditor_672a_remote_verify.py` and ran it using `/data/gaob/Re-ID/conda-envs/tri_reid/bin/python` with `CUDA_VISIBLE_DEVICES=` and two CPU threads. It imported NumPy/PyTorch only, not the project collectors, models or score helpers. Main Python and SSH exits were **0**; it ran from 2026-09-30T00:22:37.519965+08:00 to 2026-09-30T00:23:07.678236+08:00 (30.158 seconds). The supplementary checkpoint/baseline verifier also exited **0**. Runtime: Python 3.10.14, NumPy 1.24.4, PyTorch 2.5.1+cu121; the CUDA build was used solely for CPU tensor loading.

The local inventory and table checks exited 0 using the existing uv-managed Python 3.13 executable. An initial bare-Python inventory invocation failed before reading artifacts (`No pyvenv.cfg file`, exit 1); no installation or environment repair was performed. No GPU work, new model construction/inference, training, source/old receipt/manifest/checkpoint edits or data deletion occurred. Only named reports and reviewer-owned audit scripts/text evidence were written.

This replays metrics **from saved distances**. It does not establish a new image→checkpoint→embedding→distance inference chain, nor reproduce the original gradients or unselected-epoch distances.

## Verified fused results

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | mixed_once | 2 | 72.5889 | 74.1627 | 82.8947 | 88.0383 |
| RGBNT100 | mixed_once | 1 | 85.1237 | 95.0437 | 95.7434 | 96.0350 |
| MSVR310 | mixed_once | 15 | 53.0317 | 68.1895 | 82.2335 | 87.6481 |
| RGBNT201 | depth_mean | 2 | 72.5328 | 73.9234 | 82.7751 | 88.0383 |
| RGBNT100 | depth_mean | 1 | 85.1700 | 95.0437 | 95.6851 | 95.9767 |
| MSVR310 | depth_mean | 15 | 52.7232 | 68.6971 | 83.0795 | 87.4789 |
| RGBNT201 | depth_recurrent | 2 | 72.7638 | 74.2823 | 83.1340 | 88.0383 |
| RGBNT100 | depth_recurrent | 1 | 85.1635 | 95.0437 | 95.6851 | 95.9184 |
| MSVR310 | depth_recurrent | 8 | 52.0987 | 67.3435 | 83.0795 | 86.6328 |

All metrics are percentages. Global/local diagnostics were also fully replayed. The maximum difference over all 108 values was **1.4210854715202004e-14 percentage points**. The 36 table values, best epochs and nine displayed comparison numbers match `REPO/results/CROSS_DEPTH_ROLE_STATE_FULL_2026-09-30.md:7` and `:19`; the executable comparison is in `AUDIT/auditor_672a_summarize.py:22`.

## A. Ground-truth provenance and full split coverage: WARN

PASS: all 27,266 protocol rows match filename-derived identity/camera/scene/view labels; complete image catalogue and array metadata order match.

- Live directory path, size and mtime metadata match the supplied catalogue. All protocol image paths exactly cover the JPG files of the registered physical splits. RGBNT201 has two ancillary non-JPG files; they do not enter scoring.
- Train/evaluation identities are disjoint. All 836/1715/591 registered queries have valid positives. Gallery sizes are 836/8575/1055, respectively. MSVR310 retains 103 gallery-only identities.
- RGBNT201 and RGBNT100 exclude only same-identity/same-camera pairs. MSVR310 excludes same-identity/same-time-period pairs; its scene field is that period. Other identities remain distractors.
- This checks catalogue and source semantics, not image bytes, pixel content, visual identity correctness or authenticity of an official image release. No images were downloaded or inspected.

Evidence: `SRC/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:26`, `SRC/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:61`, `SRC/comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py:63`, `SRC/comparators/Signal-cd1b0a6/data/datasets/msvr310.py:67`, `SRC/comparators/Signal-cd1b0a6/utils/metrics.py:68`, `SRC/comparators/Signal-cd1b0a6/utils/metrics.py:137`, `AUDIT/auditor_672a_review_evidence.json:76`.

## B. Independent numerical replay and normalization: PASS

PASS within the saved-distance boundary: all 27 matrices and 108 reported metric values reproduced.

- A reviewer-written NumPy scorer reads every saved matrix on CPU, sorts every full-gallery row using the upstream default NumPy ordering, applies the author filter, computes AP from correct-match ranks and CMC from the first correct rank. It does not import or call the executor collector/scorers.
- All matrix shapes, finite values and six metadata vectors match the complete protocols. Author float32 CMC accumulation is reproduced, with float64 count-based values also retained. Maximum absolute discrepancy is 1.4210854715202004e-14 percentage points.
- Raw unit-interval metrics and ordinary x100 percentage scaling are recorded. L2 embedding normalization followed by squared Euclidean distance is standard feature processing, not normalization by a model-specific performance maximum.
- Saved distances range from -1.1324882507324219e-6 to 3.7508623600006104. Small negative self-distances are float32 cancellation in the documented squared-distance computation; values were neither clipped nor rescaled by the audit.
- This is numerical replay of saved distances. It does not regenerate embeddings or distances from checkpoint/image inference.

Evidence: `AUDIT/auditor_672a_remote_verify.py:144`, `AUDIT/auditor_672a_remote_journal.jsonl:14`, `SRC/tools/run_official_three_dataset_roles.py:230`, `SRC/tools/run_correspondence_context_identity.py:207`, `SRC/tools/run_correspondence_context_identity.py:215`, `SRC/comparators/Signal-cd1b0a6/utils/metrics.py:93`, `SRC/comparators/Signal-cd1b0a6/utils/metrics.py:153`.

## C. Completion, selection, checkpoints, M0 and scalar evidence: WARN

PASS for extant artifacts, recorded histories, hashes, schemas and arithmetic; original execution is not newly replayed.

- Nine registered endpoints contain all epochs 1-50; 450 history entries match original train stdout. Every selected epoch is the maximum recorded official fused mAP with latest-epoch tie handling; all observed maxima are unique. All 36 rounded public table metrics, nine best epochs and nine displayed comparison numbers match.
- All 54 supplied local full/M0/campaign text artifacts are byte-identical to their remote originals. The live parent campaign is COMPLETE at 2026-09-30T00:03:04.978340+08:00, all nine child exits are zero, and all 27 m0/train/evaluate exits are zero. A ps check at 00:23:07 found none of the recorded PIDs or cross-depth runner/queue commands active.
- All nine best checkpoints and nine M0 probes load on CPU. Schema, dataset, seed, depth_mode, variants, condition, baseline and protocol hashes, epoch, metrics, state keys/shapes and finite values pass. Each has 131 stored tensors. Independent schema accounting yields 118 trainable tensors and 2,617,345/2,431,489/2,592,769 trainable parameters for RGBNT201/RGBNT100/MSVR310.
- The nine eight-batch M0 receipts report nonzero-gradient observations covering all 118 trainable tensors, unchanged frozen Signal and zero reload difference; the active training code asserts gradient finiteness before each update. Probe hashes and eight step records per run match. This is evidence of prior M0 execution; no fresh gradients or model reload forward were performed.
- All 30,624 formal and 72 M0 scalar records are finite, sequential and consistent with epoch means. Loss equals identity plus triplet plus inactive auxiliary loss within 2.384185791015625e-7. Auxiliary loss is exactly zero throughout. Each RGBNT100 condition has 6,559 zero-triplet steps, so its observed objective contribution is identity-only.
- Within each dataset, recorded initial tensor-state hashes, parameter counts, 50 epochs, total steps, learning rate and weight decay agree across all three conditions and M0/full starts. Initial model snapshots are not saved, so matching initializations are verified as matching records plus seeded build code, not independently reconstructed initial models.
- Checkpoint-to-distance hashes and the active strict-reload code support the binding. The arrays contain metadata vectors, not per-image paths or embeddings. Unselected epoch distances, original gradients and final frozen backbone tensors are not stored in these checkpoints; their execution-level truth cannot be independently reconstructed under the no-inference boundary.

Evidence: `SRC/tools/run_cross_depth_role_state.py:35`, `SRC/tools/run_cross_depth_role_state.py:44`, `SRC/tools/run_correspondence_context_identity.py:112`, `SRC/tools/run_correspondence_context_identity.py:140`, `SRC/tools/run_correspondence_context_identity.py:147`, `SRC/tools/run_correspondence_context_identity.py:175`, `SRC/tools/run_correspondence_roles.py:111`, `AUDIT/auditor_672a_review_evidence.json:314`, `AUDIT/auditor_672a_review_evidence.json:538`, `AUDIT/auditor_672a_supplement_result.json:130`.

## D. Active code, condition meaning and source coverage: WARN

PASS for the 205 launch-bound bytes, active dispatch and current supplemental bytes; execution-time dependency closure remains incomplete.

- The depth entry replaces the model class, build, evaluator, save and reload functions; the context entry then patches the shared runner. Active training calls the official fused scorer each epoch and final evaluation calls both the upstream author scorer and the separate project parity scorer for all three paths.
- All 205 launch-bound source/config/protocol files match the local snapshot and live remote bytes, including a second remote hash check after replay. The separately bound paired-analysis helper also matches its recorded SHA256.
- Three imported dependencies were not in the launch manifest: tools/train_msvr310_trifusion_oof.py, tools/train_official_three_dataset_roles.py and tools/audit_v17_full_gallery.py. Their current supplemental hashes match, but later capture cannot prove their execution-time bytes. Package version metadata is not a historical binary attestation.
- mixed_once averages normalized depth evidence before the nonlinear role processing. depth_mean independently processes depths and averages outputs. depth_recurrent carries role outputs into the next depth and returns only the final output. Thus recurrence and final-depth-versus-mean aggregation change together.
- The backbone is one shared frozen CLIP visual encoder evaluated for three modalities with trainable role adapters at blocks 4/8/12. Adapter outputs alter downstream shared_global; frozen Signal weights do not make shared_global an independent fixed baseline. Address generation is structurally common and shared across depths, not guaranteed numerically identical across separately trained endpoints.
- M3/teacher prediction, auxiliary identity, static-context, old generic role-training objectives and the imported legacy full_gallery_scores path are dormant for this panel. Their presence is not evidence of a performed performance evaluation. The actual retrieval metric functions are called.

Evidence: `SRC/tools/run_cross_depth_role_state.py:73`, `SRC/tools/run_correspondence_context_identity.py:249`, `SRC/tools/run_official_three_dataset_roles.py:17`, `SRC/tools/train_signal_preserving_v18.py:16`, `REPO/logs/cross_depth_additional_sources_670_20260929.json:6`, `REPO/logs/cross_depth_additional_sources_670_20260929.json:28`, `SRC/modeling/trifusion/cross_depth_role_state.py:49`, `SRC/modeling/trifusion/cross_depth_role_state.py:66`, `SRC/modeling/trifusion/correspondence_roles.py:64`, `SRC/modeling/trifusion/correspondence_context_identity.py:87`.

## E. Registered scope, cost and scientific claims: WARN

PASS for full registered panel coverage and the current qualified table; consistent persistence, untouched-test and multi-seed claims are unsupported.

- The full preregistered scope is three conditions by three datasets at second-stage seed 42, with 450 second-stage epochs. There are no omitted registered endpoints and no evidence of substituted seed/retry rows in this panel.
- The reused baseline weights bind to three earlier local fixed-final runs at upstream seed 1234, with RGBNT201/MSVR310 trained for 50 epochs and RGBNT100 for 30. Their original and copied checkpoint hashes match; CPU recomputation of all three tensor-state hashes matches the initializer records. These are dataset-trained module-free baselines, not untouched pretrained CLIP or author-release weights.
- Second-stage training plus epoch-evaluation receipt time totals 29,545.260236 seconds (8.2070167322 summed hours). Complete child m0/train/evaluate wall times total 30,576.126899 seconds (8.4933685831 summed hours). Upstream baseline training adds 4,101.664515 seconds; with its final evaluation, 4,251.812489 seconds. These are recorded wall times, not measured GPU utilization or isolated FLOPs. Prior method search and original CLIP pretraining are excluded.
- Recurrent minus mean fused mAP/Rank-1 is +0.2310896/+0.3588498 points on RGBNT201, -0.0064717/0 on RGBNT100 and -0.6245549/-1.3536394 on MSVR310. Rank-1 repairs/new errors are 3/0, 0/0 and 16/24. The registered consistent-positive criterion is not met.
- The official evaluation labels guide all 50 epoch selections and prior method choices. No untouched test, training-seed variance or broad generalization is established. Any identity-level resampling would concern these fixed selected models only.
- The current result document already states the aggregation confound, dependent global/local diagnostics, official selection and image/source provenance limits. It is suitable as a qualified development result, not as proof of reliable recurrent-role benefit. Its pending-audit status and the tracker can be replaced with a link to this bounded WARN report.

Evidence: `REPO/refine-logs/cross_depth_role_state_v1/EXPERIMENT_PLAN.md:23`, `REPO/refine-logs/cross_depth_role_state_v1/EXPERIMENT_PLAN.md:25`, `REPO/refine-logs/cross_depth_role_state_v1/EXPERIMENT_PLAN.md:35`, `REPO/results/CROSS_DEPTH_ROLE_STATE_FULL_2026-09-30.md:17`, `REPO/results/CROSS_DEPTH_ROLE_STATE_FULL_2026-09-30.md:23`, `REPO/tools/queue_signal_plain_baseline.py:60`, `REPO/tools/queue_signal_plain_baseline.py:70`, `AUDIT/auditor_672a_review_evidence.json:341`, `AUDIT/auditor_672a_supplement_result.json:5`, `AUDIT/auditor_672a_supplement_result.json:270`.

## F. Evaluation type and proxy/phantom-result check: PASS

PASS at the dataset-catalogue label level; M0 and hash/schema checks are engineering diagnostics.

- Retrieval identities and camera/period exclusions come from dataset records independently cross-checked against filenames. Predictions supply only distances, not targets. No self-normalized performance denominator, substituted model-derived GT or phantom registered result was found.
- The M0 gradient/reload probes, tensor hashes and synthetic operator checks are engineering consistency evidence, not ReID performance. Dormant M3 teacher targets are not used by the active context/depth objective.
- real_gt describes the source of evaluated labels. It does not remove the catalogue-versus-image-authenticity limitation in A or the official-selection limitation in E.

Evidence: `SRC/tools/official_three_dataset_data.py:8`, `SRC/tools/run_correspondence_context_identity.py:207`, `SRC/tools/run_correspondence_context_identity.py:112`, `SRC/modeling/trifusion/correspondence_context_identity.py:92`, `AUDIT/auditor_672a_review_evidence.json:76`.

## Registered primary comparison

| Dataset | Recurrent − mean mAP | Recurrent − mean R1 | R1 repairs / new errors | Improved / worsened query identities |
|---|---:|---:|---:|---:|
| RGBNT201 | +0.231090 | +0.358850 | 3 / 0 | 21 / 4 of 30 |
| RGBNT100 | -0.006472 | +0.000000 | 0 / 0 | 25 / 22 of 50 |
| MSVR310 | -0.624555 | -1.353639 | 16 / 24 | 25 / 26 of 52 |

Identity changes use a ±1e-6 percentage-point threshold for the fixed-model mean AP difference. These are post-selection diagnostics, not repeat training or independent generalization tests. No bootstrap or significance claim is added.

## Claim impact and follow-up

The complete qualified development table is supported. Reliable cross-dataset persistence, isolation of state carry, independent global/local training controls, untouched-test generalization and multi-seed stability are not supported. The positive RGBNT201 endpoint must not replace the negative/flat endpoints.

- Replace only the pending-audit label in the current result/tracker with this WARN report and same-family/provisional status; retain all nine results and existing qualifications.
- Describe seed 42 as the second-stage development seed and include the reused seed1234 baseline stage and measured wall-time boundary in cost descriptions.
- Do not claim reliable persistence, isolated recurrence, independent global/local controls, untouched-test generalization or multi-seed stability.
- Treat absent launch-time dependency/pixel/initial-state evidence as historical limits; do not rewrite old manifests or receipts to imply retroactive proof.

## Primary evidence hashes

- Supplied matrix: `5bbe27b41feb9a4b2ea5f863d07e18159cce934c191fcad2e7eeb8c55ae51aec`.
- Full CPU replay result: `e4afab1238c2a343612e191784ac7cf4b5a88759276d73af5bcdace654c264fd`.
- Supplementary baseline/schema result: `723bc807975c55a6e2a672ad10cf35651ab0dd43c52903cec6f8fc9de62e96cc`.
- Compact claim evidence: `7b340203bec93392c038158f9f22af5e7ca3acebc63962c9d6b734b5d0447058`.
- Main CPU verifier: `bf286e80e667850af57ea5934db197583c871dbc93d9f4980171f8b202d3e170`.
- Supplementary CPU verifier: `07e46fa025932bc7fc298c14eeb94249a83faaa12733d0693f545dd0eaab1750`.
- The remote final matrix has a different collection timestamp/hash from the supplied later snapshot; its nine row objects match exactly. This is recorded in `AUDIT/auditor_672a_review_evidence.json:314`.

No required requested primary artifact is absent. The limitations above are assurance and experimental-design limits, not fabricated failures or excuses for missing arithmetic checks.
