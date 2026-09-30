# Experiment Audit Report — prompt_role_state_20260930

Date: 2026-09-30T02:07:39.778810+00:00. Auditor: fresh native Codex agent `/root/prompt_panel_audit`.
Review independence: **same-family**. Acceptance status: **provisional**.
The child host does not expose its model/effort identifier; the executor's native spawn record is authoritative. Required skill route: `gpt-6-astra`, `max`. No cross-family acceptance is claimed.

## Overall verdict: WARN

**No metric fabrication, self-normalized metric, missing formal endpoint, or mismatched primary number was found.** Deterministic replay accepted all six formal endpoints: 18 full-gallery matrices, 72 metric values, 20,416 formal steps, 48 M0 steps, 300 formal epoch events, and six M0 epoch events. Maximum replay disagreement after matching the upstream CMC float32 representation is 1.4210854715202004e-14 percentage points. Independent float64 CMC differs by at most 2.9659727545094938e-06 pp, explained by the upstream float32 CMC reduction.

The scientific result is mixed. Carry loses RGBNT201 mAP, gains only 0.0943 pp on RGBNT100, and gains 1.1773 pp on MSVR310. It **fails the preregistered cross-dataset progression gate**. This is a negative experimental outcome for that gate, not an integrity FAIL.

Warnings concern one seed, official evaluation labels used for 50-epoch checkpoint selection and earlier method development, same-checkpoint diagnostics being noncausal, the stale tracker snapshot supplied for review, and unused legacy evaluation helpers. The full distance-matrix replay succeeded independently of every project/author scorer. Fresh CPU model construction also reproduced all six initial-state hashes and strictly loaded all six M0 probes plus six selected checkpoints.

## Evidence and execution scope

Primary backend: `/data/gaob/Re-ID/Trifusion` on `172.19.12.138:2026`. All source/result access was read-only; no training/GPU job or project/source/result edit was performed. A later explicit executor authorization allowed library temporary files only under `/data/gaob/Re-ID/prompt_role_state_audit_20260930_tmp`, outside the repo. `remote_replay.py` was streamed over SSH with `CUDA_VISIBLE_DEVICES=` and `PYTHONDONTWRITEBYTECODE=1`. It imports only stdlib, NumPy, and PyTorch, and does not import the collector, model, project metrics, or upstream metrics. It computes AP directly from relevant-item ranks and CMC from the first relevant rank.

* `snapshot/`: original-byte mirror from the backend, plus `local_docs/` for the supplied plan/tracker/handoff.
* `input_hashes.json`: 279 collected original inputs, 15,251,186 bytes, each with SHA-256.
* `cpu_replay.stdout.json`: complete independent results, per-query AP/first-positive-rank data, and 267 backend artifact hashes.
* `cpu_replay_summary.json`: same evidence without long per-query vectors.
* `text_checks.json`: 207 manifest source hashes rechecked; 246 snapshot inputs agree with the later remote replay; 35 executor-provided local text copies/matrix copies agree with independently fetched backend bytes; 306 log epoch events exactly match the corresponding receipts.
* `remote_replay.py`, `run_remote_replay.py`, `collect_readonly.py`, `text_checks.py`: reviewer-owned reproducible scripts.
* `strict_model_cpu_{RGBNT201,RGBNT100,MSVR310}.stdout.json`, `strict_model_cpu.py`: fresh real model CPU construction, all six initial hashes, and 12 strict prompt-state loads; three baseline strict loads also passed.

All result tables below are in percent; changes are percentage points. JSON retains full numeric precision.

## A. Ground-truth provenance: PASS

The retrieval identities, cameras, and scenes are dataset-derived metadata, not targets synthesized from model predictions. The actual fixed protocols are `logs/official_three_dataset_protocols_20260923/{RGBNT201,RGBNT100,MSVR310}.json`; the initially suggested `refine-logs/correspondence_roles_v1/protocols` path is not the active location. `tools/queue_correspondence_roles.py:19` binds the active protocol root.

For every train/query/gallery record, the auditor parsed IDs and environments independently from the on-disk file names; checked each referenced image exists and is nonempty; compared the full set of referenced paths with every image in the corresponding dataset split; checked train relabeling; and checked train/evaluation identity disjointness. All checks passed. Evaluation metadata stored beside each distance matrix exactly matches every protocol row in order. The audit did not re-download or independently certify the dataset release; provenance here is the installed dataset and pinned parser/protocol.

| Dataset | Train records / IDs | Query records / IDs | Gallery records / IDs | Filter |
| --- | --- | --- | --- | --- |
| RGBNT201 | 3951 / 171 | 836 / 30 | 836 / 30 | camera |
| RGBNT100 | 8675 / 50 | 1715 / 50 | 8575 / 50 | camera |
| MSVR310 | 1032 / 155 | 591 / 52 | 1055 / 155 | scene |

RGBNT201 uses the same 836-record official `test` set for query and gallery; self/same-identity-same-camera items are removed. MSVR310 `query3` contains 52 query identities and the full gallery contains 155 identities; a full-gallery claim does not imply all 155 identities occur in the query subset. None of the 836/1715/591 queries was skipped in any path.

Exact source evidence:

* `comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:26`, `:61`, `:79`: official train/test splits, paired modality paths, file-name identity/camera parser.
* `comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py:29`, `:63`, `:76`: official splits and identity/camera parser.
* `comparators/Signal-cd1b0a6/data/datasets/msvr310.py:33`, `:74`, `:81`: `query3`, paired paths, identity/camera/scene parser.
* `tools/official_three_dataset_data.py:8`: paths and labels come from protocol records; evaluation uses `identity`, training uses `label`.
* `tools/run_correspondence_context_identity.py:207`: metadata comes directly from complete query/gallery protocol records.
* `comparators/Signal-cd1b0a6/utils/metrics.py:68`, `:137`: same-ID/same-scene filter for MSVR310 and same-ID/same-camera filter for RGBNT201/RGBNT100. All different-identity gallery items are retained.
* `remote_replay.py:67` onward: independent filesystem/metadata checks; `:185` onward: matrix metadata binding and independent scoring.

## B. Score normalization: PASS

Reported mAP is mean AP times 100; AP divides by the number of valid same-identity relevant gallery items. CMC divides valid-query rank-hit counts by the number of valid queries. No metric denominator depends on the prediction maximum/minimum/mean. Feature L2 normalization is a legitimate distance-definition step, and attention softmax/layer normalization operate inside the model; neither rescales the reported metric to its own best output.

Evidence: upstream AP/CMC arithmetic at `comparators/Signal-cd1b0a6/utils/metrics.py:93`, `:104`, `:153`, `:166`; metric percent conversion at `tools/run_correspondence_context_identity.py:222`; feature normalization/squared Euclidean distance at `tools/run_official_three_dataset_roles.py:230`; fused/local feature definitions at `modeling/trifusion/correspondence_context_identity.py:92`. The independent formula is `remote_replay.py:35`.

## C. Result existence, binding and numbers: PASS for formal artifacts; WARN for supplied tracker freshness

All six full runs and six separate M0 runs exist. Each child campaign records `m0 → train → evaluate`, all COMPLETE/exit 0; the parent campaign is COMPLETE. Six official receipts, six single `best_map.pth` checkpoints, six `official_distances.pt` files, six M0 probe checkpoints, all 12 step logs and training receipts, and all six 50-epoch training logs were inspected. Every full run has exactly one `.pth` checkpoint, selected on fused mAP with later-epoch tie preference. All four displayed metrics come from that one selected checkpoint. Checkpoint/protocol/baseline/receipt/distance SHA bindings match actual bytes. Each checkpoint/probe has 135 finite state entries with identical key/shape contracts. These checkpoints intentionally omit frozen Signal state, which is reloaded from its bound pure-ReID baseline.

| Dataset | Mode | Best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | --- | --- | --- | --- | --- |
| RGBNT201 | reset_roles | 18 | 71.264989466539 | 72.009569406509 | 81.459331512451 | 86.004781723022 |
| RGBNT100 | reset_roles | 1 | 85.648576943955 | 95.860058069229 | 96.618074178696 | 96.851313114166 |
| MSVR310 | reset_roles | 15 | 53.049875858170 | 69.035530090332 | 83.079528808594 | 87.648051977158 |
| RGBNT201 | carry_roles | 30 | 70.650920385859 | 72.368419170380 | 81.578946113586 | 86.602872610092 |
| RGBNT100 | carry_roles | 1 | 85.742869108470 | 96.326529979706 | 96.443146467209 | 96.559768915176 |
| MSVR310 | carry_roles | 24 | 54.227190318632 | 70.219963788986 | 84.940779209137 | 88.494080305099 |

| Dataset | carry − reset mAP (pp) | carry − reset Rank-1 (pp) | carry − reset Rank-5 (pp) | carry − reset Rank-10 (pp) |
| --- | --- | --- | --- | --- |
| RGBNT201 | -0.614069080679 | +0.358849763870 | +0.119614601135 | +0.598090887070 |
| RGBNT100 | +0.094292164515 | +0.466471910477 | -0.174927711487 | -0.291544198990 |
| MSVR310 | +1.177314460462 | +1.184433698654 | +1.861250400543 | +0.846028327942 |

The plan is explicitly a pre-result design document. The supplied tracker snapshot still says approximately 01:03, four M0s passed, two endpoints PENDING, and no terminal result (`local_docs/refine-logs/prompt_role_state_v1/EXPERIMENT_TRACKER.md:3` and `:7`). This stale snapshot does not invalidate the later backend receipts but must be refreshed before presenting it as the current state. The reviewed handoff's final prompt-panel section likewise describes that earlier running snapshot; no completed-panel performance claim was found there to contradict the artifacts.

Source evidence: `tools/run_correspondence_context_identity.py:93`, `:137`, `:140`, `:147`, `:175`, `:188`, `:229`; prompt checkpoint schema and strict key-set/load contract at `tools/run_prompt_role_state.py:37` and `:47`; exclusion of frozen state at `tools/run_correspondence_roles.py:111`; child sequencing at `tools/queue_prompt_role_state.py:83`; terminal collection at `:170`. Every formal `official_metrics.json:6` names the selected epoch, `:16` begins the metrics; each `training.json:3` records completion. The reviewer script independently verifies those values against binary checkpoint contents and every per-epoch history row, not just against the project's accepted matrix.

### M0, training budget, pairing, source freeze and strict-reload limits

| Dataset | Mode | Formal epochs / steps | Triplet nonzero steps | M0 gradient tensors | Trainable elements |
| --- | --- | --- | --- | --- | --- |
| RGBNT201 | reset_roles | 50 / 2649 | 1175 | 123 / 123 | 2628109 |
| RGBNT100 | reset_roles | 50 / 6559 | 0 | 123 / 123 | 2442253 |
| MSVR310 | reset_roles | 50 / 1000 | 980 | 123 / 123 | 2603533 |
| RGBNT201 | carry_roles | 50 / 2649 | 1210 | 123 / 123 | 2628109 |
| RGBNT100 | carry_roles | 50 / 6559 | 5 | 123 / 123 | 2442253 |
| MSVR310 | carry_roles | 50 / 1000 | 988 | 123 / 123 | 2603533 |

All auxiliary loss values are zero. The maximum formal step loss reconstruction error is 2.384185791015625e-07; logged loss is consistent with identity + triplet + auxiliary. RGBNT100 reset's triplet is zero at all 6,559 steps and carry's is nonzero at only five; this is observed loss behavior, not evidence that triplet was omitted from the active code. The active loss formula is `tools/run_correspondence_context_identity.py:111`–`:116`.

Every M0 has eight actual loader batches, 123/123 trainable parameter tensors with a nonzero gradient at least once across those batches, unchanged frozen Signal fingerprint, and recorded reload max error 0.0. This is union-of-batches coverage, not a claim that every parameter has a nonzero gradient on every batch. The source asserts these conditions before writing `M0_PASS` (`tools/run_correspondence_context_identity.py:122`, `:147`–`:167`); successful child exit/log/receipts and the hashed saved probes bind this evidence. No new neural M0 execution was performed by this auditor.

Within each dataset, reset/carry initializer dictionaries are exactly equal after deleting only `prompt_mode`; each full-run dictionary also equals its M0 dictionary. Baseline file SHA, trainable count, width, seed, optimizer settings, source bindings and initial full model state hash agree. The mode affects the forward prompt-selection branch (`modeling/trifusion/prompt_role_state.py:32`), not parameter construction (`:12`–`:20`, `:79`–`:87`). Each modality resets at layer zero, preventing cross-modality carry. Construction reseeds before new roles and hashes the initial full state (`tools/run_correspondence_roles.py:39`–`:56`). The auditor independently rebuilt the real model on CPU and regenerated all six initial hashes exactly.

| Dataset | Paired recorded initial model state SHA-256 |
| --- | --- |
| RGBNT201 | 3064592f65bf60c1c13570fbd4d8804c5575ed29dd3401f5b9d55e343dfc5f6e |
| RGBNT100 | 10ecddbe720b1b2dcfd370e1de5b84240b632745c69be6ddbece6942d2a1998a |
| MSVR310 | a75302513d9ab399f660e7535941345bb6bdfd28b98a4d7a1f6679f7470c0ed7 |

All **207** files in the frozen source manifest match live backend bytes and the independent snapshot; all seven per-run source bindings match the manifest. This verifies the declared source set, not a complete hash of the Python environment or a historical filesystem immutability guarantee. Three baseline weight files also match their actual SHA-256 bytes. `RGBNT100_PlainBaseline_30.pth` is the common frozen RGBNT100 initializer; RGBNT201/MSVR310 use their `*_50.pth` initializers (`tools/queue_correspondence_roles.py:21`). “50 epochs” describes the added role/prompt training stage, not total training from generic CLIP.

**Independent CPU model construction and strict state reload: PASS.** `strict_model_cpu.py` reconstructs the real Signal + PromptRoleTriFusion classes and real production Mamba, redirecting only explicit module CUDA transfers to CPU. All three frozen baseline state dictionaries load strictly; each frozen Signal hash matches its receipt; all six reconstructed initial full model hashes exactly match the recorded values. All six M0 probes and all six selected checkpoints then pass complete expected-key equality and `load_state_dict(strict=True)`, with zero missing/unexpected keys, 135 saved tensors each, exact loaded tensor equality and unchanged frozen Signal hashes. Trainable tensors/elements also match all receipts. The first library import was blocked by the initial zero-write policy; after explicit scratch authorization the successful complete run used a guard rejecting writes outside that isolated scratch. There was no substitution of a toy model. **This is state reconstruction/loading, not a fresh neural-forward or full inference replay.** M0 output-equality and gradient checks remain supported by the original successful execution evidence; no new GPU job was launched.

## D. Active versus dead evaluation code: WARN (unused helpers), active reported path PASS

The reported metrics are live. `tools/run_prompt_role_state.py:71` installs the prompt constructor/build/save/load/evaluate functions, then delegates to the context runner. `tools/run_correspondence_context_identity.py:244` installs the context train/evaluate pipeline into the base runner. Training calls `official_metrics` on every epoch (`:137`); final evaluation strictly loads the selected prompt checkpoint (`:188`), extracts all three paths (`:193`), calls upstream `eval_func` or `eval_func_msrv` (`:216`), checks against the secondary scorer (`:224`), and writes all three distance matrices plus the reported result (`:229`). The upstream scorer path is checked at `:189`. Saved histories, logs, distances and binary checkpoint metrics agree with this call chain.

The comparator's `R1_mAP`/`R1_mAP_eval` wrapper classes and visualization/KDE helpers (`comparators/Signal-cd1b0a6/utils/metrics.py:173`, `:222`, `:303`, `:387`, `:414`, `:514`, `:523`) are not invoked by this prompt-panel evaluation. Neither are the old standalone training/evaluation mains in the two `*_signal_oof.py` files. They must not be cited as extra evaluations for this panel. The imported `camera_scores` (`tools/train_rgbnt100_signal_oof.py:253`) and `scene_scores` (`tools/train_msvr310_signal_oof.py:223`) functions are called. This unused-helper warning has no numerical impact on the four claimed metrics. M3/prediction and auxiliary heads are intentionally disabled; their presence in inherited source is not evidence that those objectives were tested.

## E. Scope and claim assessment: WARN

The actual scope is two conditions × three datasets × one seed (42), each with 50 formal role-training epochs and one official-fused-mAP best checkpoint. There is no seed-variance estimate or untouched external test. Train identities are disjoint from evaluation identities, but that alone does not make the evaluation untouched: the official evaluation labels drive checkpoint selection at every epoch, and the plan explicitly says the formal splits participated in prior method development. Evaluation-set selection and same-weight decomposition must remain visible in all claims.

The preregistered gate requires all three mAP deltas positive, all three Rank-1 deltas nonnegative, and both RGBNT201/MSVR310 mAP deltas ≥ +0.5 pp (`local_docs/refine-logs/prompt_role_state_v1/EXPERIMENT_PLAN.md:26`). Rank-1 rises in all three and MSVR310 passes its mAP threshold; RGBNT201 mAP is −0.6140690806794709 pp. The gate therefore **FAILS**. Do not use a positive average across datasets to erase that failure or claim stable cross-dataset improvement.

All three paths below come from the same fused-selected checkpoint, not three independently trained models or three independently selected best epochs (`tools/run_correspondence_context_identity.py:187`–`:239`). The plan explicitly distinguishes this at `EXPERIMENT_PLAN.md:28`.

| Dataset | Mode | Same-checkpoint path | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | --- | --- | --- | --- | --- |
| RGBNT201 | reset_roles | shared_global | 70.941208254164 | 71.172249317169 | 80.622011423111 | 85.645931959152 |
| RGBNT201 | reset_roles | joint_local | 55.905131629838 | 60.406696796417 | 70.215308666229 | 77.033489942551 |
| RGBNT100 | reset_roles | shared_global | 85.617031393659 | 95.860058069229 | 96.618074178696 | 96.909618377686 |
| RGBNT100 | reset_roles | joint_local | 58.356401527129 | 78.309035301208 | 81.282800436020 | 83.032071590424 |
| MSVR310 | reset_roles | shared_global | 52.012435381239 | 68.358713388443 | 82.910323143005 | 87.478852272034 |
| MSVR310 | reset_roles | joint_local | 8.728754459914 | 18.950930237770 | 34.348562359810 | 43.485617637634 |
| RGBNT201 | carry_roles | shared_global | 69.740442213830 | 70.095694065094 | 78.947371244431 | 85.167461633682 |
| RGBNT201 | carry_roles | joint_local | 60.733384924011 | 60.047847032547 | 71.889954805374 | 80.143541097641 |
| RGBNT100 | carry_roles | shared_global | 85.708953004372 | 96.268218755722 | 96.501457691193 | 96.618074178696 |
| RGBNT100 | carry_roles | joint_local | 56.058778942607 | 77.317786216736 | 80.116617679596 | 81.924200057983 |
| MSVR310 | carry_roles | shared_global | 53.482852368612 | 71.235191822052 | 84.263956546783 | 89.001691341400 |
| MSVR310 | carry_roles | joint_local | 11.739711156728 | 22.842639684677 | 45.516073703766 | 52.791875600815 |

For example, MSVR310 carry fused − shared_global is +0.7443379500200606 pp mAP but −1.015228033065796 pp Rank-1. RGBNT100's fused-global mAP increment is only +0.03154555029524886 reset / +0.03391610409779844 carry. These are descriptive decompositions of the jointly trained model. They cannot establish an incremental causal benefit over independently trained global-only adaptation.

Independent paired query diagnostics:

| Dataset | Top-1 repairs / new errors | Query AP up / down / equal | Identity-macro AP delta (pp) |
| --- | --- | --- | --- |
| RGBNT201 | 35 / 32 | 296 / 287 / 253 | -0.441036803092 |
| RGBNT100 | 18 / 10 | 657 / 634 / 424 | +0.052273629514 |
| MSVR310 | 27 / 20 | 323 / 245 / 23 | +1.600987072941 |

Query-level counts and identity-macro AP use the same finite evaluation set; they are not seed robustness or causal certainty. The two RGBNT100 endpoints select epoch 1 yet continue all 50 epochs. All six final-epoch mAP values are lower than the selected best, which is relevant to selection bias/late-training behavior, not proof of fabricated performance.

## F. Evaluation type: PASS — real_gt

Formal fused, shared-global and joint-local retrieval are **real_gt** evaluations against dataset-provided identity/environment metadata. M0 is an engineering validation of loader/gradient/freezing/save-reload execution, not a retrieval-performance experiment. The earlier CPU toy contract referenced by the plan is **simulation_only/structural validation** and contributes no formal benchmark score. Inherited self-supervised M3 code is inactive in this panel.

## Claim impact and required follow-through

* Supported: six complete seed42 prompt-role runs, each trained for 50 role epochs, evaluated on the complete fixed official gallery, with the exact numbers above and bound one-best checkpoint evidence.
* Supported with qualifier: on this selected-checkpoint seed42 panel, carry changes mAP by −0.6141 / +0.0943 / +1.1773 pp on RGBNT201 / RGBNT100 / MSVR310. The predeclared progression gate is unmet.
* Unsupported: stable improvement across all datasets, multi-seed robustness, untouched-test generalization, novelty of cross-layer prompts as such, or a demonstrated independent role contribution beyond global-only adaptation.
* Supported: fresh CPU model-state construction reproduces all six initial hashes, and all 12 prompt checkpoint/probe state dictionaries load strictly. Independent matrix replay is complete. Full neural checkpoint inference was not rerun and must not be claimed.
* Refresh the current tracker/handoff to terminal mixed-result status, retain all negative results and the declared gate, and archive this report/JSON with original hashes. Do not automatically advance to the global-only follow-up under the failed registered gate or silently rewrite the gate. A later changed research question would need a separate explicit design decision.

## Audit-script execution notes

The first independent metadata pass assumed unused RGBNT201/RGBNT100 `scene` fields were zero. The protocol actually copies camera IDs into those fields; after checking all rows the audit parser was corrected to that observed encoding. The camera-based score path never uses those redundant scene fields. The successful replay is the final `cpu_replay.stdout.json`; the initial assertion is retained in `cpu_replay_initial_metadata_assumption_error.txt`.

The CPU construction attempt first selected the wrong `config` import due to auditor `sys.path` ordering; that order was corrected to match production source precedence. The next attempt was stopped by the original read-only guard on Torch temporary-directory creation. Following explicit scratch authorization, a guard permitting writes only there allowed the three complete CPU construction/strict-load checks to pass. Both earlier traces remain in `strict_model_cpu_initial_import_path_error.txt` and `strict_model_cpu_initial_readonly_tempdir_block.txt`. The local-copy checker initially included upstream MSVR310 `re.txt` visualization side-effect files, which the executor did not claim to copy; the final checker verifies the explicitly supplied training/steps/official files. These are reviewer harness corrections, not experimental failures.
