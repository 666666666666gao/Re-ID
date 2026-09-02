# TriFusion V12 independent experiment audit

- Reviewer: independent GPT-5.5 xhigh
- Mode: read-only
- Overall verdict: `WARN`
- Integrity status: `warn`
- Scientific verdict: `Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE`

The audit found no evidence of metric fraud, dev/official leakage, or a phantom positive result. Q0 supports the complete-path OOF teacher qualification; Q1 fails to convert it into a deployable Router gain. The remaining WARN is artifact packaging: large remote `.pth` checkpoints/caches are not shipped in the lightweight repository and cannot be re-hashed from a fresh local clone.

## A. Ground truth and complete-path identity isolation — PASS

- The frozen plan requires three folds, 571 queries, Signal/expert fit IDs disjoint from held-out IDs, final-only schedules, non-saturation, diversity, Oracle gain, and dev0/official0.
- The executed builder subtracts held-out identities before relabeling and training.
- Each fold records Signal fit=94, expert fit=94, held-out=47, with Signal/held-out and expert/held-out overlap both 0.
- Queries are `190/179/202=571`; Signal trains 50 epochs final-only and experts train 20 epochs.
- The Q0 receipt records `complete_path_identity_isolation_passed=true`.

## B. Normalization and metric inflation — PASS

- Retrieval uses ordinary L2-normalized embeddings, distance ranking, same-ID/same-camera junk removal, and standard AP/Rank-1 aggregation.
- Slot utility is nearest-negative distance minus farthest-positive distance.
- No model-output self-scaling is used to manufacture the metric.
- Best fixed Q0 output is `87.9968 mAP < 99`; `non_saturation_passed=true`.

## C. Result/provenance/numeric consistency — WARN

- Q0 local config and runner SHA match the raw Q0 receipt.
- Q0 target cache SHA `fdacc405...ac681` matches the cache consumed by Q1.
- Q1 correctly records `combined_checkpoint=null`, `final_training=null`, `next_phase_authorized=false`, dev0 and official0.
- The raw Q1 schema does not itself embed project/config/runner identity. The terminal provenance wrapper now binds project commit `5c2b8ee...ad14`, config SHA `d7788961...aaa5`, runner SHA `259fd8ec...66d4`, remote log SHA and byte-identical result SHA.
- Q0/Q1/preflight JSON receipts are now included in the repository evidence package.
- WARN remains because the large remote checkpoints/cache cannot be independently re-hashed from this lightweight local package.

## D. Executed versus dead code — PASS

- Q0 records real training with 5,839 optimizer steps and runs the fold metric, margin, gate and cache paths.
- Q1 calls the OOF Router evaluator and guards final all-fit training/checkpoint creation on the OOF gate.
- The receipt matches the failed branch: three fold trainings occurred, but no final training or combined checkpoint occurred.

## E. Scope, leakage and claim boundary — PASS

- Scope is seed42, RGBNT201 train-only identity OOF, three folds and 571 eligible queries.
- Q0/Q1 dev and official access are 0.
- Q1 learned margin `-0.117330` is below fixed `-0.099975`; learned Top-1 `12.2592%` is below majority `16.8126%`.
- Quality-response gates pass, but the scientific Router gate fails.
- V12 may claim only Q0 teacher qualification and the negative Q1 Router result. It cannot claim deployable gain, dev mAP, official performance, SOTA or generalization.

## F. Evaluation classification — PASS

- Q0 residual mAP/Rank-1: `real_gt_train_identity_oof`.
- Q0 slot margins: `teacher_proxy_train_identity_oof`, derived from complete-path OOF features and real identity/camera relations.
- Q1 Router gate: `teacher_proxy_train_identity_oof`; no dev or official evaluation.

## Decision

Do not promote V12. Do not run dev, official, ablations, additional seeds, HFER, or fold/epoch/LR/alpha/margin/threshold scans. Preserve Q0 as bounded train-only diagnostic evidence and Q1 as a terminal negative Router result.

Evidence:

```text
evidence/trifusion_v12_complete_path_preflight_seed42.json
evidence/trifusion_v12_complete_path_oof_seed42.json
evidence/trifusion_v12_complete_path_router_seed42.json
evidence/trifusion_v12_complete_path_execution_provenance_seed42.json
results/TRIFUSION_RGBNT201_V12_COMPLETE_PATH_OOF_ROUTER_2026-09-02.md
```
