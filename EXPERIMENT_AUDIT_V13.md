# Experiment Audit Report — TriFusion V13

**Date**: 2026-09-02  
**Auditor**: GPT-5.5 xhigh, independent read-only review  
**Overall Verdict**: WARN  
**Integrity Status**: warn  
**Terminal Scientific Status**: `Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE`

## A. Ground-Truth Provenance: PASS

Q0 utilities are explicitly model-derived teacher proxies: full fusion-path
identity margin minus the margin after removing one residual slot. They use real
identity/camera relations but are not dataset labels or dev mAP. Q1 replay AP,
Rank-1 and margin are real identity/camera-label ReID replay metrics.

Evidence: `tools/build_v13_deployment_aligned_targets.py:222`,
`modeling/trifusion/signal_preserving_v13.py:201`,
`tools/train_v13_deployment_aligned_router.py:168`, and
`tools/diagnose_v6_oracle_complementarity.py:81`.

## B. Score Normalization: PASS

No self-normalization or metric inflation was found. Fusion uses the fixed
residual-energy contract followed by standard L2 retrieval normalization; AP
and Rank-1 are computed from distance ranking after the standard ReID junk
filter.

Evidence: `modeling/trifusion/signal_preserving_v13.py:166`,
`tools/diagnose_v6_oracle_complementarity.py:119`, and
`tools/train_v13_deployment_aligned_router.py:181`.

## C. Result and Provenance Consistency: WARN

The recorded numbers are internally consistent: Q0 has 571 queries, its unique
expert and modality wins each sum to 571, and its Oracle-minus-fixed value
recomputes exactly. Q1 contains 300 optimizer steps, matching three folds times
100 epochs. Local source/config hashes match their embedded receipts.

The warning is packaging only: the tracker was stale at audit time, and the
remote `.pth` cache/checkpoint payloads are represented locally by hashes rather
than copied into the lightweight Git repository. Q1 JSON `status: PASS` denotes
runner completion; the scientific gate is explicitly false.

Evidence: `evidence/trifusion_v13_deployment_aligned_preflight_seed42.json:21`,
`evidence/trifusion_v13_deployment_aligned_q0_seed42.json:147`, and
`evidence/trifusion_v13_deployment_aligned_router_q1_seed42.json:196`.

## D. Executed and Dead-Code Check: PASS

The Q1 implementation permits final refit and checkpoint creation only after a
passing gate. The failed run correctly records `final_training: null`,
`combined_checkpoint: null`, `next_phase_authorized: false`, and exactly 300
OOF Router steps.

Evidence: `tools/train_v13_deployment_aligned_router.py:499` and
`evidence/trifusion_v13_deployment_aligned_router_q1_seed42.json:2`.

## E. Scope, Leakage, and Claim Boundary: PASS

The scope is seed 42 on the fixed 141-fit three-fold internal protocol. The
all-fit Phase-A deployment inputs remain frozen; supervision and replay are
identity-OOF. Dev and official-test access are both zero. Because the student
feature extractor has seen all 141 fit identities, the result is not evidence
of identity-heldout representation generalization.

## F. Evaluation Classification: PASS

- Preflight: engineering preflight.
- Q0 utility/Oracle/action transfer: teacher proxy and diagnostic Oracle.
- Q1 expected utility/Top-1: teacher-proxy policy validation.
- Q1 replay AP/Rank-1/margin: real-GT OOF replay, not 30-dev deployment.
- Corruption/missing-modality checks: synthetic engineering controls.

## Claim Impact

Q0 legitimately qualifies a non-degenerate actual-path teacher proxy. Q1 does
not convert that signal into a statistically reliable policy: fold 0 fails
Top-1 non-inferiority, fold 2 fails expected utility, replay AP, and replay
margin, and all four aggregate identity-cluster bootstrap lower bounds are
negative. The small positive observed aggregate means must not be promoted.

Therefore V13 supports no final refit, no dev run, no official test, no
ablation, no deployable-gain or SOTA claim, and no representation-generalization
claim.

## Action Items

- Mark V13 Q1 terminal and failed in the experiment tracker.
- Preserve the three raw result JSONs and their hashes.
- Keep Q0/Q1 metrics labeled with the evaluation classifications above.

