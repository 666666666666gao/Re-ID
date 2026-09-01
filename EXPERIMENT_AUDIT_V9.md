# Independent experiment audit — TriFusion V9

Date: 2026-09-02

Overall verdict: **WARN**

Integrity status: **warn**

Scientific verdict: **FAIL_TO_PROMOTE**

V9 is a complete but failed single-seed held-out-dev result. The evaluator's
`status=PASS` means the frozen evaluation completed; `promotion_gate=false` is
the scientific decision.

## Audit dimensions

| Dimension | Verdict | Evidence-backed conclusion |
|---|---|---|
| Ground-truth provenance | PASS | Training uses real fit identity labels; dev uses real identity/camera labels with same-ID/same-camera exclusion. V8 Phase-B is a frozen feature prefix/comparator, not ground truth. |
| Score normalization / inflation | PASS | Evaluation applies ordinary feature L2 normalization and Euclidean ReID distance, then standard CMC/AP. No prediction-self score rescaling or model-specific metric inflation was found. |
| Result existence / consistency / provenance | WARN | Readiness, 60-epoch training and frozen dev receipts are numerically consistent; V9 Phase-B exactly matches the prior V8 result. The large checkpoint remains remote-only and cannot be independently rehashed from this local source snapshot. |
| Dead code / executed paths | WARN | The V9 trainer, synthesis path, frozen evaluator and promotion gate are executed. `GATES.FUSED_STRICTLY_BEATS` is present in config, while the evaluator uses a matching hard-coded comparison tuple; current values match, so the result is not invalidated. |
| Scope / leakage / claim language | WARN | seed42, B64/K8, 60 epochs, train dev0, final dev1, optimizer0 at evaluation and official0 are evidenced. Terminal documentation was stale before this audit. |
| Evaluation classification | PASS | Readiness checks are engineering-only; training and frozen dev are real-GT. No synthetic or human evaluation is presented as final retrieval evidence. |

## Claim impact

- Supported: exact Signal and Phase-B prefixes, frozen base states, numerical
  relay orthogonality, and completion of the registered train/eval protocol.
- Failed: V9 converts complementarity into a stronger deployable identity
  representation. Fused is 56.5339 mAP, below Signal by 1.4770 and Phase-B by
  1.8711.
- Failed: 65 mAP, official-test unlock, SOTA or generalization. Promotion and
  next-phase authorization are false; official access is zero.
- Readiness PASS remains an engineering statement only.

## Required disposition

Track the two terminal JSON receipts, update the terminal report/tracker, and
seal V9 without official test, ablation, multi-seed or hyperparameter scans.
Remote artifact mirroring would be required only for independent local
checkpoint rehash; its absence is preserved as WARN rather than hidden.
