# Round 3 Review

<details>
<summary>GPT-5.5 xhigh 原始评审</summary>

## Round 3 原始评审

### Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 10 |
| Method Specificity | 9 |
| Contribution Quality | 9 |
| Frontier Leverage | 9 |
| Feasibility | 9 |
| Validation Focus | 9 |
| Venue Readiness | 8 |

**OVERALL SCORE = 9.05 / 10**

Weighted: `10*.15 + 9*.25 + 9*.25 + 9*.15 + 9*.10 + 9*.05 + 8*.05 = 9.05`.

### Verdict

**READY**

Meaning: ready as a method/validation design to implement and run under the stated gates. This is not a claim that the paper is ready to submit before Q0/Q1/dev outcomes and the two post-success deletion checks exist.

### Blocking Issues

None.

The Round 2 blocking issues are now materially fixed:

- Q1 claim boundary is correct: fixed all-fit deployment interface, identity-heldout policy validation, not representation OOF.
- all-fit Delta is correctly isolated as a one-shot binary action-transfer prerequisite.
- query-side target proxy is checked by OOF replay with learned query/gallery weights.
- hard gates now include per-fold replay mAP/margin non-inferiority and aggregate cluster-bootstrap lower bound > 0.
- shared `F(x;w)` seam plus SHA equality requirements are concrete enough to implement.

### Key Technical Judgments

Using all-fit student features is scientifically valid only under the stated narrow boundary. The backbone has seen all 141 fit identities, so it cannot support “heldout representation generalization.” But it can support “policy validation under the exact deployment feature interface,” because supervision and replay evaluation remain OOF and identity-heldout. The proposal now states that boundary correctly.

The action-transfer gate does not constitute target leakage as written. It must remain a binary prerequisite, not a knob. Since it does not enter loss, fixed policy selection, thresholding, checkpoint/epoch choice, or dev decision, it is a feasibility check for semantic slot alignment, not a tuned validation metric.

The fixed uniform gallery bank still differs from final per-sample routed query/gallery inference, but Q1 replay now closes the important part of that mismatch. The target is a local teacher signal; replay tests whether the learned policy actually improves OOF retrieval when both query and gallery weights are produced by the Router.

The bootstrap design is acceptable as a fail-fast gate, with a caveat. There are only about 21 cross-camera identities, so identity-cluster bootstrap is low-powered and should not be oversold as definitive statistical proof. But as a preregistered internal gate with 10k resamples and lower bound > 0, it is much stronger than the prior `1e-4` threshold and appropriately conservative for proceeding to one dev evaluation.

### Drift Warning

**NONE**

Problem Anchor is preserved. The proposal no longer drifts into adding descriptors, new experts, learned alpha, HFER, DINO, or hyperparameter search. The main contribution is now one focused mechanism.

### Simplification Opportunities

1. Keep `Top-slot accuracy` as a diagnostic/gate, but do not make it a narrative pillar; replay AP/margin is the real policy evidence.
2. Implement `F(x;w)` once and force Q0, Q1 replay, transfer audit, and final dev through the same function.
3. Keep all-fit Delta report-only statistics out of the main story except the binary transfer prerequisite.

### Modernization Opportunities

**NONE**

The foundation-model-era use is appropriate: frozen CLIP/Signal/V8 representation plus OOF utility distillation into a small controller. Adding DINO, HFER, LLMs, or a larger router would weaken the proposal.

### Remaining Action Items

- Define the bootstrap unit exactly in code: cluster by identity, preserve all queries within sampled identity, fixed seed42, 10k resamples, percentile 95% lower bound.
- Make replay mAP/margin gates fail-closed before final refit.
- Enforce Phase-A SHA equality between paired cache and final integration as a hard assertion.
- After dev success only, run exactly the two deletion checks: `actual-path -> residual-only` and `paired deployment input -> fold raw input`.

</details>
