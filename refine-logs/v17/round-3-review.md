# V17 DTRED Round 3 Review

<details open>
<summary>GPT-5.5 xhigh raw review</summary>

**V17 Round 3 Review**

| Dimension | Weight | Score |
|---|---:|---:|
| Problem Fidelity | 15% | 9.8 |
| Method Specificity | 25% | 9.2 |
| Contribution Quality | 25% | 9.0 |
| Frontier Leverage | 15% | 8.8 |
| Feasibility | 10% | 8.2 |
| Validation Focus | 5% | 9.5 |
| Venue Readiness | 5% | 7.8 |

Weighted overall: **9.03 / 10**

Verdict: **READY**

Anchor: **PRESERVED**

The proposal still targets the original problem exactly: fixed RGBNT201
141-fit/30-dev, seed42, RTX3090, no reranking, exact Signal preservation, V8
Phase-B as current deployable best, Q1 as identity-OOF train-only
qualification, and D1 as the only no-reranking dev gate. No graph inference,
runtime fallback, new backbone, Router revival, or experiment menu has entered
the method.

Contribution focus: **SHARP**

The dominant contribution is now clean: **heterogeneous relation-envelope
distillation into no-reranking single-sample embeddings**. The proposal no
longer overclaims max/min itself as novelty. It properly places novelty in the
combination of three heterogeneous expert relation teachers, one-sided envelope
constraints, deployable compression, and complete-path identity-OOF
qualification.

Simplicity: **IMPROVED**

The method is simpler and more defensible than Round 1. The scalar convention
is now frozen:

```text
E(u) = 0.5 E+(u) + 0.5 E-(u)
L_env = 0.5 E(f) + 1/6 sum_e E(h_e)
```

That removes the previous ambiguity without adding a component. Teacher-source
distributions are descriptive receipts only, not new gates. This is the right
boundary.

Frontier leverage: **GOOD**

CLIP/Signal plus relation distillation remains the natural modern mechanism.
The proposal is current without chasing fashionable components. No
DINO/SAM/VLM addition is warranted.

Remaining blocking method issue: **NONE**

The prior blockers are resolved: exact regression was replaced by one-sided
constraints, positive/negative imbalance is fixed, teacher leakage is
explicitly closed, loss scaling is bound, mechanism receipts are in-run
diagnostics, and the V8 query-level Oracle ceiling is correctly separated from
pair-level envelope synthesis.

Simplification opportunities: **NONE**

Do not remove the branch receipts, because they are needed to defend the
triadic-transfer claim. Do not add modules.

Modernization opportunities: **NONE**

No new backbone, reranking, test-time graph, memory bank, EMA teacher, or VLM
should be added.

Drift warning: **NONE**

The only wording boundary to preserve is that Q1 remains train-only mechanism
qualification. It must not be reported as deployable dev performance.

Remaining method-level actions:

- Implement exactly as specified.
- Keep teacher construction source-fold-only in Q1.
- Keep teacher-source distributions descriptive, not adjustable.
- Enforce no fallback, no reranking, no checkpoint selection, no post-hoc
  threshold change.
- If Q1 or D1 fails, seal V17 rather than scanning width/loss/LR/epoch.

Final judgment: **READY for the gated V17 implementation/run**, not paper
submission before results.

</details>
