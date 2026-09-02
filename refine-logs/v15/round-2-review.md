# V15 Round 2 Review

<details>
<summary>GPT-5.5 xhigh raw review</summary>

**Round 2 Review**

Read the full revised proposal. The revision fixes the main Round 1 blockers at the proposal level: the off-path counterfactual is now state-clean, BN/classifier ownership is explicit, stage-3 exchange is removed, V8 is demoted to substrate, and Q1 is correctly scoped as authorization rather than deployable evidence.

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 9.0 | The anchor is preserved. The method still attacks the real bottleneck: new representation formation beyond fixed expert selection, failed router utility prediction, and late pooled relay. |
| Method Specificity | 8.5 | Interfaces, frozen/trainable boundaries, off/on paths, pre-BN evaluation, and two-stage injection are now implementable. Remaining detail: freeze the exact regret coefficient in code/docs. |
| Contribution Quality | 8.5 | The contribution is now one focused mechanism: delta-only pre-tail exchange trained by matched no-exchange retrieval regret. Much cleaner than Round 1. |
| Frontier Leverage | 8.0 | Appropriate. Using the frozen CLIP tail as the semantic interpreter is a natural foundation-model-era move; no LLM/VLM/Diffusion/RL should be added. |
| Feasibility | 8.0 | Fits the single-3090/stagewise constraints if M0 capacity passes. Dual off/on forward is bounded because the off path is no-grad/eval. |
| Validation Focus | 8.0 | Minimal and claim-driven. Q1 gates mechanism transfer under complete-path identity isolation; D1 alone supports deployable ≥65. |
| Venue Readiness | 7.0 | Sharper now, but still not READY for a top venue until implementation proves the narrow mechanism can exceed the strong local failure history. |

**OVERALL weighted score: 8.35 / 10**

No dimension is below 7. The proposal is substantially improved, but it does not reach READY because the method is still empirically high-risk: it must create ranking capacity beyond the V8 fixed-branch Oracle regime, and its top-venue novelty depends on the exact CRDE distinction holding up in implementation and post-success ablations.

**Anchor Status**

Preserved. No drift. The revised method does not switch to routing, quality prediction, external backbones, dev tuning, or Oracle exploitation.

**Contribution Focus**

Now tight. “CRDE + matched regret” is correctly treated as one inseparable mechanism, not two contributions. V8 is substrate only. This resolves the prior contribution-sprawl issue.

**Simplicity**

Appropriate. Deleting tail11/stage-3 exchange is a good simplification because every injected delta is now interpreted by a later frozen CLIP tail. Moving nonzero edge counts to diagnostics is also correct; nonzero communication is not scientific evidence.

**Frontier Leverage**

Appropriate and not forced. The proposal’s modernity is the use of a frozen pretrained CLIP tail as a semantic transition operator after each peer delta injection. Do not add DINO, text, VLM teachers, LLM planners, diffusion, RL, or reranking.

**Implementation / Leakage / Scope Check**

No hard blocker remains, assuming the implementation follows the revised text exactly.

- **V12 reuse**: no leakage if each Q1 fold uses the corresponding complete-path identity-OOF V12 checkpoint whose base saw only the 94 source identities. Must log fold IDs and checkpoint SHAs.
- **Scope**: correctly handled. Q1 only authorizes D1; it does not support ≥65, SOTA, or deployable claims.
- **Counterfactual**: proposal-level issue fixed. Off-path is frozen V8 base, `eval()`, `torch.no_grad()`, detached pre-BN embeddings, no V15 BN/classifier calls, no state mutation.
- **BN/classifier mismatch**: proposal-level issue fixed. Heads are source-class-local and on-path-only; evaluation bypasses them.
- **Gate support**: Q1 supports “CRDE beats matched no-exchange under full-path identity isolation.” D1 is the only gate that can support the deployment claim.

**Remaining Action Items**

1. Add M0 tests that explicitly fail if off-path invokes V15 BN/classifier modules or changes any frozen state/BN running stats.
2. Freeze and name the regret coefficient exactly. The formula implies a fixed average weight, but the implementation should make this unambiguous and unscanned.
3. Ensure on/off receive the exact same post-augmentation tensors in each batch; otherwise the counterfactual is not matched.
4. In the paper framing, keep Q1 as “authorization evidence” only. The deployable claim begins and ends with D1.

**Simplification Opportunities**

NONE material. The current two-stage CRDE is about as small as the mechanism can be while still testing the intended claim.

**Modernization Opportunities**

NONE. The modern primitive is already the frozen CLIP tail. Additional foundation-model components would be drift.

**Drift Warning**

NONE.

**Verdict: REVISE**

This is a strong revision and a plausible next experiment plan. It is not READY under the stated rule because the overall score is below 9 and venue readiness still depends on exact implementation plus D1 success.

</details>
