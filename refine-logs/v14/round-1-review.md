# V14 round-1 review

**Score: 7.20 / 10.00**

**Verdict: REVISE**

The reviewer found no dev/official leakage and accepted the fold-local distance
boundary, but rejected the proposal because its Q1 gate remained tied to the
V13 pointwise utility/Top-1 target that the new diagnostic had already shown to
be nearly uniform and fold-dependent.

## Ranked issues

1. **Severe — objective/gate mismatch.** Training minimizes retrieval risk,
   while Q1 still made V13 expected utility and target-winner Top-1 core gates.
   Held-out retrieval risk must be the primary gate; V13 action metrics can be
   diagnostics only.
2. **Severe — source-fold protection is not held-out transfer.** Worst-source-
   fold training does not establish fold robustness. Q1 must directly gate
   held-out risk, AP and margin.
3. **Medium-high — OOF claim boundary.** Router inputs are all-fit by design;
   only teacher/replay embeddings are identity-OOF. The text must not claim a
   complete-path OOF feature pipeline.
4. **Medium — final-refit coordinate safety.** The per-fold risk API must bind
   rows and features to one OOF generator and prohibit cross-fold distances in
   both Q1 and final refit.
5. **Medium — evidence ceiling.** V14 is one train-only falsification of a
   relational Router objective, not an already supported deployable
   contribution.
6. **Medium-low — surrogate mismatch.** Batch-hard risk is reasonable, but Q1
   must also report and gate replay AP and margin without adding another loss
   or hyperparameter.

## Required disposition

Revise the fixed comparator to be selected only from source-fold retrieval
risk, define a held-out risk gain with an unambiguous sign, make risk/AP/margin
the Q1 gates, demote utility/target-winner Top-1 to diagnostics, and explicitly
bind every risk call to one fold's OOF generator.

