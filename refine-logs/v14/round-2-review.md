# V14 round-2 review

**Score: 9.25 / 10.00**

**Verdict: READY**

The reviewer found no remaining hard issue. The revised Q1 gates now match the
retrieval objective; the source-only minimax comparator does not use held-out
fold data; the fold-bound API prevents distances across OOF coordinate systems;
and the all-fit Router-input / OOF-teacher boundary is stated accurately.

## Resolved items

1. Held-out risk gain is the primary gate; AP and margin provide aligned
   anti-regression checks; V13 utility and action Top-1 are diagnostics only.
2. `s*` is selected only from the two source folds and cannot see the held-out
   fold.
3. Q1 and final refit bind every risk calculation to rows and embeddings from
   one OOF generator; only scalar fold regrets are compared.
4. The proposal no longer calls this complete-path OOF generalization or a
   likely deployable contribution.
5. Batch-hard risk remains the single training surrogate; no extra loss or
   tuning axis was added.

## Non-blocking suggestions absorbed

- Narrow the transfer wording to the three registered fit folds under all-fit
  deployment inputs and identity-OOF teacher/replay metrics.
- Report held-out best-fixed-slot risk/AP/margin as a non-gating oracle
  diagnostic so later wording cannot overstate the source-selected comparator.

