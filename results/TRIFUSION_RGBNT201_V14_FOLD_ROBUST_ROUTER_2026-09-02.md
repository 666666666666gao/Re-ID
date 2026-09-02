# TriFusion RGBNT201 V14 Fold-Robust Retrieval-Regret Router

## Terminal verdict

`Q0_PASS_Q1_FAIL_DO_NOT_PROMOTE`

V14 replaced V13's near-uniform pointwise utility-KL with fold-local
cross-camera retrieval risk and a worst-source-fold regret objective. Q0 passed;
the unique seed-42 Q1 failed. No final refit, combined checkpoint, dev evaluation
or official-test access occurred.

## Frozen protocol

- Remote RTX3090 only; seed42.
- 571 fit-only cross-camera queries, 21 identities, three registered folds.
- Router input: frozen all-fit V8 Phase-A deployment features.
- Teacher/replay embeddings: identity-OOF; distances remain inside one fold.
- Source comparator: one minimax fixed slot selected from the two source folds.
- Fusion: exact Signal prefix plus fixed `alpha=0.2` routed residual bank.
- Router: hidden128, AdamW, LR3.5e-4, 100 epochs/fold; unchanged quality loss.

## Q0 zero-step qualification

| Quantity | Result |
|---|---:|
| Status | PASS |
| Queries by fold | 190 / 179 / 202 |
| Identities by fold | 7 / 7 / 7 |
| Minimax fixed slot | 2 |
| Minimax worst-fold risk | 0.7034838 |
| Initial worst-fold regret | 0.0018155 |
| Router gradient tensors finite/nonzero | 14 / 14 |
| Cross-fold feature distances | 0 |
| Optimizer steps | 0 |
| Dev / official accesses | 0 / 0 |
| Phase-A state | unchanged |
| Peak reserved memory | 636 MiB |
| Elapsed | 9.30 s |

Q0 verifies only executable signal, not a performance gain.

## Q1 held-out replay

| Held-out fold | Source fixed slot | Risk gain | AP gain | Margin gain | Gate |
|---:|---:|---:|---:|---:|---|
| 0 | 2 | +0.0003567 | **-0.0005571** | +0.0004422 | FAIL: AP |
| 1 | 2 | +0.0045235 | +0.0049162 | +0.0091674 | PASS |
| 2 | 2 | **-0.0016102** | +0.0001532 | **-0.0033642** | FAIL: risk/margin |

The held-out best fixed slots were `2/0/2`; they were diagnostic only.

| Learned-minus-fixed quantity | Observed mean | 95% lower bound | Gate |
|---|---:|---:|---|
| Retrieval-risk gain | +0.0009671 | **-0.0018584** | FAIL |
| Replay AP gain | +0.0014100 | **-0.0054337** | FAIL |
| Replay margin gain | +0.0018309 | **-0.0039411** | FAIL |

The positive aggregate means are not promoted because all registered lower
bounds are negative and two folds fail per-fold gates.

## Quality, resources and state

| Modality | Clean mass | Corrupted mass | Gate |
|---|---:|---:|---|
| RGB | 0.335696 | 0.108547 | PASS |
| NI | 0.332487 | 0.117822 | PASS |
| TI | 0.331818 | 0.116403 | PASS |

Missing-modality maximum mass is zero. Phase-A SHA before/after is
`ecfd7fbc...fb77`. Q1 used 300 steps in 37.79s with peak allocated/reserved
`2459.15/3400 MiB`. Dev and official accesses are zero.

The Q1 JSON `status=PASS` means runner completion; scientific qualification is
`router_oof.gate.passed=false` and `next_phase_authorized=false`.

## Result-to-claim and next boundary

Independent result-to-claim is `no/high`. Integrity is WARN only because the
tracker was stale during review and runner-completion `PASS` could be confused
with gate success; GT provenance, normalization, executed path, leakage/scope
and evaluation classification pass.

V14 is sealed. Current deployable best remains V8 Phase-B at
`58.4050 mAP / 59.3939 Rank-1`, `6.5950 mAP` below the 65 dev gate. V14 adds no
dev metric. Changing the Router loss alone is insufficient when the all-fit
sample-local input does not reliably expose held-out relational utility.

## Evidence

- `evidence/trifusion_v14_q0_seed42.json`
- `evidence/trifusion_v14_q1_seed42.json`
- `EXPERIMENT_AUDIT_V14.md/.json`
- `RESULT_TO_CLAIM_V14.md/.json`
