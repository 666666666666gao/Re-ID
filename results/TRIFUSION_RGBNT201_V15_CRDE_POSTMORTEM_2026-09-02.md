# TriFusion RGBNT201 V15 CRDE Read-Only Postmortem

## Scope

This diagnostic replays the three registered V15 Q1 final checkpoints on their
existing identity-OOF heldout train records. It performs one matched exchange-
on/off forward, no component deletion, no optimization, no dev and no official
test. It is a postmortem, not an ablation or a new performance result.

## Raw exchange geometry

Incoming energy is reported relative to the receiver's own role-adapter delta.

| Fold | Stage | CNN | Transformer | Mamba |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.024 | 0.357 | 0.145 |
| 0 | 1 | 0.251 | 0.395 | 0.180 |
| 1 | 0 | 0.106 | **0.428** | 0.202 |
| 1 | 1 | 0.131 | 0.339 | 0.146 |
| 2 | 0 | 0.110 | 0.418 | 0.144 |
| 2 | 1 | 0.175 | 0.291 | 0.132 |

The corresponding incoming-versus-own-delta cosine values are between
`-0.030` and `+0.060`, with most near zero. The exchange is therefore not
simply reinforcing the receiver's existing identity direction.

## Static edge instability

Ten of twelve stage-edge combinations have fold sign agreement `1/3`; only
stage0 Transformer→CNN and stage1 Mamba→CNN retain one sign in all folds.

| Stage | Edge | Fold scales | Sign agreement |
|---:|---|---|---:|
| 0 | CNN→Transformer | -0.01392 / +0.01451 / -0.01641 | 0.333 |
| 0 | CNN→Mamba | -0.00596 / +0.00655 / +0.00565 | 0.333 |
| 0 | Transformer→CNN | -0.00029 / -0.00021 / -0.00052 | 1.000 |
| 0 | Transformer→Mamba | -0.00273 / +0.00486 / -0.00269 | 0.333 |
| 0 | Mamba→CNN | +0.00019 / -0.00114 / +0.00138 | 0.333 |
| 0 | Mamba→Transformer | -0.00406 / +0.00403 / -0.00409 | 0.333 |
| 1 | CNN→Transformer | +0.01385 / +0.01175 / -0.01087 | 0.333 |
| 1 | CNN→Mamba | +0.00470 / +0.00568 / -0.00624 | 0.333 |
| 1 | Transformer→CNN | +0.00127 / +0.00120 / -0.00059 | 0.333 |
| 1 | Transformer→Mamba | -0.00456 / +0.00389 / +0.00245 | 0.333 |
| 1 | Mamba→CNN | +0.00205 / +0.00073 / +0.00167 | 1.000 |
| 1 | Mamba→Transformer | +0.00530 / +0.00473 / -0.00448 | 0.333 |

## Query-level matched changes

| Fold | Output | Improved | Harmed | Unchanged | Mean displacement |
|---:|---|---:|---:|---:|---:|
| 0 | fused | 34 | 56 | 100 | 0.1343 |
| 1 | fused | **5** | **30** | 144 | **0.1391** |
| 2 | fused | 48 | 55 | 99 | 0.1160 |
| all | fused | 87 | 141 | 343 | — |
| all | CNN | 113 | 136 | 322 | — |
| all | Transformer | 107 | 143 | 321 | — |
| all | Mamba | **153** | **89** | 329 | — |

The displacement/AP-gain correlation is not stable: fused correlations are
`-0.208/+0.112/-0.135` across folds. Larger hidden-state change is not a
reliable proxy for helpful retrieval change.

## Findings

1. **Observation:** edge signs reverse across identity folds. **Interpretation:**
   static directed exchange learns fold-specific transport polarity.
   **Implication:** neither increasing nor decreasing one global exchange scale
   addresses the transfer failure.
2. **Observation:** Transformer receives the largest effective injection but
   has negative aggregate Q1 gain. **Interpretation:** V15 is not merely too
   weak; substantial injected energy can be harmful. **Implication:** the next
   hypothesis must control what relational evidence is transferred, not just
   its magnitude.
3. **Observation:** Mamba has more improved than harmed queries and is the only
   positive aggregate receiver. **Interpretation:** collaboration signal exists
   but is receiver- and identity-dependent. **Implication:** forcing symmetric
   hidden-vector exchange is unsupported.

## Next hypothesis boundary

Do not tune V15 scale, edge, regret, epoch or checkpoint. A successor should
move collaboration from inference-time hidden-vector injection to training-time
selective retrieval-relation transfer: peers teach only relations on which they
agree and improve the exact Signal anchor, while every expert keeps its private
inference representation. This is a new hypothesis and requires a new frozen
proposal before implementation.

## Integrity and evidence

- Commit: `27f9a6a5ef02aa478c493bdee59a258d28eb4e9b` with empty runtime diff.
- Optimizer steps / training rerun: `0 / false`.
- Dev / official accesses: `0 / 0`; D1 false.
- Frozen state unchanged in all folds.
- Runtime 106.68 s; peak reserved 6,354 MiB.
- `evidence/trifusion_v15_crde_postmortem_seed42_27f9a6a.json`
- `evidence/trifusion_v15_crde_postmortem_seed42_27f9a6a.console.log`
