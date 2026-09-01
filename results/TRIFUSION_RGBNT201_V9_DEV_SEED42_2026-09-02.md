# TriFusion V9 seed-42 frozen dev terminal result

Date: 2026-09-02

Hardware: one remote NVIDIA RTX 3090 24 GB

Protocol: RGBNT201 fixed 141-fit / 30-dev, seed 42 only

Official-test access: 0

## Outcome

V9 Orthogonal Triadic Relay Synthesis is a complete negative main result. The
engineering contract passed, but the new representation did not generalize to
the identity-disjoint held-out dev split.

| Output | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| exact Signal baseline | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| frozen V8 Phase-B | **58.4050** | **59.3939** | **71.2727** | **76.6061** |
| V9 fused | 56.5339 | 57.2121 | 68.3636 | 75.5152 |
| V9 CNN | 55.8825 | 57.3333 | 68.6061 | 75.8788 |
| V9 Transformer | 51.3416 | 49.3333 | 65.8182 | 73.3333 |
| V9 Mamba | 54.6342 | 54.7879 | 68.1212 | 76.2424 |

The fused output is `-1.4770 mAP / -0.2424 Rank-1` versus exact Signal and
`-1.8711 mAP / -2.1818 Rank-1` versus V8 Phase-B. It is `8.4661 mAP` below
the preregistered 65 mAP gate. It exceeds the three degraded V9 expert outputs,
but that does not establish collaborative improvement.

## Training and evaluation integrity

- Code identity for the 60-epoch trainer: `b40b171`.
- Final checkpoint SHA-256:
  `c118ada931451929ec91cc374f9be8c3f518766b4dc02dda7372e525f07c7cfa`.
- Training completed 60/60 epochs and 2,520 optimizer steps in 1,334.80 s,
  with zero AMP overflows and 1,426.80/2,020 MiB peak allocated/reserved.
- The training loss moved from `3.45323` at epoch 1 to `0.62362` at epoch 60.
- Phase-A and Router state hashes were unchanged during training.
- Training read dev zero times. The final checkpoint was evaluated on dev
  exactly once; evaluation executed zero optimizer steps and did not mutate the
  checkpoint. Official-test access remained zero.
- The complete 3,072D Signal and 7,680D Phase-B embeddings remained exact
  prefixes of the V9 outputs.

## Mechanism diagnostics

The peer relay executed and remained numerically orthogonal to the receiver:
maximum absolute relay cosine was `1.01e-7`, and mean relay norm was `11.6084`.
The learned beta gate nearly saturated its fixed upper bound: mean/min/max were
`0.498794/0.462330/0.499998` with `beta_max=0.5`. This is an observation, not
a causal explanation; no post-hoc beta ablation or scan is authorized.

## Independent reviews

Independent result-to-claim review is `claim_supported=no`, confidence high.
It supports only the engineering facts that V9 preserves the frozen prefixes,
executes the orthogonal relay and completes the registered training protocol.
It does not support cross-identity synergy, the 65 mAP gate, official-test,
Mamba necessity, generalization or SOTA.

Independent integrity audit is `overall=WARN`, `integrity=warn`,
`scientific=FAIL_TO_PROMOTE`. Ground-truth provenance, standard ReID feature
normalization, executed evaluation path and evaluation-type classification all
pass. WARN reflects remote-only checkpoint packaging, previously untracked
terminal JSON files, stale pre-terminal documentation, and a config gate key
that is not the evaluator's runtime source of truth. None changes the numeric
negative result.

## Decision

V9 is sealed. Do not run official test, ablations, multiple seeds, checkpoint
selection, or beta/epoch/learning-rate/residual scans. Any successor must be a
new representation-level hypothesis and must first demonstrate positive
retrieval utility on identity-disjoint fit-only folds, including train-side
suppression of harmful additions, before another single final-only dev read.

## Evidence

```text
evidence/trifusion_v9_preflight_seed42.json
evidence/trifusion_v9_capacity_seed42.json
evidence/trifusion_v9_overfit_seed42.json
evidence/trifusion_v9_train_seed42.json
evidence/trifusion_v9_dev_seed42.json
EXPERIMENT_AUDIT_V9.md
EXPERIMENT_AUDIT_V9.json
```
