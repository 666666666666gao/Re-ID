# V16 SATR fit-only threshold freeze probe

Purpose: freeze V16 relation thresholds without reading the 30-identity dev or
official test. This is a proposal-time, optimizer-0 diagnostic, not a performance
result or hyperparameter scan.

Complete candidate disclosure:

- `delta_r`: only `0.05` was evaluated; no alternative delta was tried.
- `gamma_p`: an initial fold0 sanity pass used `0.10` and protected `100%` of
  valid queries, so it was rejected as saturated before the three-fold freeze.
  The single replacement `0.30` was then evaluated on all three folds and frozen.
- `epsilon_p=0.02`, `lambda_r=1.0`, and `lambda_p=0.25` were set from the
  observed normalized-margin scale and existing registered loss weights; no
  alternative values were executed.
- No Q1 endpoint, 30-dev metric, or official-test metric participated in any
  threshold decision.

- Source: the three registered V12 complete-path fold checkpoints.
- Records: each fold's source-training identities only.
- Sampling: the first eight deterministic seed-42 B64/K8 batches per fold.
- Relation: for every query with an available cross-camera positive, exact Signal
  selects its lowest-similarity positive and highest-similarity negative. All
  experts are measured on that same pair.
- Candidate repair gap: `delta_r=0.05`.
- Candidate protection threshold: `gamma_p=0.30`.
- Optimizer steps / dev accesses / official accesses: `0 / 0 / 0`.

| Fold | Valid queries | CNN repair coverage | Transformer | Mamba | Signal protection coverage (`m0>=0.30`) |
|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 1.5625% | 3.1250% | 4.6875% | 48.4375% |
| 1 | 64 | 3.1250% | 1.5625% | 14.0625% | 26.5625% |
| 2 | 72 | 4.1667% | 1.3889% | 5.5556% | 58.3333% |

Signal hardest-relation margin quantiles `(q10/q25/q50/q75/q90)` were:

- fold0: `0.2302/0.2604/0.2975/0.3368/0.4318`;
- fold1: `0.2095/0.2356/0.2771/0.3052/0.3284`;
- fold2: `0.2579/0.2815/0.3096/0.3397/0.3597`.

Frozen interpretation:

- `delta_r=0.05` is selective but non-empty for every receiver/fold in this
  train-only probe;
- `gamma_p=0.30` protects a non-trivial, non-saturated portion of the exact
  Signal-hard relations;
- V16 pre-registers per-receiver **fixed-initial** coverage in `[0.5%, 25%]`
  on the same deterministic optimizer-0 batches and does not change thresholds
  after Q1 starts. Training-trajectory coverage is diagnostic only: it may fall
  as receivers repair relations and cannot fail an otherwise successful run.
  Initial coverage outside the interval fails the mechanism-activity gate and
  does not trigger a fallback teacher.

