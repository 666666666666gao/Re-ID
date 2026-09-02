# TriFusion V16 SATR M0 Experiment Audit

**Date**: 2026-09-02

**Auditor**: GPT-5.5 xhigh, read-only independent review

**Overall verdict**: WARN

**Integrity status**: `warn`

**Routing verdict**: `M0_FAIL_Q1_BLOCKED`; stop V16 under this experiment
identity

## A-F checks

### A. Ground-truth provenance: PASS

RGBNT201 identity and physical-camera labels come from the fixed dataset
protocol. V12 fit and held-out identities are disjoint, and V16 verifies those
fold identities before loading the fold checkpoints. The SATR peer and Signal
relations are model-derived training signals, but the experiment is correctly
labelled as a train-only mechanism probe rather than a deployment result.

Key evidence: `protocols/rgbnt201_dev_v1.json:1-18`,
`tools/build_v12_complete_path_oof_targets.py:29-56`,
`tools/train_signal_preserving_v16.py:315-319`, and
`modeling/trifusion/signal_preserving_v16.py:72-76`.

### B. Score normalization: PASS

No prediction-statistic normalization or hidden rescaling was found. M0 repair
coverage is the raw eligible-query fraction. The unused conditional retrieval
path uses ordinary L2-normalized embeddings, Euclidean distance, AP and
Rank-1. M0 produced no retrieval metric, as the result report states.

Key evidence: `tools/train_signal_preserving_v16.py:493-509`,
`tools/diagnose_v6_oracle_complementarity.py:81-107`, and
`results/TRIFUSION_RGBNT201_V16_SATR_M0_2026-09-02.md:12-14`.

### C. Result existence and claimed numbers: WARN

The evidence JSON, result report and both trackers agree that M0 completed as a
scientific failure and that Q1/D1 were not run. Capacity, overfit, paired-state
and formal coverage numbers match. The warning concerns the proposal-time
threshold probe: it did not bind sampler indices or transformed-batch hashes,
and its nonzero three-receiver coverage did not reproduce in the formal M0
runner. The terminal report discloses this discrepancy. This is a
reproducibility and packaging warning, not metric fraud.

Key evidence: `evidence/trifusion_v16_satr_m0_seed42_20260902.json:9473-9623`,
`results/TRIFUSION_RGBNT201_V16_SATR_M0_2026-09-02.md:41-59`, and
`refine-logs/EXPERIMENT_TRACKER.md:5-18`.

### D. Executed path: PASS

The Signal-hard pair selection, detached two-peer teacher, receiver-only SATR
repair and fused protection loss are live. The optimizer calls the model,
criterion and backward path. Formal M0 calls the SATR objective on the first
eight matched batches and records 203/203 finite nonzero gradient tensors. The
zero Transformer coverage is therefore an executed negative result, not dead
code.

Key evidence: `modeling/trifusion/signal_preserving_v16.py:57-168`,
`modeling/trifusion/signal_preserving_v16.py:218-233`, and
`tools/train_signal_preserving_v16.py:385-496`.

### E. Scope and leakage boundary: PASS

The run is one seed-42 RGBNT201 train-only M0 on a single RTX3090. It accessed
neither the 30-identity dev split nor official test. The registered fail-closed
plan makes M0 failure block Q1 and D1. No deployment, 65 mAP, official, SOTA or
ablation claim is supported.

Key evidence: `refine-logs/FINAL_PROPOSAL.md:5-28`,
`refine-logs/EXPERIMENT_PLAN.md:32-57`, and
`results/TRIFUSION_RGBNT201_V16_SATR_M0_2026-09-02.md:7-14`.

### F. Evaluation type: PASS

Classification: `self_supervised_proxy_train_only_engineering_probe`, with
real train identity and camera labels used for ReID relations and supervised
capacity losses. It is not dev, official, deployment or human evaluation.

## Claim impact

Supported:

- The V16 SATR path is executable on RTX3090 B64/K8.
- Exact Signal prefix and frozen states were preserved.
- The paired initial state, trainable names, sampler order, transformed batches
  and seed contract matched.
- Capacity and fixed-batch optimization sanity passed.

Unsupported:

- Active three-receiver SATR teaching.
- Identity-disjoint mutual promotion.
- A promoted V16 checkpoint or any new retrieval metric.
- Dev, official or SOTA improvement.

## Action items

1. Keep V16 sealed and do not run Q1 or D1 under this experiment identity.
2. Do not relax thresholds or alter worker/RNG/sampler order after the failure.
3. Require any successor activity probe to store sampler indices and
   post-transform RGB/NIR/TIR tensor hashes.
4. Include per-file hashes for the core, builder, runner and config in future
   receipts.
