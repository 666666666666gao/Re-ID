# MSVR310 supported gradient balance R2 — complete Q1

Status: Q1_FAIL; full repaired CPU and executor text replay complete; fresh Q1 integrity audit IN_PROGRESS. Seed42, three identity folds, six fixed 20-epoch endpoints. Internal Q1 only; official table unchanged. Execution 1381639f778f77f124a2726ee55c07610092a438.

| Output | Control mAP | Balanced mAP | Delta | Control R1 | Balanced R1 |
|---|---:|---:|---:|---:|---:|
| baseline_only | 53.129381 | 53.129381 | +0.000000 | 63.000000 | 63.000000 |
| fused | 53.399384 | 53.452649 | +0.053266 | 62.166667 | 62.333333 |
| cnn | 50.262770 | 50.280983 | +0.018213 | 59.166667 | 59.666667 |
| transformer | 51.500319 | 51.356379 | -0.143941 | 60.500000 | 60.666667 |
| mamba | 52.025538 | 52.148367 | +0.122829 | 61.166667 | 61.333333 |

Paired fold fused gains: [0.12996670700828616, -0.1072138487788834, 0.14677427896478434]. Identity-bootstrap lower bound: -0.126141312. Paired gates 1/5; candidate vs Signal gates 1/5. Gates unchanged.

600 legal queries, 60 identities, 2,069,520 rank positions, all five outputs. Fused query changes: {"ap_improved": 240, "ap_declined": 203, "ap_unchanged": 157, "rank1_repaired": 3, "rank1_new_errors": 2}.

## Source optimization and actual intervention

| Endpoint | Last65 batch hard | Last65 extended hard | Last65 standard AP | Last65 cross-scene AP |
|---|---:|---:|---:|---:|
| control | 0.06453696 | 0.16116493 | 0.00557987 | 0.01459188 |
| balanced | 0.06576468 | 0.16247204 | 0.00548344 | 0.01361001 |

Registered cross-scene objective improved in each fold late in training, while both hard-margin diagnostics worsened in each fold. This is descriptive evidence, not proof of an AdamW cause. Balanced supported-role median ranking coefficients range 1.074397 to 1.156140.

Paired source records/pixels match, but warmup numeric trajectories already differ from step2: maximum total-loss differences per fold [0.0011240243911743164, 0.0014045238494873047, 0.0017092227935791016]. Do not interpret the tiny endpoint delta as a bitwise-isolated coefficient effect. Supported balanced updates also retain the disclosed separate-backward versus combined-backward numerical difference.

## Evidence and limitations

- All 1560 step records, 120 epoch records and 4680 role-step records retained. Epoch tables contain real training objectives, not invented heldout epoch scores.
- Q1 current-rank and direct-aux backward each run1560 times across both arms; Q1 direct-reference vector checks0. M0 reference checks are separate engineering evidence. Saved norm/cosine witnesses cannot reconstruct full parameter gradients or task-specific AdamW contributions.
- Original CPU stopped on sqrt exactness; first repair stopped on execution binding; bound repair stopped on weighted norm using double coefficients. All failures remain. Fresh same-family/provisional arithmetic audit supports only sqrt plus FP32 coefficient formula fixes, with all thresholds unchanged. Original execution files/config/checkpoints unchanged.
- Full final CPU receipt q1_cpu_arithmetic_recheck/verification.json: 1560 steps,116501504 memory distances,581120 historical-VJP record forwards,2069520 retrieval/rank elements,0 model forwards,0 updates. Original pipeline remains STOPPED_AT_Q1_CPU, not rewritten.
- Final verification rechecks stored arrays/checkpoint hashes; it does not independently regenerate every training gradient. AdamW moment/scaler states were not saved, so exact original task-state attribution is unavailable.
- Full Q1 independent audit pending; do not confuse arithmetic review with full scientific closure. No next experiment launched, no gate changes, no official-test access.

## Location

Raw evidence: evidence/supported_gradient_balance_q1_complete_20260921. Training tables, gradient summary and all-query/all-identity ranking tables have adjacent dedicated evidence directories. Arithmetic audit: evidence/supported_gradient_balance_cpu_equations_audit_20260921.
