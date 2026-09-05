# V24 M0 experiment-integrity audit

Date: 2026-09-06  
Auditor: Codex child audit agent. The audit request asked for GPT-5.5 xhigh; this is an OpenAI/Codex same-family independent audit, not a cross-family acquittal.

Overall verdict: WARN. V24 M0 engineering integrity is supported within bounded evidence, but scientific efficacy and Q1 qualification are not established. I found no evidence in the traced M0 path that heldout/dev/gallery features or crossfold distances were used to initialize or update the source prototype memory. Material limits remain: binary checkpoint/prototype-memory evidence is receipt-level here, Q1 was nonterminal, and this is same-family review.

## Scope and file provenance

I read the audit request, all 42 primary files in `.aris/traces/experiment-audit/2026-09-06_run14/input_file_sha256.json`, and the additional local dependencies needed by called V24 paths: `data/datasets/make_dataloader.py`, `modeling/trifusion/builder.py`, `modeling/trifusion/intervention_targets.py`, `modeling/trifusion/model.py`, and `modeling/trifusion/standalone.py`. I wrote only this Markdown report and the paired JSON report.

All 42 primary local files matched the supplied manifest exactly: `all_matched=True`. The supplied file verifier claims execution Git blobs and 12 prototype-memory binary receipts, while stating no tensor/image/model execution and live Q1 not evaluated (`evidence/trifusion_v24_m0_file_verification_20260906.json:2-8`, `evidence/trifusion_v24_m0_file_verification_20260906.json:471-476`). It also records known CRLF-only differences and an intentionally updated tracker (`evidence/trifusion_v24_m0_file_verification_20260906.json:477-635`).

## A-F verdicts

| Check | Verdict | Judgment |
|---|---:|---|
| A. Ground truth/source provenance | PASS | Fold construction relabels only fit/source records and separates heldout records (`tools/build_v12_complete_path_oof_targets.py:29-55`). V24 initializes memory from the passed source records, saves `source_only`, and checks model state unchanged (`tools/train_signal_preserving_v24.py:110-130`). The membership contract is metadata-only and reports no model/image/tensor/feature access (`evidence/trifusion_v24_source_membership_contract_20260906.json:2`, `evidence/trifusion_v24_source_membership_contract_20260906.json:1669-1672`). |
| B. Score mathematics | PASS | Recomputed all 116 M0 scalar rows. Max original weighted-sum discrepancy `1.0430812835693359e-07`; max prototype half-sum discrepancy `2.3283064365386963e-10`; max weighted-prototype discrepancy `2.3283064365386963e-10`; max step-total discrepancy `7.8968393379952317e-08`; max loss-list discrepancy `7.8968393379952317e-08`; rows over `1e-5`: `0`. |
| C. Result existence/provenance | WARN | Local manifest hashing passed. Remote execution Git blob and binary-memory evidence is available here as supplied receipts only, not local tensor inspection. |
| D. Execution | PASS | T0, M0, nonterminal Q1, and unrun D1/dev/official stages are distinct. T0 is synthetic only with zero real dataset/checkpoint/model/optimizer/eval activity (`evidence/trifusion_v24_t0_20260906.json:27-50`). M0 passed with 116 optimizer steps, 232 view pairs, 96 preflight view forwards, live gradients, unchanged frozen state, and zero overflow (`evidence/trifusion_v24_progress_20260906_011042.json:14-28`). Q1 had one observed epoch row and 29 logged updates at the 2026-09-06T01:10:42+08:00 snapshot (`evidence/trifusion_v24_progress_20260906_011042.json:2-13`). |
| E. Scope | WARN | M0 is explicitly engineering-only, not unknown-identity evidence (`refine-logs/v24/EXPERIMENT_PLAN.md:101-114`). Q1 was planned as reused train-internal OOF qualification, with dev and official access held at zero (`refine-logs/v24/EXPERIMENT_PLAN.md:88-93`). |
| F. Claim impact | WARN | Engineering claims are supported or qualified. Q1 completion, retrieval improvement, and D1/dev/official advancement are rejected or unavailable from this snapshot. The final evaluator aggregates scientific gates only after all six endpoint/fold runs (`tools/train_signal_preserving_v24.py:420-493`). |

## Source target semantics and membership

The prototype module normalizes detached source features, builds unique `(label, camera)` pairs, asserts contiguous local labels, and registers non-parameter buffers (`modeling/trifusion/source_prototype_v24.py:12-30`). Global prototypes are camera-balanced (`modeling/trifusion/source_prototype_v24.py:32-34`); strong-view loss uses global identity and same-camera environment classification with exactly one target (`modeling/trifusion/source_prototype_v24.py:36-49`); weak EMA updates one grouped `(identity, camera)` pair per observed pair and leaves unobserved pairs untouched (`modeling/trifusion/source_prototype_v24.py:51-62`). The learned prototype target is the local source label 0..93, not the original dataset identity and not heldout retrieval GT.

| Fold | Source records | Source identities | ID-camera pairs | Single-camera IDs | Multi-camera IDs | Identities per camera |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 2126 | 94 | 108 | 80 | 14 | {'0': 72, '1': 31, '2': 2, '3': 3} |
| 1 | 2075 | 94 | 108 | 80 | 14 | {'0': 71, '1': 31, '2': 3, '3': 3} |
| 2 | 2051 | 94 | 108 | 80 | 14 | {'0': 69, '1': 30, '2': 3, '3': 6} |

Every checked M0 initial memory receipt matched the corresponding fold membership pairs and started with zero update counts: `True`. Six preflight endpoint pairs matched on initial state, batch receipts, output hashes, and initial memory state: `True`. The two capacity probes shared first-eight batch receipts, and the fixed 100-step overfit probe started from the same initial memory as candidate capacity.

## Arithmetic and accounting

Arithmetic runtime: `0.000758700` seconds. The original loss formula matched the config and V24 code (`configs/RGBNT201/TriFusion-signal-preserving-v24-source-prototype-rtx3090.yml:42-50`, `tools/train_signal_preserving_v24.py:145-154`). V24 backpropagates `0.5 * weak_original`, then `0.5 * strong_original + enabled * prototype`, checks gradients, steps, and updates memory from detached weak features (`tools/train_signal_preserving_v24.py:157-198`).

| Quantity | Value |
|---|---:|
| M0 rows recomputed | 116 |
| Optimizer steps | 116 |
| View forward/backward pairs | 232 |
| Source initialization records | 18882 |
| Source initialization forward calls | 153 |
| Preflight view forward calls | 96 |
| Recomputed smoothing floor | 0.57838292104621 |
| Reported smoothing floor | 0.57838292104621 |
| Floor discrepancy | 0 |
| Recomputed fixed 100th-step ratio | 0.059999361786045424 |
| Reported fixed 100th-step ratio | 0.059999361786045424 |
| Ratio discrepancy | 0 |

| Run | Steps | View pairs | Prototype loss | Trainable tensors | Nonzero grad tensors | Missing grad count | Overflow | Frozen unchanged | Peak reserved MiB | First loss | Last loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Capacity control | 8 | 16 | False | 203 | 203 | 0 | 0 | True | 5950.0 | 0.611952334642 | 0.707601159811 |
| Capacity candidate | 8 | 16 | True | 203 | 203 | 0 | 0 | True | 6010.0 | 0.613431125879 | 0.711617618799 |
| Fixed 100-step candidate | 100 | 200 | True | 203 | 203 | 0 | 0 | True | n/a | 0.613431125879 | 0.580485790968 |

The parameter count was not recomputed by instantiating the model. The V24 build path records and asserts total parameters `98,800,141`, trainable parameters `7,841,292`, and trainable tensors `203` (`tools/train_signal_preserving_v24.py:81-96`), and the supplied preflight metadata repeats these values for all six fold/endpoint preflights (`evidence/trifusion_v24_preflight_metadata_verification_20260906.json:5-78`).

## Coverage and refresh limits

Coverage uses global negatives, same-camera environment negatives, retained cross-camera positives, age, and update counts (`modeling/trifusion/source_prototype_v24.py:64-76`). M0 rows had 93 global negative identities per anchor and 8 anchors with real other-camera positives. The dual-view loader uses the original cross-camera identity sampler (`modeling/trifusion/dual_view_data_v24.py:65-70`, `modeling/trifusion/aligned_data.py:97-221`) while the prototype objective expands source competition through memory. The fixed 100-step probe updated 9 of 108 prototype pairs and left 99 untouched; this is expected for a fixed batch, but it does not validate long-run Q1 freshness.

## Final claim impact

Supported: manifest-matched local primary files; V24 training-only identity-camera prototype code; retained source membership with 94 identities and 108 identity-camera pairs per fold; internally consistent M0 scalar arithmetic and gates.

Qualified: execution Git blob and binary prototype-memory receipts; parameter accounting; same-family review.

Rejected or unavailable: Q1 completion, retrieval improvement, D1/dev/official advancement, and cross-family acquittal.
