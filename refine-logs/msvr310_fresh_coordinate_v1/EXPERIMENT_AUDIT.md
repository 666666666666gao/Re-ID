# MSVR310 fresh-coordinate V1 complete M0 audit

Date: 2026-09-08. Auditor: `/root/audit_msvr_fresh_coordinate_m0`, requested gpt-6-astra / max, fresh context, local read-only. `review_independence: same-family`; `acceptance_status: provisional`. Overall **WARN**, deterministic local text checks **PASS**, engineering evidence **PASS_WITH_LIMITS**. No training-code revision or restart required. This document records the actual reviewer conclusions, not cross-family acceptance.

## A–F findings

| Check | Status | Evidence and finding |
|---|---|---|
| A. Ground-truth provenance | PASS with remote-data limits | Protocol stores dataset identities/scene/camera; trainer labels follow those indices, not model outputs. Reviewer checked all1032 records, three-fold identity separation, all600 legalquery masks and all248 training rows. `protocols/msvr310_train_oof_v1.json:4`, `tools/train_msvr310_signal_oof.py:68`, `tools/train_msvr_fresh_coordinate.py:85`, `tools/msvr_instance_memory.py:47`. Original images/download and remote model payloads were not locally reauthenticated. |
| B. Score normalization | PASS | L2 feature normalization defines Euclidean distance; AP/CMC denominators use true legalpositives/querycounts. Overfit ratio is explicitly a convergence gate with initial/final/analyticfloor reported, not retrieval accuracy. `tools/msvr_instance_memory.py:43`, `tools/train_msvr310_signal_oof.py:223`, `tools/run_signal_preserving_v5.py:1580`. |
| C. Existence and numbers | PASS, binary-access WARN | All27 rawtexts/2,571,292B size/SHA match, six receipts/eight training files/eight JSONL files agree. All248 updates and reportnumbers checked. CPUreceipt lists45 files:23 localtexts verified,22 binaries unavailable locally (6checkpoints,16f32). Localreview checks declaredlayouts/counts/CPUreceipt, not rawdistance values or modelreload. `evidence/msvr310_fresh_coordinate_complete_m0_20260908/intake_manifest.json`, `m0_cpu.json:121`. |
| D. Calls and update semantics | PASS | Actualrefresh/capture, both expanded losses, endpoint choice, replacement and backward/step connected. All248 rows select registered coordinates and losses. `tools/train_msvr_fresh_coordinate.py:96`, `:113`, `:133`; gradientprobe in `tools/msvr_freshness_probe.py:97`. Q1 branches are not treated as completed evidence. |
| E. Scope | PASS with limits | Single dataset/seed42; six8-step capacity ends and two100-step fixed ends. Eachcapacity has5 historical updates, actualages1–5, maxima195/176/221 per fold. Fixedbatch200updates have nohistory. M0 does not prove modelcoverage of age8/capacity512 or heldoutperformance. Frozenplan and M0report retain these limits. |
| F. Evaluation type | WARN, wording corrected | M0 label-supervised engineering is real_gt/training-only. Zero-age feature and supervised-loss gradient comparisons are synthetic_proxy/numerical_consistency, not an unsupervised training benchmark or externalgradientGT. T0 constructed-vector math is synthetic_proxy. Finalreport uses these distinctions. |

## Independent complete checks

Reviewer used an independent standard-library read-only script, without remoteaccess, training, inference or filewrites. All248 rows were included, including both100step overfit paths. Initialstates, indices, three-modalpixelSHA, historicalmetadata and reencodingcounts match per pair. Runtime gradient summaries were read, not independently regenerated from modelbackpropagation.

All90 role/update differences exceed duplicate-backward noise; undefinedcosines0. Capacity extra reencoding960+64perend, overfit64perend =>6272; fixed-view drift adds512. Extra historical parameter-gradient witness firstoccurs atstep4 in allsixcapacityends. Per-end cumulative livegrad coverage203/203 means at least once across the run, not everytensor at everystep.

Independent 14-loss scalar weighted-sum discrepancy maximum4.4083e-7. Saved-layout element counts training1,236,480+counterfactual441,344=1,677,824. All1244 aggregate numeric leafvalues match. Bothoverfit analyticfloors and excessratios match. Selectedexpanded Triplet is independently recomputed by the remoteCPU from f32; other13losses remain savedruntime scalars, not independently re-forwarded.

| fold/end | historyupdates | candidateexposures | fresh−stale loss | meanC/T/Mcosine |
|---|---:|---:|---:|---|
| 0/control | 5 | 547 | .0268221229 | .745236945/.607046515/.720808554 |
| 0/fresh_memory | 5 | 547 | .0210182846 | .804543602/.672157097/.816924596 |
| 1/control | 5 | 519 | .0236522704 | .738345730/.641046679/.686857760 |
| 1/fresh_memory | 5 | 519 | .0216437429 | .791545630/.680540729/.771780825 |
| 2/control | 5 | 658 | .0271000028 | .792193818/.656740010/.742365396 |
| 2/fresh_memory | 5 | 658 | .0181581825 | .830921149/.701258266/.817087567 |

All780 T0 metadata rows were independently replayed; predicted Q1 reencodedrecords97600/end match plan. This is metadata/budget evidence, not actualcompletedQ1. Reviewer also checked bound oldsource summary/CPU SHA and oldinstance-memorydiagnostic background numbers, without importing them into newQ1 performance.

Recursive source contracts:53 of58 file references rawbytesSHA match locally;5 established files match afterCRLF→LF, documented in configs/MSVR310/TriFusion-source-style-paired-v1-r2.json:100 and:126 (criterion.py,state.py,builder.py,experts/mamba.py,experts/semantic_residual.py). No other contentmismatch or filemodification found. This localnormalization boundary does not replace remote rawhash guards.

## Reporting corrections and disposition

Reviewer reread revisions that restrict203/203 to cumulativecoverage and distinguish expanded-distance arithmetic from the other13runtime loss scalars. Final requested proxy-type correction has also been applied. No metrics, checkpoint, code or frozencontract changed for these wording fixes. Preserve WARN for remote-binary access, runtimegradient witnesses, single seed, shortM0 and uncompletedQ1. EngineeringM0 permits the alreadyregistered Q1 pipeline; it does not establish scientificqualification or complete the long-termGoal.
