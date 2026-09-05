# TriFusion-ReID workspace

## Experiment environment

- Runtime: remote Linux GPU server; local WSL2 is SSH transport only and must
  not store the project or run training.
- GPU: one NVIDIA GeForce RTX 3090, 24 GB
- Conda: `/root/miniconda3/bin/conda`; project environment `tri_reid` and
  comparator environment `signal`. The latter was receipted on 2026-09-01 at
  Python 3.10.13, PyTorch 2.1.1+cu118, and CUDA 11.8. The remote host does not
  contain separate `peft_boa` or `mfrnet` environments; either must be created
  and receipted before use.
- Code directory: `/root/autodl-tmp/trifusion-v2/TriFusion-ReID`
- Dataset root: `/root/autodl-tmp/trifusion-v2/data`
- Artifact root: `/root/autodl-tmp/trifusion-v2/artifacts`
- W&B: false

- MSVR310 was installed on 2026-09-05 from the ICPL-ReID author link and
  SHA/ZIP-CRC/modality-pair verified: train1032/query3 591/gallery1055 triplets.
  It has no project training or retrieval-evaluation result yet. RGBNT100 is
  installed and SHA/ZIP-CRC/modality-pair verified: train8675/query1715/
  gallery8575, with disjoint 50-train/50-test identities and 768x128 montages.
  Its training and retrieval evaluation counts are also zero. Keep RGBNT201 as the
  active model experiment until its fixed main gate is resolved.

## Research provenance

- Implementation base: official DeMo repository, AAAI 2025, MIT licensed.
- High-metric licensed baseline: Signal commit `cd1b0a6`, MIT licensed;
  the local 141-fit/30-dev floor is 58.0109/57.4545 from the exact 3072D
  path, while 80.3/85.2 is an upstream official-test log only and must not be
  called locally reproduced.
- Measurable checkpoint anchor: official MDReID RGBNT201 checkpoint. Its pinned
  repository has no
  repository-level license, so keep it isolated and do not copy its code.
- Source-visible training comparator: PEFT-BoA commit `d2b198b`; its pinned
  repository has no repository-level license. Use only the isolated `peft_boa`
  environment and fixed-epoch protocol; do not call it open source or copy its
  code into TriFusion.
- Supplemental checkpoint comparator: MFRNet commit `ec54a13`; use the
  isolated `mfrnet` environment and label its released best as test-selected.
  Its pinned repository likewise has no repository-level license.
- Preserve the official baseline path and metrics; put new architecture behind explicit config flags.
- The latest user direction supersedes the earlier no-baseline-reproduction
  constraint: first establish a remote-only Signal baseline floor. Keep the
  upstream `80.3/85.2` label until a local checkpoint and same-protocol result
  have actually been verified.
- Do not report SOTA unless the same dataset split, input resources, inference protocol, and metrics have been reproduced and audited.

## TriFusion implementation status

- The exact `接缝同意` reply is receipted before core implementation.
- The v2 build runs one complete shared 12-layer CLIP visual trunk and then
  three 768-wide CNN, Transformer, and Mamba semantic-residual experts.
- Reliability-conditioned HFER performs two stage-wise bidirectional exchanges;
  the final quality posterior is refreshed from the collaboratively enhanced
  expert features before identity fusion.
- The real RTX 3090 B32/K4 capacity gate passes: 92,682,578 parameters, AMP
  initial scale 512, zero overflow events, finite gradients for 366/366 trainable
  tensors, and 6,444 MiB peak reserved memory.
- The full CIRC/URGC development and post-freeze configuration chain is defined.
  The CIRC protocol must be re-anchored to the final committed v2 source.
- Capacity and overfit gates are train-only engineering evidence, not retrieval
  metrics, and never support a SOTA claim.
- The exact Signal seed42 B64/K8 run completed 50/50 epochs on the frozen
  141-fit/30-dev protocol. The deterministically reloaded best checkpoint is
  `1f5c200c...66c3` with 58.0109 mAP / 57.4545 Rank-1 and official-test access
  zero. This is the V5 development floor, not an official-test reproduction.
- Signal-preserving V5 readiness passes on the remote RTX3090: exact 825/825
  baseline feature/metric parity, real B32/K4 8-step capacity with 213/213
  trainable gradient tensors and 3,542 MiB peak reserved memory, and a
  100-step fixed-batch loss ratio of 0.02102. Official-test access is zero.
- Signal-preserving V5 completed its only seed-42 60-epoch held-out-dev run.
  Selected epoch 51 gives baseline/fused/CNN mAP 58.0109/58.0168/58.0181;
  fused fails both the best-expert and 65 mAP gates. The frozen Signal state
  stayed unchanged and official-test access stayed zero. Do not launch V5 on
  the official test and do not start ablations.
- The V5 read-only diagnostic shows a 0.02747 fused-suffix/baseline norm ratio,
  distance correlation 1.0 and 99.9879% Top-10 overlap versus baseline. The
  next main version must materially change retrieval geometry while preserving
  the exact Signal output; do not scan hyperparameters or residual multipliers.
- V6 is the single diagnostic-driven main-only correction: it removes the V5
  learned residual scale, deterministically matches the routed residual-bank
  norm to the exact Signal norm per sample, and trains/routes from residual-only
  identity embeddings so the frozen baseline cannot satisfy expert objectives.
  It is not an ablation or multiplier scan. Preflight, real B32/K4 capacity and
  fixed-batch overfit all pass: exact 825/825 baseline parity, 218/218 gradient
  tensors, 3,554 MiB peak reserved, zero overflow and loss ratio 0.05655.
  The only seed-42 60-epoch held-out-dev run completed and selected epoch 8:
  baseline/fused/CNN mAP is 58.0109/58.7321/59.1022. Fused improves baseline
  by 0.7212 but trails CNN by 0.3701 and misses the 65 mAP gate by 6.2679.
  Official test, ablations and multiple seeds remain forbidden.
- The V6 read-only diagnostic shows all residual-bank norm ratios equal 1.0,
  fused/baseline distance correlation 0.96875 and Top-10 overlap 95.3939%.
  Residuals are diverse, but the router stays high entropy (0.97435) and gives
  the strongest CNN expert the lowest mean weight. The next allowed main-only
  change must target marginal-gain routing alignment while retaining exact
  Signal preservation; generalization is a secondary issue. Do not tune epoch,
  batch size, learning rate, residual multiplier or run an ablation matrix.
- The V6 ground-truth Oracle is diagnostic only: branch Oracle is 63.6089 mAP,
  4.5067 above the strongest fixed branch, and all three experts have positive
  leave-one-out marginal mAP. V7 therefore retains CNN/Transformer/Mamba as
  shallow heterogeneous residual experts over the frozen Signal feature field;
  do not describe them as three independent full backbones.
- V7 is the one permitted main-only structural correction. It uses shared
  geometric augmentation, matched-token residuals, hierarchical
  `P(modality) * P(expert | modality)` routing, per-slot L2-normalized marginal
  identity gain, controlled-degradation quality supervision and bounded sample
  alpha. Its exact-parity preflight, real B64/K8 two-view capacity (222/222
  gradient tensors, 11,486 MiB peak reserved) and overfit gate pass. Its sole
  seed-42 60-epoch dev run completed and passed the epoch-10 corruption gate.
  Selected epoch 1 gives baseline/fused/Mamba mAP
  58.0109/58.3293/58.3476. Fused gains 0.3184 over baseline but trails Mamba
  by 0.0183 and misses 65 by 6.6707, so the main gate fails. Joint training
  degrades fused mAP to 57.7550 at epoch 60; do not rerun V7, start ablations,
  use multiple seeds or access the official test.
- The V7 read-only diagnostic records router entropy 0.99791, modality entropy
  0.99994, nearly constant alpha 0.198947, 14.0625% predicted/target top-slot
  agreement, and 99.6364% Top-10 overlap with baseline. Residual-only Oracle
  reaches 62.7435 mAP, 3.6118 above the strongest fixed residual, so the
  demonstrated bottleneck is learned routing plus destructive joint
  optimization, not absent expert diversity. Any next main version requires a
  single evidence-derived structural hypothesis and fresh train-only gates.
- The optimizer-free V8 frozen-router probe is terminal and must not be rerun.
  Among the 21 cross-camera-eligible fit identities (571 queries), the
  residual-expert winner is CNN for 100% of queries, so the fit utility teacher
  cannot learn sample-level expert diversity. On disjoint dev identities it
  only matches the 55.27% CNN majority policy, while the V7 Router reaches
  27.39%. Equal-energy uniform/teacher fusion reaches 59.6188 mAP, still 5.3812
  below 65. Do not implement a Router-only V8; the next main change must improve
  expert representation/task division while preserving exact Signal.
- V8 Phase-A implements that representation change. It branches after frozen
  CLIP block 8, reuses frozen pretrained tail blocks 9/10/11 in all three
  paths, and adds role-specific CNN local-detail, Transformer global-CLS, and
  Mamba spatial/cross-modal residual heads. Router and HFER are disabled during
  formation. Exact preflight, real B64/K8 capacity (203/203 gradients, 6006 MiB
  reserved) and 100-step overfit (excess-loss ratio 0.000534) pass.
- The only V8 Phase-A seed42 probe trained 20 epochs/840 steps and evaluated
  held-out dev once at the final epoch. Fixed fused is 58.0972 mAP and is not a
  deployable gain. Branch GT Oracle is 64.7850 mAP (+6.7741 over the strongest
  fixed output), and residual-only Oracle is 63.4813 (+9.6153); all experts
  have unique AP wins and positive leave-one-out margins. Oracle is diagnostic
  and non-deployable. Result-to-claim is partial/medium; V8 audit is WARN due
  only to remote large-artifact packaging. Official access remains zero.
- The next authorized step is exactly one frozen-expert, fit-only hierarchical
  Router feasibility phase. It must not use dev Oracle labels for training,
  tuning or checkpoint selection, and must pass missing/corrupted-modality
  quality gates. Do not enable HFER until the learned Router proves deployable
  gain and preserves complementarity. Branch Oracle is still 0.2150 below 65,
  so hard expert selection alone cannot satisfy the dev gate.
- V8 Phase-B replaced saturated OOF AP labels with continuous identity margins
  from the same three expert folds. The read-only 571-query target gate passed:
  unique slot winners are CNN/Transformer/Mamba `38/350/183` and RGB/NI/TI
  `215/59/297`; slot-Oracle margin exceeds the best fixed slot by `0.164303`.
  Three identity-disjoint Router folds plus one all-fit refit used 400 Router
  optimizer steps while Phase-A experts stayed byte-stable. The OOF learned
  margin `0.102034` barely exceeds fixed `0.101720`, and Top-1 `17.8634%`
  barely exceeds majority `17.6883%`; all three blur-response and missing-zero
  quality gates pass. Treat this as weak feasibility evidence, not a strong
  routing-generalization claim.
- The one frozen V8 Phase-B dev evaluation is terminal: baseline/fused mAP is
  `58.0109/58.4050`, and fused strictly beats CNN/Transformer/Mamba
  `57.6071/56.3031/56.6260`; Rank-1 is `57.4545/59.3939`. This supports a
  narrow same-checkpoint deployable gain of `+0.3941 mAP/+1.9394 Rank-1`, but
  fused misses the 65 mAP gate by `6.5950`. Promotion and next-phase gates are
  false. Do not enable HFER, run ablations/multiple seeds, access official test,
  or scan Router/alpha/epoch/LR settings. Any successor requires a new
  representation-level main hypothesis and fresh train-only gates.
- Independent V8 Phase-B integrity audit is `WARN`, not a result-logic fail:
  GT provenance, score normalization, live code path, fit/dev scope and
  evaluation-type classification pass. The warning is that large checkpoint
  and cache artifacts remain remote-only and cannot be re-hashed from a fresh
  local clone. Keep all Phase-B claim boundaries unchanged.
- V9 Orthogonal Triadic Relay Synthesis is terminal and failed. Its exact
  Signal / Phase-B prefixes and frozen states pass, and the two-round peer
  relay is numerically orthogonal, but the sole final-only seed42 dev result is
  fused `56.5339 mAP / 57.2121 Rank-1`, below exact Signal by `1.4770 mAP`
  and below Phase-B by `1.8711 mAP`. It misses 65 by `8.4661 mAP`.
- V9 independent result-to-claim is `no/high`; integrity audit is
  `WARN/warn/FAIL_TO_PROMOTE`. Seal V9: no official test, ablation, multi-seed,
  checkpoint choice or beta/epoch/LR/residual scan. A successor must be a new
  representation-level hypothesis and pass a fit-only identity-disjoint
  positive-retrieval-utility gate before another final-only dev access.
- V10-Q0 tested frozen DINOv2 only on 21 cross-camera fit identities / 571
  queries with optimizer0/dev0/official0. Phase-B is saturated at 100 mAP,
  DINOv2 is 7.6284, fixed equal-block concat is 92.2120, Oracle gain is zero,
  and unique AP wins are Phase-B/DINO=`571/0`. Qualification is false.
- Seal V10: do not implement Q1, train, access dev, or scan DINO modality,
  resolution, block, token, weighting or head choices. This result rejects only
  the fixed V10 representation/protocol, not DINOv2 in general. Any future DINO
  route must be a new hypothesis with a non-saturated identity-isolated gate.
- V11-Q0 corrected the distance aggregation to three fold-local held-out expert
  evaluations and excluded explicit Signal/Phase-B embeddings, but the residual
  bank and Transformer residual still reach 100 mAP because every expert reads
  a frozen Signal token field trained on all 141 fit identities. DINOv2 reaches
  14.1323 mAP, fixed concat 95.8582, Oracle gain zero, and DINO unique AP wins
  zero. The intended non-saturation gate is false.
- Seal V11: the 100 mAP value is leakage/saturation evidence, not a deployable
  result or metric fraud. Do not implement Q1/Q2, train, access dev/official,
  run ablations, or scan DINO choices. Any successor must isolate the complete
  measured feature path from held-out identities without rerunning baseline.
- V12-Q0 repaired the complete-path identity leakage by training a raw-CLIP
  Signal teacher and the three experts only on the other 94 fit identities in
  each fold. Across 571 held-out-fit queries, fixed CNN/Transformer/Mamba
  residuals reach `83.7717/86.9549/85.8870 mAP`, residual bank reaches
  `87.9968`, and diagnostic hard Oracle reaches `92.2679`, or `+5.3130 mAP`
  over the strongest fixed expert. All experts and modalities have unique
  utility wins, so the preregistered Q0 qualification passed.
- V12-Q1 reused the exact V8 Phase-A checkpoint, Router structure and all
  Router hyperparameters, changing only the complete-path OOF cache. It failed
  the train-only gate: learned/fixed OOF margin is
  `-0.117330/-0.099975`, and learned/majority Top-1 is
  `12.2592%/16.8126%`. Quality corruption and missing-zero gates pass, but no
  final refit or combined checkpoint was produced; dev/official access is zero.
  Seal V12 without dev, official, ablations, multiple seeds, HFER or tuning
  scans. Current deployable dev best remains V8 Phase-B `58.4050 mAP /
  59.3939 Rank-1`.
- V13-Q0 aligned the supervision with the deployed fixed-alpha fusion path and
  paired every identity-OOF teacher row with the frozen all-fit Phase-A Router
  input. Its train-only gate passed: expert unique positive wins are
  CNN/Transformer/Mamba `218/196/157`, modality wins RGB/NI/TI
  `241/109/221`, Oracle-minus-fixed utility is `+0.0014682`, and the one-shot
  read-only action-transfer gain is `+0.0008706` with all folds non-inferior.
- V13-Q1 is terminal and failed. Three Router folds used 300 optimizer steps;
  Phase-A stayed byte-stable, all quality/missing-modality gates passed, and
  dev/official access remained zero. Fold0 failed Top-1 non-inferiority; fold2
  failed expected utility, replay AP and replay margin. The 95% identity-cluster
  bootstrap lower bounds are negative for utility `-0.0004691`, Top-1
  `-0.0396049`, AP `-0.0081192`, and margin `-0.0028545`.
- V13 produced no final refit or combined checkpoint and is sealed as
  `Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE`. Independent result-to-claim is
  `no/high`; integrity is `WARN/warn` only for remote-artifact packaging and
  tracker staleness, not metric fraud or leakage. Do not run V13 dev, official,
  ablations, multiple seeds, or fold/epoch/LR/temperature/threshold scans.
  V8 Phase-B remains the current deployable best at `58.4050 mAP / 59.3939
  Rank-1`; any successor needs a new preregistered train-only hypothesis.
- V14 replaced V13 pointwise utility-KL with fold-local cross-camera retrieval
  risk and worst-source-fold regret. Q0 passed with optimizer0, dev0/official0,
  no cross-fold feature distances, 14/14 finite nonzero Router gradients and an
  unchanged Phase-A SHA. The sole seed42 Q1 then failed: fold0 AP gain is
  `-0.0005571`; fold2 risk/margin gains are `-0.0016102/-0.0033642`; bootstrap
  lower bounds for risk/AP/margin are all negative. Quality and missing-modality
  gates pass, but there is no final refit/checkpoint/dev. Seal V14 without any
  LR/epoch/loss/fold/margin scan, ablation, multi-seed or official evaluation.
  Current deployable best remains V8 Phase-B `58.4050/59.3939` mAP/Rank-1.
- V15 CRDE completed its only three-fold seed42 Q1 and failed the scientific
  gate: aggregate fused/CNN/Transformer/Mamba gains were
  `-0.1721/-0.1576/-0.2606/+0.2898` mAP and fused bootstrap lower bound was
  `-0.9503`. It has no D1/dev/official result and is sealed without tuning.
- V16 SATR implemented training-only two-peer relation repair with zero new
  inference modules. Its B64/K8 capacity and 100-step gates passed, but formal
  fixed-initial coverage was zero for Transformer in all three folds, so M0
  failed and Q1/D1/dev/official were not run. Do not relax the registered
  thresholds or replay V16 under another worker/RNG draw. The current
  deployable best remains V8 Phase-B `58.4050/59.3939`. Independent audit is
  `WARN/warn` for the unbound proposal-time threshold receipt only; formal M0
  is a valid negative result. Result-to-claim is `no/high`, routing
  `M0_FAIL_Q1_BLOCKED`.

## Remote experiment constraints

- V21 is preregistered and not yet launched. It keeps the original V8/Signal
  inference model and seven ID/triplet objectives. Compare ordinary AdamW 40
  epochs with SAM 20 epochs at fixed rho 0.05, seed42, B64/K8: each arm has
  3360 forward/backward pairs across three folds, while optimizer updates are
  3360/1680. This is not equal epochs, updates, data exposure or wall time.
  Read refine-logs/v21/EXPERIMENT_PLAN.md before execution. T0 CUDA math and
  M0 pairing/capacity/100-step overfit must pass before the full six-endpoint
  Q1. Preserve full 3126 gallery/571 queries, all five outputs, final-only
  checkpoints and all five scientific gates. No rho/epoch/LR/seed scans.

- V20 cross-modal identity supervision completed all three folds x two endpoints
  x20 epochs, seed42, 3360 optimizer steps, and is sealed as Q1_FAIL. Candidate/
  actual matched-control fused mAP is 79.195387/80.206258, gain -1.010871;
  fold gains are -1.087608/-2.539986/+0.416314. CNN/Transformer/Mamba gains
  are -1.009739/-4.779791/+1.306367, and the 21-identity bootstrap lower bound
  is -3.812656. All 3126 gallery records and 571 eligible queries were retained.
  No D1/dev/official, temperature/loss-weight/branch/epoch/seed scans or V20
  retraining. Whole-file remote SHA and complete local NumPy arithmetic passed;
  independent Q1 audit is complete: engineering PASS, integrity WARN, scientific
  FAIL. Preserve the provenance limits in EXPERIMENT_AUDIT_V20_Q1.md/json.
  Any successor needs a new evidence-grounded hypothesis and preregistered
  train-only qualification. Current deployable best and unmet objective are unchanged.

- V19 private semantic tail completed three folds x two endpoints x20 epochs,
  seed42, 3360 optimizer steps, and is sealed as Q1_FAIL. Trained/frozen fused
  mAP is 80.496828/80.240792, gain +0.256035; fold gains are
  +1.619524/-0.900867/-0.001279, Mamba aggregate gain -0.422824,
  and the 21-identity bootstrap lower bound is -1.615129. All 3126 gallery
  records and 571 eligible queries were retained. No D1/dev/official, private
  tail layer/width/LR/epoch/seed scans or V19 retraining. Read the V19 result
  report and frozen plan; any successor needs a new representation hypothesis
  and preregistered train-only qualification. Current deployable best remains
  V8 Phase-B 58.4050 mAP / 59.3939 Rank-1; the main objective is unmet.


- V18 PVNP completed all three folds x two endpoints x20 epochs, seed42,
  and is sealed as Q1_FAIL. Projected/uncentered full-gallery fused mAP is
  81.482001/80.560497, gain +0.921504, below the frozen +1.0 gate; the
  21-identity bootstrap lower bound is -0.117338. Every fold and all expert
  aggregate gains are positive, but the scientific gate still fails.
  No D1/dev/official, direction/rank/epoch/LR scans or V18 retraining.
  Read the complete result report and frozen V18 plan. Any successor
  requires a new representation hypothesis and preregistered train-only
  qualification, preserving the same complete-gallery and seed42 rules.

- V17 DTRED completed all three folds, two endpoints, and 20 epochs per
  endpoint (3360 steps). M0 passed; Q1 failed with fused matched gain
  -0.338635 mAP. Its six final checkpoints were strictly reload-verified on
  2026-09-05 using all 3126 held-out gallery records and all 571 eligible
  cross-camera queries: fused DTRED/weight0 is 80.286024/80.614939 mAP,
  gain -0.328915, bootstrap lower bound -1.124616. These are fit-only OOF
  results, not dev/official metrics. V17 has no D1 and remains sealed without
  training/width/loss/LR/epoch/checkpoint scans.
- Future complete-gallery evaluation must retain single-camera held-out
  identities as gallery distractors while excluding only queries with no
  valid cross-camera positive from the metric denominator. The fixed 141-fit
  registry has 3126 triplets, distinct from the supplied train_141 directory.
  Preserve both the original Q1 and supplementary full-gallery receipts;
  do not choose between their branch results for a favorable claim.

- Check the remote RTX 3090 before every launch. The V7 `rtx3090_b64k8` gate
  requires at least 22,000 MiB free before a new training process starts.
- Use real B64/K8 without gradient accumulation so batch-hard losses see eight
  identities. V7 uses AMP and separate clean-ReID/controlled-degradation views.
- Run only seed 42. Signal baseline work is now explicitly authorized, but only
  on the remote GPU and without hyperparameter or epoch selection on the
  official test. Preserve its full 3072D direct-plus-SIM retrieval feature,
  camera SIE and baseline-only output; do not call the 1536D projected-CLS
  anchor a Signal reproduction.
- Any fused successor must emit baseline-only and fused embeddings from the
  same checkpoint. Fused is promoted only if it is not worse than baseline on
  the frozen development protocol; otherwise fused is rejected and no
  fusion-gain claim is allowed. Do not add runtime fallback logic unless a
  later explicit requirement and test justify it.
- Start with RGBNT201. Do not run ablations until the frozen same-protocol main
  target (85.3 mAP / 87.9 Rank-1) has been exceeded.
- Mamba CUDA extensions are source-locked to the receipted SM86 builds; do not
  replace them with unverified wheels.
