# TDD seam agreement for TriFusion-ReID

Core implementation and tests start only after these public seams are accepted.
The aim is to test observable collaboration behavior without coupling tests to
private layer layouts.

## Proposed seams

1. `TriBranchEncoder(images, modality_mask)`
   - proves all three full experts receive all valid modalities;
   - checks shapes, finite values and gradient reachability per expert.
2. `HeterogeneousRelay(states, reliability, stage)`
   - proves synchronous no-self exchange, normalized valid gates, private
     residual preservation and an actual change in receiving states.
3. `ReliabilityPosterior(states, modality_mask)` and
   `CollaborativeFusion(states, reliability, modality_mask)`
   - proves one joint shared function emits valid Beta parameters for all nine
     entries, invalid modalities have zero mass, weights normalize, and the
     inference seam never constructs a self-referential target.
4. `RoleDirectedPeerTeaching(states, reliability, labels)`
   - proves teacher stop-gradient, direction selection, rejection and role
     payload dispatch.
5. `TriFusionReID.forward(batch, targets=None, return_aux=False)`
   - proves stable train/inference contracts, all missing-mask combinations,
     branch gradients and deterministic tiny-config integration.
6. Official evaluator fixed worked example
   - proves exact CMC/mAP and same-identity/same-camera filtering independently
     of model quality.

The offline reproducibility seam is:

`tools/build_circ_targets.py --config CFG --mode development|postfreeze-final --output DIR`

- emits immutable `targets.jsonl` and `receipt.json`;
- proves every target identity is absent from its frozen generator's training
  fold;
- rejects a primary row without different-camera positive support;
- proves deterministic intervention seeds reproduce the same rows and hashes;
- records sample key, fold, availability mask, nine expert×modality outcomes,
  signed effect statistics, reference-bank provenance, cross-camera-support
  flag, generator checkpoint SHA-256 and intervention seeds;
- records total/direct/relay effects for all nine contributions and exactly one
  hash-selected valid edge per relay stage and query-condition row; the edge is
  audit-only, never a primary training target, and the receipt proves the
  frozen salt/order/validity rule, two-per-row budget and group coverage;
- records helpful/neutral/harmful signed labels separately for each corruption
  family, severity and seed, while exposing only helpful-vs-not-helpful as the
  learned scalar target;
- binds the pre-target protocol/code hash, development versus post-freeze-final
  mode, per-condition BCE/Brier/ECE, overdispersion, empirical concentration
  coverage, identity/query-cluster effective sample size, and per-camera and
  identity-frequency group assignments;
- emits separate query+gallery symmetry and proxy-target-to-deployed-model
  transfer receipts with frozen agreement thresholds;
- development mode never reads the 30 dev identities; final mode may include
  them only as all-171 training rows after configuration freeze and must reject
  any further selection; neither mode reads official-test labels or writes a
  gradient-bearing target.

## Test layers

- CPU unit tests use injected tiny expert/mixer implementations to validate
  algebra and contracts.
- CUDA integration tests exercise the real `mamba_ssm.Mamba` kernel.
- A tiny real-image loader test covers RGBNT201 without embedding dataset paths
  into unit-test fixtures.
- A tiny identity-fold fixture exercises the target-builder CLI and verifies
  zero train/target identity overlap through its emitted receipt.
- A one-batch overfit test is the first learning gate before any full run.

No core model or test file is added until the user explicitly accepts these
seams (reply: `接缝同意`). Environment, dataset and baseline reproduction tools
are already independently validated and are outside this pending core seam.

## V13 deployment-aligned counterfactual extension (accepted scope)

The user's existing `接缝同意` also covers the offline target-builder seam used
by V13. The public behavior added by V13 is intentionally small:

1. `compose_v13_fusion(baseline, modal_residual, weights)`
   - uses one shared blockwise fusion implementation for Q0, Q1 replay and dev;
   - preserves the exact Signal prefix before evaluator normalization;
   - fixes residual energy at `alpha=0.2`.
2. `query_side_counterfactual_utilities(...)`
   - builds one immutable uniform reference bank;
   - removes a slot before residual-bank normalization;
   - returns higher-is-better `full_margin - removed_margin` for all nine slots.
3. `identity_cluster_bootstrap_lower_bound(...)`
   - samples identities as whole clusters with fixed statistical seed 42;
   - never retrains a model and is not a model multi-seed experiment.
4. `tools/build_v13_deployment_aligned_targets.py --mode preflight|q0`
   - pairs identity-OOF teacher target/replay features with the exact all-fit
     Phase-A deployment inputs by ordered sample key;
   - records all checkpoint, feature, ordering and access-count hashes;
   - fails closed before Q1 when target health or action transfer fails.
5. `evaluate_v13_q1_gate(...)` and
   `tools/train_v13_deployment_aligned_router.py`
   - train the fixed-alpha hierarchical Router on paired deployment inputs;
   - replay held-out decisions on identity-OOF teacher features;
   - require per-fold non-inferiority plus positive identity-cluster bootstrap
     lower bounds for expected utility, Top-1, AP and retrieval margin;
   - produce a combined checkpoint only after every Q1 gate passes.

Tests observe only these return values and CLI receipts. They do not inspect
private Router layers or mock project-owned model collaborators.

## Baseline crash-recovery seam (already in baseline-reproduction scope)

The public recovery boundary is:

`tools/run_demo_resumable.py --output-dir DIR [the frozen DeMo options]`

and its state API is:

`save_training_checkpoint(...)` / `restore_training_checkpoint(...)`

- a fresh run records a complete epoch-0 boundary before the first batch;
- `post_train` means training for that epoch is durable and evaluation is the
  next action, while `post_eval` means the next action is the following epoch;
- model, both optimizers, both criteria/scheduler state where applicable, AMP
  scaler, best metrics, and Python/NumPy/CPU/CUDA RNG states are one atomic
  checkpoint;
- baseline commit, resolved configuration, pretrained-weight hash, runtime,
  and recovery-code hashes form a fail-closed run identity;
- a corrupt, incomplete, foreign, or phase-invalid checkpoint is rejected;
- the pinned DeMo checkout and its training/evaluation computations remain
  unmodified. At epoch 10, every candidate state-dict tensor must exactly equal
  the hash-bound original DeMo seed-42 checkpoint before epoch 11 is allowed;
  this early replacement-run parity is the empirical semantic regression gate.

This seam repairs baseline experiment durability only. It does not authorize
tests or implementation of the pending TriFusion model/evaluator seams above.

## PEFT-BoA fixed-endpoint recovery seam (accepted and implemented)

The implemented public boundary is:

`tools/run_peft_boa_resumable.py --output-dir DIR --mode capacity|fixed120`

- `capacity` proves eight real B64/K4 AMP steps, full trainable-gradient
  coverage, finite values and the 8 GiB memory gate without opening the
  official test loader;
- `fixed120` records a complete epoch-0 boundary, then atomic current/previous
  full-state epoch generations containing model, AdamW, scheduler, scaler,
  criteria/center optimizer where constructed, and all RNG states;
- the run identity binds the clean PEFT commit, resolved config, CLIP/data,
  runner/runtime, seed 1111, B64/K4 and worker/device settings;
- official test iteration count remains zero until the fixed epoch-120
  checkpoint is durable; the primary mode then evaluates it exactly once;
- epoch80 and epoch120 model exports are saved before any test access, but
  epoch80 is only a separately labeled published-protocol calibration;
- corrupt, foreign, partial, phase-invalid or model-only upstream checkpoints
  are rejected; no best checkpoint is selected from official-test metrics.

The detailed contract is `docs/PEFT_BOA_REPRODUCTION_SPEC.md`. The exact user
reply `接缝同意` is bound in `evidence/tdd_seam_consent_20260831.json`; the
runner and CLI tests are now implemented. Real capacity/fixed120 execution is
still gated on GPU `memory.used < 500 MiB`.

## MFRNet checkpoint evaluation seam (accepted and implemented)

The implemented public boundary is:

`tools/run_mfrnet_checkpoint_eval.py --mode preflight|official128 --output-dir DIR`

- `preflight` verifies source/config/environment/data/pretrain/checkpoint hashes,
  resolves but does not execute the upstream command, and enforces the strict
  `<500 MiB` launch gate;
- `official128` preserves the released B128, no-rerank, normalized complete-
  modality protocol, records peak VRAM and atomically binds all logs/metrics;
- OOM or CUDA incompatibility is a failed local parity attempt, never an
  automatic lower-batch retry;
- a lower batch is non-comparable because Tutel capacity routing is not
  generally batch-partition invariant.

The detailed contract and deterministic routing counterexample are
`docs/MFRNET_CHECKPOINT_REPRODUCTION_SPEC.md` and
`evidence/mfrnet_eval_batch_semantics_audit_20260831.json`. The exact user
reply `接缝同意` is bound in `evidence/tdd_seam_consent_20260831.json`; the
runner and CLI tests are now implemented. Real `official128` remains gated on
GPU `memory.used < 500 MiB`.
# V14 fold-robust Router public seams

- `cross_camera_retrieval_risk`: L2-normalized hardest cross-camera positive
  versus nearest negative softplus risk.
- `fold_bound_retrieval_risk`: exact V13 fusion plus mandatory single-OOF-fold
  row binding.
- `select_minimax_fixed_slot`: source-fold-only fixed comparator selection.
- `evaluate_v14_q1_gate`: held-out risk/AP/margin and safety/access contract.

These seams were frozen in the V14 READY proposal before implementation. Tests
do not inspect private training helpers.

## V15 counterfactual role-delta exchange public seams

The prior user reply `接缝同意` covers this representation-level successor. Its
public behavior is restricted to:

1. `CounterfactualRoleDeltaExchangeStage(before, after)`
   - exchanges only `after-before` role deltas, synchronously and without
     self-edges;
   - initializes to exact no-exchange parity and exposes bounded directed-edge
     scales/messages as diagnostics.
2. `matched_retrieval_regret_v15(on, off, identities, cameras)`
   - scores pre-BN L2-normalized embeddings with cross-camera batch-hard risk;
   - treats the matched off embeddings as stop-gradient and fixes the registered
     regret coefficient to exactly `1.0`.
3. `SignalPreservingCollaborativeV15.forward_paired(batch)` and
   `.forward(batch, return_aux=...)`
   - reuse the same post-augmentation tensor objects for off/on paths;
   - the off path calls no V15 BN/classifier and mutates no frozen parameter,
     buffer or running statistic;
   - retrieval outputs bypass training-only source-class heads and preserve the
     exact Signal prefix.
4. `evaluate_v15_q1_gate(...)` and the V15 runner receipt
   - bind every CRDE-on result to the exact same-fold frozen V12 checkpoint SHA
     used by its no-exchange comparator;
   - authorize D1 only after all preregistered per-fold, aggregate, bootstrap,
     receiver, state and access gates pass.

Tests observe these return values and receipts. Directed module internals and
private runner helpers are not test seams.
