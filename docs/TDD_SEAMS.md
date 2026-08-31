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
