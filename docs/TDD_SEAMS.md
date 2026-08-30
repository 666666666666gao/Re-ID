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
   - proves Beta parameters are valid, invalid modalities have zero mass,
     weights normalize, counterfactual targets are detached and degradation
     can reduce the corresponding weight.
4. `RoleDirectedPeerTeaching(states, reliability, labels)`
   - proves teacher stop-gradient, direction selection, rejection and role
     payload dispatch.
5. `TriFusionReID.forward(batch, targets=None, return_aux=False)`
   - proves stable train/inference contracts, all missing-mask combinations,
     branch gradients and deterministic tiny-config integration.
6. Official evaluator fixed worked example
   - proves exact CMC/mAP and same-identity/same-camera filtering independently
     of model quality.

## Test layers

- CPU unit tests use injected tiny expert/mixer implementations to validate
  algebra and contracts.
- CUDA integration tests exercise the real `mamba_ssm.Mamba` kernel.
- A tiny real-image loader test covers RGBNT201 without embedding dataset paths
  into unit-test fixtures.
- A one-batch overfit test is the first learning gate before any full run.

No core model or test file is added until the user explicitly accepts these
seams (reply: `接缝同意`). Environment, dataset and baseline reproduction tools
are already independently validated and are outside this pending core seam.
