# TriFusion-ReID implementation blueprint v1

Status: implementation-ready design, not an implementation. This document
makes no SOTA claim. The public seams in TDD_SEAMS.md are still awaiting the user's exact
reply 接缝同意. Until then, this document may be refined, but no core model or
core-model test file is added.

This blueprint translates METHOD_SPEC_V1.md into one concrete implementation
for an 8 GiB RTX 4060 Laptop GPU. METHOD_SPEC_V1.md remains authoritative if a
conflict is discovered.

## 1. Reproducibility pins and resource track

The primary resource track is fixed as follows:

| Item | Frozen value |
|------|--------------|
| Dataset | RGBNT201, official train_171 and official test query/gallery |
| Train-only selection | rgbnt201_dev_v1: 141 fit identities, 30 dev identities |
| Modalities | RGB, NI, TI in that exact order |
| Input | 256 by 128, three channels per modality |
| Pretraining | OpenAI CLIP ViT-B/16 image weights only |
| Baseline source | DeMo commit b4f323a430b32e3a1637c3e7acb25868cb52e9cd |
| Baseline training | 50 epochs, seed 42, B32, K4, test batch 64 |
| Primary metric | official no-reranking mAP and CMC Rank-1 |
| Hardware | one RTX 4060 Laptop GPU, 8,188 MiB visible memory |
| Environment | conda tri_reid, Python 3.10, PyTorch 2.5.1+cu121 |
| Mamba kernel | mamba-ssm 2.2.6.post3 and causal-conv1d 1.6.0, SM89 smoke passed |

The frozen DeMo epoch-10 state has 417 tensors and 98,816,543 parameters,
occupying 395,266,296 raw tensor bytes. Its real B32/K4 eight-step probe peaked
at 6,894.18 MiB. These are measured reference values, not estimates.

The DeMo implementation uses one shared CLIP image backbone three times, then
forms a 1,536-dimensional direct feature, a 3,584-dimensional seven-token MoE
feature, and a 5,120-dimensional joint evaluation feature. TriFusion does not
inherit these output widths or the legacy return-pattern interface. Its frozen
primary fused embedding is 512-dimensional, as specified in METHOD_SPEC_V1.

CNN and Mamba bodies start from random initialization. They must not import a
second supervised pretraining dataset. A variant with additional pretrained
weights belongs to a separately labelled resource-enhanced track.

## 2. Design objective and deletion test

TriFusion is one deep module whose external seam is:

    TriFusionReID.forward(batch, targets=None, return_aux=False)

The caller supplies images, a modality mask, and optional training targets.
The module owns tokenization, three complete experts, two synchronous relays,
one shared reliability posterior, role-directed peer teaching, fusion, and
named auxiliary results. Deleting the module would force the trainer,
evaluator, missing-modality logic, and ablations to reimplement those rules;
therefore the interface earns its depth.

Internal seams are introduced only where two real adapters exist:

- StageCollaborator has NoOpCollaborator and HFERCollaborator adapters.
- ReliabilityGate has UniformGate, SoftmaxGate, and EvidentialGate adapters.
- ExpertFactory has TinyExpertFactory for CPU contract tests and
  ProductionExpertFactory for the real CNN, Transformer, and Mamba experts.

These adapters remain internal to the TriBranchEncoder implementation. They
are not passed through every call site and do not enlarge the external seam.

## 3. Proposed package layout

The frozen upstream baseline clone remains untouched. New implementation lives
in the research repository:

    modeling/trifusion/
        __init__.py
        state.py
        tokenizer.py
        encoder.py
        relay.py
        reliability.py
        peer_teaching.py
        fusion.py
        model.py
        criterion.py
        experts/
            __init__.py
            cnn.py
            transformer.py
            mamba.py
            tiny.py
    engine/
        trifusion_processor.py
    configs/RGBNT201/
        TriFusion.yml
    tools/
        train_trifusion.py
        eval_trifusion.py
        profile_trifusion.py

The only public names exported by modeling.trifusion are the six frozen seams
and their result types:

    TriBranchEncoder
    HeterogeneousRelay
    ReliabilityPosterior
    RoleDirectedPeerTeaching
    CollaborativeFusion
    TriFusionReID

Their callable interfaces remain exactly:

    TriBranchEncoder(images, modality_mask) -> ExpertStateMap
    HeterogeneousRelay(states, reliability, stage) -> RelayResult
    ReliabilityPosterior(states, modality_mask) -> ReliabilityResult
    RoleDirectedPeerTeaching(states, reliability, labels) -> PeerTeachingResult
    CollaborativeFusion(states, reliability, modality_mask) -> FusionResult
    TriFusionReID.forward(batch, targets=None, return_aux=False)
        -> Tensor or TriFusionOutput

Private mixers, projections, scans, losses, and configuration helpers are not
re-exported.

The package builder injects an ExpertFactory, a StageCollaborator, and a
ReliabilityGate when it constructs TriBranchEncoder. Production and tiny-test
adapters therefore use the same encoder interface, while training and
evaluation callers see only the two-input call above.

## 4. Input, mask, and state contracts

### 4.1 Batch contract

The model accepts either the official loader mapping or a normalized batch
object. The canonical form is:

| Field | Type and shape | Invariant |
|-------|----------------|-----------|
| images.RGB | float tensor B,3,256,128 | normalized with the frozen transform |
| images.NI | float tensor B,3,256,128 | same spatial size as RGB |
| images.TI | float tensor B,3,256,128 | same spatial size as RGB |
| modality_mask | bool tensor B,3 | columns are RGB, NI, TI |
| targets | optional long tensor B | required only for supervised losses |
| camids | optional long tensor B | training metadata, never test selection |

The first operation validates that every sample has at least one true mask
entry. An all-missing row raises ValueError with the failing row indices.
Pixel values alone never infer availability.

Valid modality slots are packed before an expert body and scattered back after
it. This prevents invalid zeros from contaminating BatchNorm statistics or
contributing gradients. Internal CNN and Mamba blocks use GroupNorm or
LayerNorm; only final ReID necks use BatchNorm.

### 4.2 Canonical tensor shapes

Let E=3 experts, M=3 modalities, N=128 patch tokens, Dv=768, Dh=256,
Dr=64 relay channels, and Dz=512 metric channels.

| Representation | Shape |
|----------------|-------|
| stacked pixels | B,M,3,256,128 |
| modality mask | B,M |
| packed CLIP patch tokens | Nv,N,Dv |
| Transformer stage state | B,M,N,Dv |
| CNN stage state | B,M,N,Dh |
| Mamba stage state | B,M,N,Dh |
| relay shared state | B,E,M,N,Dr, represented as an expert-keyed map |
| reliability alpha, beta, r, u | B,E,M |
| projected contribution embeddings | B,E,M,Dz |
| per-expert collaborative embeddings | B,E,Dz |
| fused metric embedding | B,Dz |
| peer direction gates | B,E,E, with zero diagonal |

Nv is the number of valid sample-modality slots in the current mini-batch.
No unrestricted squeeze call is permitted; every dimensional reduction names
its dimension.

### 4.3 ExpertStateMap

ExpertState is a typed result containing:

- tokens: B,M,N,D tensor in the expert's native width;
- global_embedding: B,M,D tensor derived from valid tokens;
- private_embedding: B,M,Dp tensor used by the diversity objective;
- role_payload: compressed, role-specific relational evidence;
- modality_mask: the exact bool tensor used for packing;
- stage: an integer in 1, 2, 3;
- expert: exactly cnn, transformer, or mamba.

ExpertStateMap is an immutable mapping with exactly those three expert keys.
Its keys method and iteration never expose metadata keys. Read-only metadata
properties carry the one ReliabilityResult, the two RelayResult summaries, the
modality mask, and optional StageTrace diagnostics. This lets
TriFusionReID pass the same posterior to CollaborativeFusion without mutable
module caches or an undocumented second return value. The state object exposes
no internal layer list, and callers cannot drive the implementation one private
block at a time.

## 5. Shared tokenizer and fair initialization

All three experts consume the same CLIP ViT-B/16 patch grid. The tokenizer is
a common input adapter, not the Transformer expert:

1. apply the frozen-compatible CLIP patch convolution with kernel and stride
   16 to obtain a 16 by 8 grid;
2. flatten to 128 tokens and add the frozen CLIP positional embedding;
3. add learned modality and expert embeddings only after the common patch
   representation is formed;
4. project 768 to 256 for CNN and Mamba; keep 768 for Transformer.

The Transformer expert receives the pretrained CLIP class token, pre-LayerNorm,
12 Transformer blocks, post-LayerNorm, and projection to 512. The common patch
convolution is shared by reference, not copied three times.

The CNN and Mamba projections and bodies are randomly initialized. A gradient
test must prove that each body changes its own output; the design must not
degenerate into three heads on a frozen shared Transformer representation.

## 6. Three complete heterogeneous experts

Every expert processes every valid modality using tied weights within that
expert. Expert weights are not tied across architectures.

### 6.1 CNN expert

The CNN expert has width 256 and nine residual local-mixing blocks grouped into
three stages of three blocks. Every block preserves the 16 by 8 grid and uses:

- a 1 by 1 channel expansion;
- depthwise 3 by 3, depthwise 5 by 5, or dilated depthwise 3 by 3 mixing;
- gated channel projection;
- GroupNorm and residual stochastic depth.

The receptive-field pattern within each stage is 3 by 3, 5 by 5, dilated
3 by 3. No downsampling is used because person boundary and part alignment
must remain token-compatible with the other experts.

The role payload is a compact local-neighbour and horizontal-part affinity:
four stripe descriptors plus immediate 4-neighbour cosine responses. It is
computed in linear memory, not as a full 128 by 128 matrix.

### 6.2 Transformer expert

The Transformer expert is the full CLIP ViT-B/16 image tower:

- native width 768;
- 12 pretrained residual attention blocks;
- interaction depths after blocks 4, 8, and 12;
- original CLIP post-LayerNorm and 768-to-512 projection.

The class token is retained inside this expert. Patch tokens are retained at
all three stage exits for relays. The role payload contains class-to-patch
attention or its output-equivalent global-to-patch relation and the branch ID
logits. It does not expose every attention matrix to the outer module.

### 6.3 Mamba expert

The Mamba expert has width 256 and nine residual Mamba blocks grouped into
three stages. Its production block parameters are:

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| d_state | 16 |
| d_conv | 4 |
| expand | 2 |

At each block, one parameter-shared Mamba core processes four orderings:
row-major forward, row-major reverse, column-major forward, and column-major
reverse. Outputs are restored to the canonical 16 by 8 grid and combined by a
four-way content gate. Sharing the core across directions preserves the
linear-token-complexity argument and controls parameters.

The role payload is the difference and consensus between forward/reverse and
row/column context descriptors. Reverse operations use explicit flip
dimensions.

### 6.4 Standalone branch guarantee

Each expert has its own 512-dimensional metric projection, BatchNorm neck,
classifier, ID loss, and triplet loss. CNN-only, Transformer-only, and
Mamba-only configurations bypass relay and cross-expert routing but retain
the same tokenizer and branch head. Each post-collaboration branch embedding
is also evaluable in the full model.

## 7. Causal collaboration schedule

One forward pass follows this acyclic schedule:

    validate and pack valid modality slots
    shared CLIP-compatible tokenization
    run stage 1 of all three experts from one immutable stage snapshot
    predict one URGC posterior from pre-relay stage-1 evidence
    apply synchronous HFER relay 1
    run stage 2 of all three experts
    apply synchronous HFER relay 2 using the same URGC posterior
    run stage 3 of all three experts
    form branch heads and contribution embeddings
    fuse with the same URGC posterior
    compute RDPT only when targets are present and training is true

The posterior is predicted once because the claim says the same evidential
belief controls relay bandwidth, final fusion, and missing/degraded
suppression. Its counterfactual target is produced from final head
contributions, so the stage-1 router receives supervision about downstream
identity-margin usefulness without a second backbone pass.

HFER-uniform replaces the reliability adapter with UniformGate but leaves the
same schedule intact. A no-relay adapter makes A0/A1 actual alternatives at
the same seam rather than conditionals copied throughout the experts.

## 8. Innovation I implementation: HFER

HeterogeneousRelay receives an immutable ExpertStateMap, one ReliabilityResult,
and stage 1 or 2. It returns RelayResult containing a new ExpertStateMap and
observable gate summaries.

For each expert and modality:

1. normalize the native state;
2. project it to a rank-64 shared state;
3. reconstruct the shared component in native width;
4. define the private residual as original minus reconstructed shared state;
5. compute a source-role message from the source state and its masked
   cross-modal consensus;
6. project each source message into the target native width;
7. sum only j-to-e messages with j not equal to e;
8. add the message through a learnable target scalar gamma initialized to zero.

All source messages for a relay are computed before any target update.
Therefore iteration order cannot alter outputs. Gates are zero for invalid
modalities, zero on self edges, and normalized over valid source experts.
When no source is valid for one target-modality position, the message is zero
and the private residual path remains valid.

Role mixers remain small:

- CNN source: depthwise local convolution in rank-64 space;
- Transformer source: low-rank global-to-token attention;
- Mamba source: bidirectional rank-64 selective mixing.

The relay test surface observes input/output states, gates, private residual
energy, and gradients. Tests do not inspect individual projection weights.

## 9. Innovation II implementation: URGC

ReliabilityPosterior consumes the three pre-relay stage-1 states and the bool
mask. For every expert-modality contribution it forms evidence from:

- masked global token statistics;
- cross-modal agreement inside the expert;
- cross-expert agreement for the same modality;
- signal quality statistics such as token variance, norm, and entropy;
- the availability bit.

A shared evidence trunk followed by expert-specific heads predicts:

\[
\alpha_{e,m}=1+\operatorname{softplus}(a_{e,m}),\qquad
\beta_{e,m}=1+\operatorname{softplus}(b_{e,m}).
\]

It returns expected helpfulness r, uncertainty u, and normalized collaboration
mass:

\[
r=\alpha/(\alpha+\beta),\qquad
u=2/(\alpha+\beta).
\]

Invalid entries are hard-zeroed before every normalization. A residual floor
is applied only to valid entries during warm-up; it never gives a missing
entry non-zero mass.

CollaborativeFusion projects every final expert-modality embedding to 512
dimensions. The evidential score r times one-minus-u times learned
complementarity gives nine weights, normalized globally over valid
contributions. The primary embedding is exactly:

\[
z=\operatorname{BN}\left(\sum_{e,m}\pi_{e,m}P^f_e z_{e,m}\right)
\in\mathbb R^{B\times512}.
\]

For diagnostics, a per-expert embedding is the renormalized weighted sum over
that expert's valid modalities.

Counterfactual supervision removes one of the at most nine projected head
contributions, renormalizes the remaining weights, and recomputes only the
512-dimensional fused head. It never reruns an expert. Correct-class prototype
margin deltas are detached before becoming Beta helpfulness targets.

The reliability result exposes alpha, beta, r, u, contribution weights,
counterfactual targets when training, and valid-count diagnostics. It does not
expose router private layers.

## 10. Innovation III implementation: RDPT

RoleDirectedPeerTeaching is training-only. It receives final expert states,
the shared ReliabilityResult, and labels. For each sample and expert, quality
q is the mask-weighted mean of r times one-minus-u.

Direction gates have shape B,E,E. The diagonal is zero. Teacher quality,
direction comparisons, and all teacher payloads are detached. A gate is active
only when the teacher exceeds both the student by delta and the minimum
teacher-quality threshold. A sample with no valid teacher rejects teaching and
contributes zero peer loss, not a zero-divided NaN.

Each source role uses a compressed payload:

- CNN supplies local-neighbour and part affinity;
- Transformer supplies temperature-scaled ID logits and global-to-patch
  relation;
- Mamba supplies bidirectional sequential-context response.

Small directional adapters map a student's compressed relation to the
teacher's payload space. No full token-by-token affinity matrix is retained.
The result contains named logit-KL, role-relation, private-diversity, rejection
rate, and direction-frequency terms.

Private pooled embeddings are protected with the frozen cosine-hinge
redundancy penalty. Independent branch ID and triplet losses prevent a low-CKA
but nondiscriminative solution. RDPT is absent from inference graphs and has
zero inference FLOPs.

## 11. Model output and loss ownership

TriFusionReID inference with return_aux false returns one finite B,512 tensor.
With return_aux true it returns TriFusionOutput containing:

- fused_embedding;
- branch_embeddings keyed by cnn, transformer, mamba;
- contribution_embeddings;
- ReliabilityResult;
- RelayResult summaries for stages 1 and 2;
- optional PeerTeachingResult;
- optional fused and branch logits;
- mask and finite-value diagnostics.

TriFusionCriterion owns the full named training objective:

| Loss key | Meaning |
|----------|---------|
| id_fused | fused classifier loss |
| triplet_fused | fused batch-hard triplet loss |
| id_cnn, id_transformer, id_mamba | branch classifier losses |
| triplet_cnn, triplet_transformer, triplet_mamba | branch metric losses |
| reliability | counterfactual proper-score and evidence regularizer |
| peer_logits | directional KL |
| peer_role | role-preserving relation transfer |
| private_diversity | cosine-hinge private protection |

The trainer sums a configured dictionary of named losses and logs every term.
It must not infer meaning from tuple position or tuple parity, as the DeMo
trainer does. A missing required key is an error. Optional disabled mechanisms
return an explicit zero scalar on the correct device.

The evaluator calls the model once per image batch with return_aux true and
selects fused or a named branch embedding from that result. It must not run
three complete forwards through return-pattern switches.

## 12. Configuration and ablation mapping

MODEL.ARCH selects DeMo or TriFusion. Existing DeMo defaults remain unchanged.
TriFusion uses its own namespace:

    MODEL.TRIFUSION:
        EXPERTS: [cnn, transformer, mamba]
        TOKEN_GRID: [16, 8]
        CNN_WIDTH: 256
        MAMBA_WIDTH: 256
        RELAY_RANK: 64
        EMBED_DIM: 512
        COLLABORATOR: hfer
        GATE: evidential
        COUNTERFACTUAL: true
        PEER_MODE: rdpt
        PEER_REJECT: true
        ROLE_PAYLOAD: true
        ACTIVATION_CHECKPOINT: true

The minimum matrix maps to flags without source edits:

| ID | Concrete configuration |
|----|------------------------|
| A0 | all experts; collaborator none; fusion uniform mean |
| A1 | all experts; collaborator none; concat-to-512 MLP |
| A2 | HFER; uniform gate; private residual enabled |
| A3 | HFER; same-size softmax gate; no evidential/counterfactual loss |
| A4 | HFER; evidential URGC; counterfactual target enabled |
| A5 | A4 plus symmetric KL; no direction and no reject |
| A6 | A4 plus directional teaching; reject disabled; common logits only |
| A7 | full HFER plus URGC plus rejectable role-preserving RDPT |
| A8 | no collaboration; enlarged concat MLP parameter-matched to A7 |

The required A2 control without private residual is a separate one-flag
diagnostic. The required A4 no-counterfactual control uses the identical
evidential head trained without the counterfactual target.

A8 must actually instantiate and train its extra layers. Its total parameter
count must be within 2 percent of A7 and activated FLOPs within 5 percent where
feasible; otherwise it is labelled only approximately matched and cannot
satisfy the HFER claim gate.

## 13. Capacity and 8 GiB execution budget

The first production implementation has these design budgets:

| Quantity | Gate |
|----------|------|
| total parameters | at most 120 million before full training |
| trainable parameters | reported separately from total |
| real B32/K4 peak allocated memory | at most 7,600 MiB |
| finite-gradient probe | 8 consecutive optimizer steps |
| valid-expert gradient coverage | 100 percent of intended trainable tensors |
| token count | fixed at 128 per valid modality |

The parameter and memory values are gates to measure after implementation, not
claims that the present design already meets them.

Memory controls that preserve the scientific method are:

- automatic mixed precision;
- pack only valid modality slots;
- non-reentrant activation checkpointing around each expert stage;
- discard pre-relay activation references after the synchronous update;
- compute compressed role payloads instead of full affinity matrices;
- compute nine counterfactual heads sequentially;
- return auxiliary tensors only when the caller requests them.

Gradient accumulation must not be described as a real B32 batch if a smaller
batch is used. CPU activation offload is not the default because WSL paging has
already caused unstable evaluation latency. If B32 still fails after the
method-preserving controls, the protocol is amended explicitly before training;
no expert may be removed or silently narrowed to make the run pass.

Efficiency reporting includes total and trainable parameters, activated
FLOPs, peak allocated and reserved memory, images per second, evaluation wall
time, and final embedding width.

## 14. Training and selection protocol

Development proceeds only on the frozen 141-fit/30-dev train-only split:

1. fit on 141 identities;
2. select router thresholds, loss weights, and checkpoint epoch on the 30
   train-only dev identities;
3. freeze the entire configuration and selected epoch;
4. retrain from initialization on all 171 training identities;
5. evaluate the official test set once per frozen seed;
6. report three fixed seeds and aggregate uncertainty.

The running DeMo job evaluates the official test set each epoch and updates
DeMobest.pth by test mAP. That released-style best curve is reported explicitly
as test-selected calibration only. The fair comparison uses the pre-registered
fixed DeMo_50.pth. TriFusion must never use the running job's test curve to
choose an epoch or hyperparameter.

The primary SOTA gate remains both mAP above 84.1 percent and Rank-1 above
87.2 percent under static CLIP-B/16, 256 by 128, no reranking, no test-time
training, three fixed seeds, and an experiment-integrity audit.

## 15. Test surface after seam consent

The accepted interface is the test surface. Tests assert observable behavior:

### TriBranchEncoder

- exact expert keys and modality order;
- all valid modalities reach all three experts;
- invalid slots remain masked after scatter;
- stage and output shapes are stable;
- each complete expert has finite non-zero gradients;
- all-missing rows raise the specified ValueError.

### HeterogeneousRelay

- order-independent synchronous outputs;
- no self message;
- invalid gate mass exactly zero;
- valid incoming gates normalize;
- gamma-zero initialization is an identity within tolerance;
- non-zero gamma changes at least two receiver states;
- private residual energy and gradients remain non-trivial.

### ReliabilityPosterior and CollaborativeFusion

- alpha and beta are strictly greater than one;
- r and u are finite and bounded;
- valid fusion weights sum to one;
- invalid contributions have exactly zero mass;
- detached counterfactual targets have no gradient function;
- controlled degradation can lower the corresponding learned weight;
- single-available-modality inference remains finite.

### RoleDirectedPeerTeaching

- teacher tensors receive no peer-loss gradient;
- direction follows detached quality difference;
- low-confidence samples reject;
- zero-teacher batches return finite zero peer loss;
- CNN, Transformer, and Mamba payload adapters are all exercised.

### TriFusionReID

- stable training and inference results;
- deterministic tiny configuration;
- every non-empty one-, two-, and three-modality mask pattern;
- one real RGBNT201 loader batch;
- one-batch overfit before full training;
- real CUDA Mamba forward/backward.

### Official evaluator

- hand-computed CMC and mAP example;
- same-identity and same-camera exclusion;
- query with no valid positive is treated exactly as the official protocol;
- named fused and branch embeddings require only one model forward.

## 16. Vertical implementation order after consent

Each slice starts with an interface-level failing test and ends with the
smallest production implementation that makes it pass:

1. official evaluator worked-example regression;
2. immutable result types, mask validation, and tiny three-expert encoder;
3. production tokenizer and three standalone expert heads;
4. HFER with NoOp and uniform adapters;
5. URGC with uniform, softmax, and evidential adapters;
6. CollaborativeFusion and detached counterfactual targets;
7. RDPT direction, rejection, and role payloads;
8. full TriFusionReID plus named criterion and trainer;
9. CUDA Mamba integration and real-loader smoke;
10. one-batch overfit, parameter/FLOP report, and eight-step B32 memory probe.

No full experiment starts until the one-batch overfit and capacity gates pass.

## 17. Claim-to-evidence gates

| Proposed contribution | Required evidence before paper promotion |
|-----------------------|-------------------------------------------|
| HFER | A2 beats A1 and matched A8; at least two branch embeddings improve; private diversity and three-expert gradients remain |
| URGC | A4 beats A3 and no-counterfactual control; reliability has useful Brier/ECE, Spearman correlation, corruption monotonicity, and missing-modality robustness |
| RDPT | A7 beats A5/A6; rejection is non-zero and non-saturated; weak branches improve without private CKA collapse; degradation shifts teacher direction |
| SOTA | frozen same-protocol three-seed results exceed both 84.1 mAP and 87.2 Rank-1 and pass integrity audit |

An innovation that fails its own gate is removed or renamed. Full-model gain
alone cannot rescue an unsupported mechanism claim.

## 18. Explicit hazards inherited from the baseline audit

- Never mutate the pinned DeMo clone to make TriFusion appear compatible.
- Never select the final method from an official-test epoch curve.
- Never propagate DeMo's positional tuple loss contract.
- Never use pixel zeros as the only missing-modality signal.
- Never call squeeze without a dimension.
- Never perform three full evaluation forwards when one named auxiliary output
  can supply all embeddings.
- Never let invalid modality slots enter BatchNorm statistics.
- Never hide a resource-enhanced pretrained branch in the primary track.
- Never report a fixed-epoch result as equivalent to a test-selected best.
- Never claim SOTA from one seed, a mismatched protocol, or one metric alone.
