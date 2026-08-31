# TriFusion-ReID implementation blueprint v1.1

Status: independently reviewed implementation-ready design, not an
implementation. This document makes no SOTA claim. The public seams in
TDD_SEAMS.md were accepted by the user's exact reply 接缝同意 at
2026-08-31T11:57:50+08:00. The pre-implementation document hashes and accepted
scope are bound by `evidence/tdd_seam_consent_20260831.json`; implementation
now proceeds as test-driven vertical slices.

This blueprint translates novelty-hardened METHOD_SPEC_V1.md v1.1 into one concrete implementation
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
one joint CIRC reliability posterior, URGC control, optional role-directed peer
teaching, fusion, and named auxiliary results. An offline builder owns
identity-disjoint full-intervention target generation; it never enters the
inference graph. Deleting the module would force the trainer, evaluator,
missing-modality logic, and ablations to reimplement those rules; therefore
the interface earns its depth.

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
        intervention_targets.py
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
        build_circ_targets.py

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
| sample_keys | optional sequence length B | immutable training keys used only to look up CIRC targets |

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
    predict one joint CIRC posterior from all pre-relay stage-1 evidence
    apply synchronous HFER relay 1
    run stage 2 of all three experts
    apply synchronous HFER relay 2 using the same posterior
    run stage 3 of all three experts
    form branch heads and contribution embeddings
    fuse with the same posterior
    look up immutable CIRC targets only in the criterion during training
    compute RDPT only when explicitly enabled, targets are present, and training is true

The posterior is predicted once because the claim says the same
intervention-calibrated belief controls relay bandwidth, final fusion, and
missing/degraded suppression. Its target is never produced inside this
forward. A frozen out-of-fold HFER-uniform generator produces complete-network
intervention targets offline, so the stage-1 router receives downstream
identity-margin supervision without a self-referential target graph.

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

## 9. Innovation II/III implementation: CIRC and URGC

ReliabilityPosterior consumes all three pre-relay stage-1 states and the bool
mask. For every expert-modality contribution it forms an entry token from:

- masked global token statistics;
- cross-modal agreement inside the expert;
- cross-expert agreement for the same modality;
- signal quality statistics such as token variance, norm, and entropy;
- learned expert/modality identifiers and the availability bit.

The nine entry tokens pass through one small masked set mixer, so every
prediction can observe the whole expert-modality context. One shared output
function is applied to all mixed entry tokens; there is no expert-specific
ModuleList of uncertainty heads. It predicts an evidence-allocation logit and
evidence mass:

\[
\mu_{e,m}=\sigma(\ell_{e,m}/\tau_{\rm shared}),\qquad
\kappa_{e,m}=\kappa_{\min}+\operatorname{softplus}(c_{e,m}),
\]

\[
\alpha_{e,m}=1+\mu_{e,m}\kappa_{e,m},\qquad
\beta_{e,m}=1+(1-\mu_{e,m})\kappa_{e,m}.
\]

The common learned temperature and concentration prior are shared across all
nine entries. The result exposes \(r=(1+\mu\kappa)/(2+\kappa)\), so \(\mu\) is
not mislabeled as the posterior mean:

\[
r=\alpha/(\alpha+\beta),\qquad u=2/(\alpha+\beta).
\]

Invalid entries are hard-zeroed before every normalization. A residual floor
is applied only to valid entries during warm-up; it never gives a missing
entry non-zero mass.

### 9.1 Offline CIRC target builder

tools/build_circ_targets.py is a training-only, deterministic builder:

1. assign training identities to three hash-registered folds;
2. fit one HFER-uniform target generator on each two-fold complement;
3. freeze the generator, embed an unperturbed held-out reference bank, and
   build cross-camera leave-one-query-out identity prototypes without its
   classifier head;
4. score each held-out query against the same fixed reference prototypes,
   requiring a different-camera positive for every primary row;
5. for every expert-modality execute total, direct-only and relay-only
   full-network removals; then execute exactly one deterministically sampled
   valid single-edge removal per relay stage and query-condition row, one
   intervention at a time to stay inside 8 GiB;
6. store helpful/neutral/harmful labels and signed deltas separately for every
   corruption family, severity and seed; never pool heterogeneous conditions
   into one iid count;
7. run a pre-registered sampled query+gallery symmetric audit that rebuilds
   the intervened reference prototypes;
8. write immutable targets.jsonl and receipt.json keyed by sample_keys and
   condition, with fold, mask, fixed-bank hash, generator checkpoint hash,
   intervention seeds, decomposed effects and SHA-256.

The builder asserts zero identity overlap between a target row and its
generator training fold and never emits a gradient-bearing tensor. During
development, target generation uses only the 141 fit identities and never the
30 dev identities. After configuration freeze, a separately receipted final
mode may include the former 30 dev identities as ordinary training-only rows
in the all-171 folds, with no subsequent selection. Official-test identities
are never read. The receipt separately counts cross-camera, same-camera-only
and invalid-support rows. Same-camera-only and no-positive rows never enter the
primary CIRC loss or calibration tables.

The total intervention zeros the source's outgoing messages at relay stages 1
and 2 and its final fusion contribution. Direct-only keeps relays active;
relay-only keeps final fusion active. Edge removals are audit-only: for each
stage, the builder lexicographically orders the valid
source/target/modality edges and selects exactly one using SHA-256 over the
fixed salt TriFusion-CIRC-edge-v1, protocol hash, sample key, condition and
stage, interpreting the digest as an unsigned big-endian integer. This adds two
edge reruns per primary row rather than all 36. Every
query-side intervention uses the unchanged full reference bank. The symmetric
audit applies the same intervention to query and sampled reference embeddings
and rebuilds prototypes. No path is implemented as pixel zeroing or a
512-dimensional head-only shortcut. The interaction residual
total-minus-direct-minus-relay is reported, not assumed additive.

Before the builder runs, protocols/circ_target_v1.json freezes and hashes the
fold rule, epsilon, corruption families/severities/seeds, encoder config,
target-cache schema/code commit, calibration metrics, symmetric-audit sample,
edge salt/order/validity/two-per-row budget, final-mode policy, and agreement
thresholds. This happens before dev labels are used.

Cheap head-level leave-one-contribution-out values remain optional diagnostics.
They can become a dense auxiliary loss only after a frozen dev audit shows
useful signed agreement and rank correlation with full-network interventions.

The primary loss consumes each condition's binary helpfulness outcome with BCE
and Brier. It is explicitly helpful versus not-helpful: neutral and harmful are
both negative labels. Their signed effects are logged separately as audits, and
low \(r\) is never interpreted as harmfulness.
Condition-wise beta-binomial/logistic-normal fits are audits for overdispersion
and concentration coverage only. Inferential effective sample size is based on
identity/query clusters, never raw corruption-seed count.

### 9.2 Online URGC control

CollaborativeFusion projects every final expert-modality embedding to 512
dimensions. The default promoted CIRC score is \(r\) times learned
complementarity, giving nine weights normalized globally over valid
contributions. Multiplication by one-minus-\(u\) is a pre-registered
U2-evidence ablation and may replace the default only if condition-wise
empirical concentration coverage passes before final configuration freeze.
Otherwise \(u\) is diagnostic. The primary embedding is exactly:

\[
z=\operatorname{BN}\left(\sum_{e,m}\pi_{e,m}P^f_e z_{e,m}\right)
\in\mathbb R^{B\times512}.
\]

For diagnostics, a per-expert embedding is the renormalized weighted sum over
that expert's valid modalities. The same ReliabilityResult instance supplies
both relay gates and fusion weights. Separate-router, relay-only and
fusion-only adapters exist only as ablation configurations.

The reliability result exposes alpha, beta, r, u, contribution weights,
valid-count diagnostics and a target-cache provenance key when training. It
does not expose router private layers or construct targets.

After U2 training, a frozen dev-only target-transfer audit repeats the
decomposed interventions with the deployed model. Weak proxy-target/router/
deployed-effect agreement invalidates the causal-reliability claim.

## 10. Auxiliary implementation: RDPT

RoleDirectedPeerTeaching is training-only and disabled in the promoted v1.1
core configuration. It receives final expert states,
the shared ReliabilityResult, and labels. For each sample and expert, quality
q is the mask-weighted mean of the collaboration score active in that
configuration (mean-only \(r\) by default).

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
| reliability | per-condition out-of-fold intervention BCE/Brier and evidence regularizer |
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
        GATE: circ
        CIRC_TARGET_MODE: full_oof_intervention
        CIRC_PROTOCOL: protocols/circ_target_v1.json
        CIRC_FOLDS: 3
        CIRC_REQUIRE_CROSS_CAMERA: true
        CIRC_INTERVENTIONS: [total, direct, relay]
        CIRC_EDGE_AUDIT: true
        CIRC_EDGE_PER_STAGE_PER_ROW: 1
        CIRC_EDGE_SALT: TriFusion-CIRC-edge-v1
        CIRC_POOL_CONDITIONS: false
        CIRC_SYMMETRY_AUDIT: true
        CIRC_USE_UNCERTAINTY_MULTIPLIER: false
        CIRC_TARGETS_REQUIRED: true
        CIRC_TARGET_TABLE: null
        PEER_MODE: none
        PEER_REJECT: true
        ROLE_PAYLOAD: true
        ACTIVATION_CHECKPOINT: true

The minimum matrix maps to flags without source edits:

| ID | Concrete configuration |
|----|------------------------|
| A0 | all experts; collaborator none; fusion uniform mean |
| A1 | all experts; collaborator none; concat-to-512 MLP |
| A2 | HFER; uniform gate; private residual enabled |
| A3 | A2 with private residual or role mixer removed |
| A4 | no collaboration; enlarged concat MLP parameter-matched to U2 |
| R0 | HFER; same-size softmax gate |
| R1 | joint Beta router; observational end-to-end supervision |
| R2 | joint Beta router; cheap head-only LOO supervision |
| R3 | CIRC; frozen identity-out-of-fold full-intervention targets |
| R4 | nine independent evidential heads |
| R5 | global non-evidential router matched to R3 |
| U0 | R3 posterior used only at final fusion |
| U1 | R3 posterior used only at HFER relays |
| U2 | one R3 posterior reused at relays and fusion; promoted core |
| U3 | separate parameter-matched relay and fusion routers |
| C0 | shuffled targets, expert permutation and capacity-floor controls |
| K0 | U2 with no peer teaching |
| K1 | U2 plus symmetric KL |
| K2 | U2 plus directional teaching without rejection |
| K3 | U2 plus full RDPT auxiliary |
| K4 | K3 wrong-payload and adaptive-transfer controls |

R1–R5 use the identical router parameter budget wherever possible. C0 must
include a learned context-free path with matched capacity, not only a constant
mean. K3 is not part of the promoted core unless it passes its separate
promotion gate.

A4 must actually instantiate and train its extra layers. Its total parameter
count must be within 2 percent of U2 and activated FLOPs within 5 percent where
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
- generate offline full-network interventions sequentially and cache only
  scalar target statistics;
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

1. before dev-label use, freeze and hash circ_target_v1.json, including folds,
   epsilon, corruption conditions, target-generator recipe/code commit,
   cache schema, metrics and audit thresholds;
2. split the 141 fit identities into three deterministic folds, train each
   two-fold target generator, and create held-out intervention targets;
3. fit CIRC/URGC on 141 identities using only immutable out-of-fold targets;
4. run query/gallery symmetry and proxy-to-deployed target-transfer audits,
   then select router thresholds, loss weights, and checkpoint epoch on the 30
   train-only dev identities;
5. freeze the entire configuration and selected epoch;
6. reclassify the former 30 dev identities as training-only, repeat the
   deterministic three-fold target generation on all 171 identities, then
   retrain from initialization with no further selection or early stopping;
7. evaluate the official test set once per frozen seed;
8. report three fixed seeds and aggregate uncertainty.

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
- one shared joint output function emits all nine entries;
- valid fusion weights sum to one;
- invalid contributions have exactly zero mass;
- target-cache tensors have no gradient function and match sample keys;
- every target row's identity is absent from its generator training fold;
- every primary row has different-camera positive support;
- total, direct and relay fields are present for all nine contributions;
- exactly one valid edge per stage is selected by the frozen hash rule for each
  query-condition row, the receipt proves two-per-row cost and group coverage,
  and sampled edge values are never consumed as primary training targets;
- helpful, neutral and harmful signed fields are present, while the learned
  target is explicitly helpful versus not-helpful;
- corruption family/severity/seed rows remain separate;
- per-condition BCE/Brier/ECE, overdispersion, empirical concentration
  coverage and identity/query-cluster effective sample size are reproducible;
- per-camera and identity-frequency calibration group assignments are
  deterministic and complete;
- deterministic intervention seeds reproduce target rows and hashes;
- receipt binds the frozen protocol, code, generator and reference-bank hashes;
- query-only and symmetric audit fixtures use their registered reference-bank
  semantics;
- proxy-target-to-deployed-model transfer emits a frozen agreement receipt;
- shuffled/permuted target controls remain constructible at the same seam;
- controlled degradation can lower the corresponding learned weight;
- single-available-modality inference remains finite.

### RoleDirectedPeerTeaching

- disabled mode produces no peer result or loss in the promoted core;
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
5. joint ReliabilityPosterior with uniform, softmax, observational, and CIRC
   adapters;
6. deterministic fold registry, frozen target-cache schema, and total/direct/
   relay/edge intervention builder paths;
7. CollaborativeFusion and immutable target lookup in the criterion;
8. same-posterior URGC relay/fusion adapters and separate-router controls;
9. optional RDPT direction, rejection, and role payloads;
10. full TriFusionReID plus named criterion and trainer;
11. CUDA Mamba integration and real-loader smoke;
12. one-batch overfit, parameter/FLOP report, and eight-step B32 memory probe.

No full experiment starts until the one-batch overfit and capacity gates pass.

## 17. Claim-to-evidence gates

| Proposed contribution | Required evidence before paper promotion |
|-----------------------|-------------------------------------------|
| HFER | A2 beats A1 and matched A4; at least two branch embeddings improve; private diversity and three-expert gradients remain |
| CIRC | R3 beats R0/R1/R2/R4/R5; decomposed full interventions, helpful/harmful audit, grouped BCE/Brier/ECE, overdispersion/coverage, query-gallery symmetry, proxy transfer, negative controls and corruption monotonicity pass |
| URGC | U2 beats U0/U1/U3 and C0; fused plus at least two branches improve; missing/degraded mean and worst-case robustness pass |
| RDPT auxiliary | K3 beats K0/K1/K2/K4; rejection is non-zero and non-saturated; weak branches improve without private CKA collapse; otherwise do not promote |
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
