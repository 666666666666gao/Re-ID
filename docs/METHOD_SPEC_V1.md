# TriFusion-ReID method specification v1

Status: architecture and claim contract frozen for the first implementation. No
SOTA claim is implied by this document; every claim below has an explicit
experiment gate.

## 1. Problem and resource boundary

For each sample, the input is an aligned RGB–NIR–TIR tuple

\[
\mathcal X=\{x_m\}_{m\in\{R,N,T\}},\qquad
x_m\in\mathbb R^{B\times3\times256\times128},
\]

with an availability mask \(a\in\{0,1\}^{B\times3}\). Missing inputs are never
represented only by zero pixels: the Boolean mask is propagated through every
router, relay and normalization operation.

The fair primary track uses the same OpenAI CLIP ViT-B/16 image pretraining as
the DeMo/MDReID family. CNN and Mamba expert bodies do not silently introduce a
second supervised pretraining dataset. Any future fully pretrained variant is
reported in a separate resource-enhanced track.

Development thresholds and router hyperparameters use a deterministic,
train-only cross-camera validation protocol. A data audit found that 20 of the
30 identities in the provided `train_171 - train_141` difference are
single-camera; under the official same-identity/same-camera exclusion they have
no valid retrieval positive and therefore cannot serve as the primary dev
retrieval set.

Protocol v1 selects only from the 51 multi-camera identities in `train_171`.
Identities are ordered by
`SHA256("TriFusion-RGBNT201-dev-v1:" + identity)`; the first 30 are held out and
the remaining 141 are used for fitting. Dev query and gallery both contain all
held-out samples, exactly as in the official RGBNT201 loader, and the official
same-identity/same-camera filter is applied. This rule depends only on training
metadata and a pre-registered salt, not test labels or model scores. Once the
configuration is frozen, the final model is retrained on all 171 training
identities. Test labels are never used for routing thresholds, early stopping
or corruption strengths. The generated identity lists and audit live under
`protocols/` and `artifacts/rgbnt201_dev_protocol_v1_audit.json`.

## 2. Full heterogeneous experts

The three branches are architecture experts, not modality-specific branches.
Every expert processes every available modality with tied within-expert
weights:

\[
h^0_{e,m}=E_e(x_m)+p_e+q_m,
\quad e\in\{C,V,S\},
\]

where \(C\), \(V\), and \(S\) denote the CNN, Vision Transformer and selective
state-space (Mamba) experts; \(p_e\) is positional encoding and \(q_m\) is a
modality embedding. The modality dimension is kept explicit as
`[batch, modality, token, channel]` until final fusion.

The production profile uses a 16×8 token grid and three interaction depths:

- CNN expert: same-resolution residual depthwise-convolution stages with
  3×3/5×5/dilated receptive fields, preserving local texture and boundaries.
- Transformer expert: CLIP ViT-B/16 blocks grouped at depths 4/8/12, preserving
  global content-dependent relations.
- Mamba expert: bidirectional row-major and column-major selective scans,
  grouped at depths 3/6/9, preserving long-range context with linear token
  complexity.

Each branch has its own BN neck, classifier and metric embedding. Therefore
`CNN-only`, `Transformer-only`, and `Mamba-only` are executable models, and the
post-collaboration embedding of every branch can be evaluated independently.

## 3. Innovation I — HFER

### Heterogeneous Full-Expert Relay

HFER performs synchronous, bidirectional exchange after each of the first two
expert stages. Features exchanged at a relay are computed from the pre-relay
states, so the update has a well-defined acyclic graph.

Each expert state is split into a low-rank shared relay and a private residual:

\[
s^k_{e,m}=P^{k,s}_e\operatorname{LN}(h^k_{e,m}),\qquad
p^k_{e,m}=h^k_{e,m}-P^{k,r}_e s^k_{e,m}.
\]

For target expert \(e\), the message from source \(j\ne e\) is

\[
\mu^k_{j\rightarrow e,m}
=P^{k,o}_{j\rightarrow e}\,\phi^k_j
\left(s^k_{j,m},\bar s^k_j\right),
\]

where \(\bar s^k_j\) is the masked spectral consensus of expert \(j\), and
\(\phi_j\) preserves the source role: local convolution for CNN, low-rank
attention for Transformer, and bidirectional selective mixing for Mamba.

The target update is

\[
\tilde h^k_{e,m}=p^k_{e,m}+P^{k,u}_e
\left(s^k_{e,m}+\gamma^k_e
\sum_{j\ne e}g^k_{j\rightarrow e,m}\mu^k_{j\rightarrow e,m}\right).
\]

`HFER-uniform` fixes valid \(g\) to uniform weights and is the isolated
Innovation-I model. URGC supplies the learned gates only in later ablations.
The updated states enter the next native expert stage; consequently, a relay
can improve the receiving branch itself rather than only its final ensemble.

#### HFER claim gate

HFER remains a main contribution only if it:

1. improves fused retrieval over no-exchange concat/mean fusion;
2. improves at least two receiving branch embeddings after interaction;
3. beats a parameter-matched late-fusion MLP;
4. retains non-trivial private-feature diversity and non-zero gradients in all
   three experts.

If only the fused head improves, the claim is reduced to feature fusion rather
than deep expert collaboration.

## 4. Innovation II — URGC

### Unified Reliability-Guided Collaboration

URGC predicts one evidential posterior for every sample, expert and modality.
It is trained to estimate whether that contribution improves the correct-ID
margin, not merely to reproduce an unconstrained attention weight.

From global evidence, cross-modal agreement, cross-expert agreement and learned
quality statistics, the router predicts

\[
(\alpha_{e,m},\beta_{e,m})=
1+\operatorname{softplus}(Q_\theta(\xi_{e,m})),
\]

\[
r_{e,m}=\frac{\alpha_{e,m}}{\alpha_{e,m}+\beta_{e,m}},\qquad
u_{e,m}=\frac{2}{\alpha_{e,m}+\beta_{e,m}}.
\]

Here \(r\) is expected helpfulness and \(u\) is lack of evidence. Invalid
modalities are hard-masked before every normalization.

### Counterfactual supervision

Training contributions \(v_{e,m}\) are fused once, after which cheap head-level
leave-one-contribution-out embeddings are formed without rerunning a backbone.
For the correct class \(y\), define the prototype margin

\[
M(z,y)=\cos(z,w_y)-\max_{c\ne y}\cos(z,w_c).
\]

The detached counterfactual target is

\[
t_{e,m}=\operatorname{stopgrad}\left[
\sigma\left(\frac{M(z_{\rm full},y)-M(z_{-(e,m)},y)}{\tau_{cf}}\right)
\right].
\]

The reliability loss combines a proper scoring rule and evidence regularizer:

\[
\mathcal L_{rel}=\frac1{|\Omega|}\sum_{(e,m)\in\Omega}
(r_{e,m}-t_{e,m})^2+\lambda_u\mathcal R_{evidence}.
\]

The same \((r,u)\) posterior controls all three collaboration sites:

1. relay bandwidth \(g^k_{j\rightarrow e,m}\);
2. final expert–modality fusion weight \(\pi_{e,m}\);
3. suppression of missing or degraded contributions.

For example,

\[
g^k_{j\rightarrow e,m}\propto
a_m\,r_{j,m}(1-u_{j,m})\,c^k_{j\rightarrow e,m},
\]

where \(c\) is learned complementarity. A small residual floor prevents
premature expert death, while missing entries receive exactly zero weight.

#### URGC claim gate

URGC remains a main contribution only if:

1. it improves over a same-size softmax gate and an uncalibrated scalar gate;
2. predicted reliability correlates with held-out counterfactual margin gain;
3. Brier score/ECE and routing weights respond monotonically to controlled
   blur, occlusion, exposure and noise;
4. it improves both mean and worst-case missing-modality retrieval without a
   material complete-modality regression.

If retrieval improves but calibration/counterfactual correlation does not,
the method is reported as dynamic gating rather than reliability estimation.

## 5. Innovation III — RDPT

### Rejectable Directional Peer Teaching

RDPT prevents conventional symmetric mutual distillation from collapsing the
three inductive biases. For a potential teacher \(j\) and student \(e\), the
detached direction gate is

\[
d_{j\rightarrow e}=\operatorname{stopgrad}\left[
\sigma\left(\frac{q_j-q_e-\delta}{\tau_d}\right)
\mathbf 1(q_j>q_{min})\right],
\]

where \(q_e\) is the availability-weighted URGC reliability. If no teacher
passes the confidence and gap conditions, that sample rejects peer teaching.
The teacher tensors are always stop-gradient.

The payload is role preserving:

- CNN teaches local neighbourhood/part affinity;
- Transformer teaches global identity distribution and global token relation;
- Mamba teaches bidirectional sequential-context response.

With adapters \(A_{j\rightarrow e}\), the loss is

\[
\mathcal L_{peer}=\frac{
\sum_{j\ne e}d_{j\rightarrow e}
\left[\tau^2\operatorname{KL}(p_j^\tau\Vert p_e^\tau)
+\lambda_{role}\,D(A_{j\rightarrow e}(\rho_e),
\operatorname{stopgrad}(\rho_j))\right]
}{\epsilon+\sum_{j\ne e}d_{j\rightarrow e}}.
\]

Only shared/relational payloads are taught. Private embeddings are protected by
a hinge redundancy penalty

\[
\mathcal L_{private}=\sum_{e<j}
\max(0,\cos(p_e,p_j)-\rho_{max})^2,
\]

while independent ID and triplet losses keep every private subspace
discriminative.

#### RDPT claim gate

RDPT remains a main contribution only if it:

1. improves over symmetric KL and always-on one-way distillation;
2. produces a non-zero but non-saturated rejection rate;
3. improves weak branches without increasing private-feature CKA to collapse;
4. gives the expected teacher-direction shifts under controlled modality
   degradation.

If directional/rejectable behavior is not observed, RDPT is removed rather
than retained on the strength of the full model alone.

## 6. Fusion and training objective

The fused embedding is

\[
z=\operatorname{BN}\left(\sum_{e,m}\pi_{e,m}
P^f_e z_{e,m}\right),\qquad
\sum_{e,m}\pi_{e,m}=1,
\]

with a residual uniform-consensus path during warm-up. The full loss is

\[
\mathcal L=\mathcal L_{id}^{f}+\lambda_{tri}\mathcal L_{tri}^{f}
+\lambda_b\sum_e(\mathcal L_{id}^{e}+\mathcal L_{tri}^{e})
+\lambda_{rel}\mathcal L_{rel}
+\lambda_{peer}\mathcal L_{peer}
+\lambda_p\mathcal L_{private}.
\]

No loss uses test labels. Reliability targets and teacher directions are
computed only for training samples. At inference, RDPT adds no operation and
URGC uses only its learned evidence features.

## 7. Public software contracts

The implementation must expose these stable seams:

```text
TriBranchEncoder(images, modality_mask) -> ExpertStateMap
HeterogeneousRelay(states, reliability, stage) -> RelayResult
ReliabilityPosterior(states, modality_mask) -> ReliabilityResult
RoleDirectedPeerTeaching(states, reliability, labels) -> PeerTeachingResult
CollaborativeFusion(states, reliability, modality_mask) -> FusionResult
TriFusionReID.forward(batch, targets=None, return_aux=False) -> Tensor | TriFusionOutput
```

Required tensor invariants:

- expert keys are exactly `cnn`, `transformer`, `mamba`;
- modality order is exactly `RGB`, `NI`, `TI`;
- all normalized gates sum to one over valid contributors and are zero for
  invalid contributors;
- every relay is synchronous and excludes self-messages;
- inference embeddings are finite and L2-normalizable for every non-empty
  modality mask;
- an all-missing sample is rejected with a clear exception, not silently
  converted to a zero embedding.

## 8. Minimum ablation matrix

| ID | Variant | Mechanism isolated | Necessary comparison |
|----|---------|--------------------|----------------------|
| A0 | Three independent experts + mean | extra capacity only | single experts |
| A1 | Three independent experts + concat/MLP | learned late fusion | matched-width control |
| A2 | HFER-uniform | staged relay | A1; no-private-residual relay |
| A3 | HFER + soft gate | generic routing | A2 |
| A4 | HFER + URGC | calibrated counterfactual routing | A3; no-counterfactual target |
| A5 | A4 + symmetric KL | ordinary mutual teaching | A4 |
| A6 | A4 + directional, no reject | direction only | A5 |
| A7 | Full RDPT | direction + rejection + role payload | A6 |
| A8 | Larger concat control | no collaboration, no fewer parameters | A7 |

Every promoted variant reports fused and per-branch mAP/Rank-1, parameter count,
activated FLOPs, peak memory, gradient utilization and three-expert routing
entropy. Reliability variants additionally report Brier/ECE and Spearman
correlation with counterfactual margin gain; peer variants report rejection
rate and private/shared CKA.

## 9. Claim and SOTA decision

The primary fair target is RGBNT201, static CLIP-B/16, 256×128, no reranking,
no test-time training. A SOTA statement requires the frozen model to exceed
both 84.1% mAP and 87.2% Rank-1 under the same protocol, with three fixed seeds
and an experiment-integrity audit. DINOv3, multi-stage external distillation and
test-time-training results remain separate tracks. Failure to cross the fair
target does not invalidate useful ablation findings, but it forbids a SOTA
claim.
