# TriFusion-ReID method specification v1.1

Status: novelty-hardened architecture and claim contract with final independent
implementation-readiness PASS. The six public software seams remain frozen,
but v1.1 replaces
the self-referential head-only reliability target and demotes RDPT from a main
contribution after the 2026-08-31 novelty audit. No SOTA claim is implied by
this document; every claim below has an explicit experiment gate.

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
identities. At that point the former 30 development identities may rejoin only
as ordinary training identities: their labels may be used to build immutable
out-of-fold targets and optimize the final model, but no metric, threshold,
epoch, loss weight or architecture is selected again. Official-test labels are
never used for routing thresholds, early stopping or corruption strengths. The
generated identity lists and audit live under
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

## 4. Innovation II — CIRC

### Cross-fitted Interventional Reliability Calibration

CIRC trains one global router to predict a jointly comparable evidential
posterior for every sample, expert and modality. The final joint head emits all
nine \((\alpha,\beta)\) pairs together; it is not a collection of independently
trained expert-local uncertainty heads. Every entry uses the same target
definition, proper scoring rule, temperature family and concentration prior.
This design addresses, but does not by itself solve, the evidence-scale
identifiability problem exposed by TMUR.

From global evidence, cross-modal agreement, cross-expert agreement and learned
quality statistics, one masked set mixer and one shared output function produce
an evidence-allocation logit \(\ell_{e,m}\) and evidence mass \(c_{e,m}\):

\[
\mu_{e,m}=\sigma(\ell_{e,m}/\tau_{\rm shared}),\qquad
\kappa_{e,m}=\kappa_{\min}+\operatorname{softplus}(c_{e,m}),
\]

\[
\alpha_{e,m}=1+\mu_{e,m}\kappa_{e,m},\qquad
\beta_{e,m}=1+(1-\mu_{e,m})\kappa_{e,m},
\]

\[
r_{e,m}=\frac{\alpha_{e,m}}{\alpha_{e,m}+\beta_{e,m}},\qquad
u_{e,m}=\frac{2}{\alpha_{e,m}+\beta_{e,m}}.
\]

Here \(r\) is expected helpfulness and \(u\) is lack of evidence. Invalid
modalities are hard-masked before every normalization. The same
\(\tau_{\rm shared}\), \(\kappa_{\min}\), set mixer and output function are used
for all nine entries. Because \(r=(1+\mu\kappa)/(2+\kappa)\), \(\mu\) is an
evidence allocation rather than being mislabeled as the posterior mean.

### Identity-disjoint intervention supervision

The primary targets come from a frozen HFER-uniform target generator, not from
the router or fusion head being optimized. Training identities are assigned to
three deterministic folds. For fold \(f\), target generator
\(F_{\bar f}\) is fit on the other two folds and produces targets only for
identities in \(f\). Thus no identity is scored by a target generator trained
on that identity. The final all-171 training run regenerates the same
three-fold table only after the complete development configuration is frozen.
The former 30 dev identities then become training-only rows with no further
selection or early stopping; official-test samples are never used.

The held-out generator has no classifier weight for an unseen identity.
Therefore target construction never uses its classification head. It first
embeds an unperturbed held-out reference bank. Primary CIRC rows require at
least one same-identity, different-camera reference. For target row \(i\),
define the cross-camera positive prototype

\[
p_{y_i,{\rm xcam}}^{(-i)}=\operatorname{norm}\left(
\sum_{\substack{j\in\mathcal H_f\\y_j=y_i,\ {\rm cam}_j\ne{\rm cam}_i}}
F_{\bar f}(\mathcal X_j)\right),
\]

and unperturbed negative identity prototypes \(p_c\) for \(c\ne y_i\). The
query-side retrieval margin is

\[
M_i^Q(z)=\cos(z,p_{y_i,{\rm xcam}}^{(-i)})
-\max_{c\ne y_i}\cos(z,p_c).
\]

The official same-identity/same-camera exclusion is mandatory for the primary
target and calibration tables. Same-camera-only and no-positive rows are
excluded from the primary CIRC loss; an appendix-only auxiliary may use them
under a separately named, default-zero loss.

Every corruption family \(g\), severity \(\ell\), and seed \(s\) is a separate
registered condition. The target cache stores three primary interventions for
all nine expert-modality contributions:

1. total \(T\): suppress source \((e,m)\)'s outgoing messages at both HFER
   relays and its final fusion contribution;
2. direct \(D\): suppress only its final fusion contribution;
3. relay \(R\): suppress only its outgoing messages at both relays.

Single-edge \(E\) removal is an audit-only sampled intervention, not a training
target. For each valid query-condition row and each relay stage \(k\), exactly
one valid directed edge \((k,j\!\rightarrow\!e,m)\) is chosen by indexing the
lexicographically ordered valid-edge list with

\[
\operatorname{int}\left[
\operatorname{SHA256}(
{\tt TriFusion\mbox{-}CIRC\mbox{-}edge\mbox{-}v1}
\Vert{\tt protocol\_hash}\Vert{\tt sample\_key}
\Vert g\Vert\ell\Vert s\Vert k)\right]\bmod |\mathcal E_{i,k}|.
\]

Thus each primary row incurs two, not 36, edge-audit reruns. The salt, edge
ordering, validity rule and two-per-row budget are bound in the frozen protocol;
the hexadecimal digest is interpreted as an unsigned big-endian integer, and
the receipt reports coverage by stage, source, target, modality, camera and
condition.

Each intervention performs a complete query forward against the unchanged
reference bank. For \(a\in\{T,D,R\}\),

\[
\Delta^{g,\ell,s,a}_{i,e,m}
=M_i^Q(F_{\bar f}(\mathcal X_i^{g,\ell,s}))
-M_i^Q(F_{\bar f}(\mathcal X_i^{g,\ell,s};
\operatorname{do}_a(c_{e,m}=0))).
\]

The same margin difference is stored for each deterministically selected edge,
keyed by its full edge identity. No implementation may substitute pixel zeroing
or head-only removal. The total effect is the primary router target. Direct and
relay effects justify where the score is used; sampled edge effects are a
directional audit only. The non-additive interaction residual
\(\Delta^T-\Delta^D-\Delta^R\) is reported rather than forced to zero.

For every condition, store three outcomes:

\[
y^{+,g,\ell,s}_{i,e,m}=\mathbf 1(
\Delta^{g,\ell,s,T}_{i,e,m}>\epsilon_{cf}),\qquad
y^{-,g,\ell,s}_{i,e,m}=\mathbf 1(
\Delta^{g,\ell,s,T}_{i,e,m}<-\epsilon_{cf}),\qquad
y^{0,g,\ell,s}_{i,e,m}=1-y^{+,g,\ell,s}_{i,e,m}
-y^{-,g,\ell,s}_{i,e,m}.
\]

The router's expected helpfulness is trained per condition, not from one
pooled clean/blur/occlusion/missing count:

\[
\mathcal L_{rel}=\frac1{|\Omega|}\sum_{(i,e,m,g,\ell,s)\in\Omega}
\operatorname{BCE}(r_{i,e,m}^{g,\ell,s},y^{+,g,\ell,s}_{i,e,m})
+\lambda_b\operatorname{Brier}(r_{i,e,m}^{g,\ell,s},
y^{+,g,\ell,s}_{i,e,m})
+\lambda_u\mathcal R_{evidence}.
\]

The learned scalar is explicitly helpful-versus-not-helpful: both \(y^0\) and
\(y^-\) are negative labels for \(r\). Neutral and harmful signed effects are
reported separately as audits; low \(r\) is not a harmfulness estimate and no
three-state discrimination claim is made. Condition-wise beta-binomial or
logistic-normal models are calibration audits only. They group exchangeable
seeds within one corruption family and severity, report overdispersion, and
use identity/query clusters—not the raw seed count—as the inferential effective
sample size. Clean, blur, occlusion, missing and sensor-noise outcomes are
never pooled as iid trials. Posterior concentration is not called calibrated
unless condition-wise empirical coverage supports it.

The immutable target table records fold identity, frozen reference-bank hash,
generator checkpoint, condition, intervention seed, availability mask and
SHA-256. Before any target generation or dev-label use, the folds,
\(\epsilon_{cf}\), corruption suite, target-encoder configuration, cache
schema/code commit, calibration metrics, edge salt/order/validity/two-per-row
budget, development-versus-post-freeze-final mode contract and query/gallery
audit thresholds are written to a hashed protocol file. The fusion head and
router never update the target within the same optimization graph.

Cheap head-level leave-one-contribution-out margins may be computed as a dense
auxiliary only after they agree with full reruns on frozen data. They are never
the sole evidence for a causal reliability claim.

The primary target is query-side utility against a fixed full reference bank.
A pre-registered symmetric audit additionally applies the same intervention to
query and sampled gallery/reference embeddings, rebuilds prototypes, and
measures \(\Delta^{QG}\). If query-only and symmetric effects fail the frozen
agreement gate, all claims are explicitly limited to query-side contribution.

Finally, after U2 is trained, a target-transfer audit recomputes full
interventions with the frozen deployed model on train-only dev data. Weak
agreement between the HFER-uniform proxy target, router prediction and deployed
effect invalidates the causal-reliability claim.

#### CIRC claim gate

CIRC remains a main contribution only if:

1. the out-of-fold full-intervention target beats an otherwise identical
   end-to-end observational or head-only target;
2. per-condition posterior mean has useful BCE/Brier/ECE overall and within
   every expert, modality, expert-modality, camera and identity-frequency
   group; concentration and effective sample size are claimed only after
   coverage/overdispersion audits;
3. total, direct and relay effects are separately measured; the total score
   predicts helpful versus not-helpful utility, while neutral/harmful signed
   strata and the deterministically sampled edge effects remain audits rather
   than a claimed three-state or all-edge predictor;
4. cheap head-only effects, query/gallery symmetry and proxy-to-deployed target
   transfer pass their pre-registered agreement gates;
5. shuffled targets, permuted expert identities, router-temperature
   rescaling, identity/camera leakage probes and capacity-floor controls do not
   reproduce the gain;
6. held-out reliability changes monotonically under separately reported blur,
   occlusion, exposure, NIR noise and thermal noise interventions; every
   primary target/calibration row has cross-camera positive support.

If these gates fail, the component is described only as a global dynamic gate,
not as causal or calibrated reliability.

## 5. Innovation III — URGC

### Unified Reliability-Guided Collaboration

URGC is the control policy that reuses the single CIRC posterior. It is not a
claim that evidential gating, dynamic fusion or missing-modality masking is new.
The same posterior mean \(r\) controls all three collaboration sites:

1. relay bandwidth \(g^k_{j\rightarrow e,m}\);
2. final expert–modality fusion weight \(\pi_{e,m}\);
3. suppression of missing or degraded contributions.

For example,

\[
g^k_{j\rightarrow e,m}\propto
a_m\,r_{j,m}\,c^k_{j\rightarrow e,m},
\]

where \(c\) is learned complementarity. A small residual floor prevents
premature expert death, while missing entries receive exactly zero weight.
The uncertainty multiplier \((1-u)\) is a pre-registered U2-evidence ablation,
not part of the default promoted control. It may replace the mean-only score
only if condition-wise empirical coverage passes before final configuration
freeze; otherwise \(u\) remains diagnostic and no concentration claim is made.

#### URGC claim gate

URGC remains a main contribution only if:

1. one posterior reused everywhere beats three separate parameter-matched
   routers as well as fusion-only and relay-only use;
2. it improves over same-size softmax, uncalibrated scalar, expert-only and
   modality-only gates;
3. actual intervention of high-score contributions produces larger held-out
   margin loss than low-score intervention;
4. it improves fused retrieval and at least two post-collaboration branch
   embeddings rather than only reallocating ensemble capacity;
5. it improves both mean and worst-case missing/degraded-modality retrieval
   without a material complete-modality regression.

If separate routers match it, the unified-control claim is removed. If
retrieval improves but intervention calibration does not, the method is
reported as dynamic gating rather than reliability estimation.

## 6. Auxiliary mechanism — RDPT

### Rejectable Directional Peer Teaching

The novelty audit found direct prior art for selective, reliability-aware,
heterogeneous and adaptive teacher selection. RDPT is therefore training-only
auxiliary research in v1.1, not one of the three promoted contribution slots.
It remains implemented behind the same public seam because a hard ablation may
show useful branch-specific interaction.

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

#### RDPT promotion gate

RDPT may be promoted from an auxiliary only if it:

1. improves over symmetric KL and always-on one-way distillation;
2. improves over confidence-only selection, HeteroAKD/MST-Distill-style
   adaptive transfer and wrong-payload swaps;
3. produces a non-zero but non-saturated rejection rate;
4. improves weak branches without increasing private-feature CKA to collapse;
5. gives the expected teacher-direction shifts under controlled modality
   degradation.

If directional/rejectable behavior is not observed, RDPT is removed rather
than retained on the strength of the full model alone.

## 7. Fusion and training objective

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

No loss uses test labels. CIRC targets are immutable out-of-fold training
artifacts. RDPT teacher directions are computed only for training samples. At
inference, CIRC/URGC use only learned evidence features and RDPT adds no
operation.

Training is phased to preserve target independence:

1. train three HFER-uniform target generators on the registered two-fold
   complements;
2. freeze them and generate the out-of-fold full-intervention table;
3. warm-start the global router against only immutable targets;
4. jointly optimize HFER/URGC while retaining the immutable proper-score
   anchor and never feeding router gradients into target construction;
5. after the development configuration is frozen, reclassify the former 30 dev
   identities as training-only, repeat the registered folds and target
   generation on all 171 identities, and perform no further selection.

## 8. Public software contracts

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

## 9. Minimum ablation matrix

| ID | Variant | Mechanism isolated | Necessary comparison |
|----|---------|--------------------|----------------------|
| A0 | Three independent experts + mean | extra capacity only | three single experts |
| A1 | Three independent experts + concat/MLP | learned late fusion | A0 |
| A2 | HFER-uniform | staged branch collaboration | A1 |
| A3 | A2 without private residual/role mixer | HFER structure | A2 |
| A4 | Larger no-relay concat control | capacity/FLOP floor | A2 and final model |
| R0 | HFER + same-size softmax | generic routing | A2 |
| R1 | Joint Beta router, observational end-to-end target | evidence head only | R0 |
| R2 | Joint Beta router, cheap head-only LOO target | cheap counterfactual | R1 |
| R3 | CIRC with frozen out-of-fold full interventions | causal/calibrated target | R1, R2 |
| R4 | Nine independent evidential heads | scale comparability | R3 |
| R5 | TMUR-style global non-evidential router | global arbitration | R3 |
| U0 | R3 used only at final fusion | fusion control | R3/U2 |
| U1 | R3 used only at HFER relays | communication control | R3/U2 |
| U2 | One R3 posterior reused at relays and fusion | full URGC | U0, U1 |
| U3 | Separate parameter-matched relay/fusion routers | unification test | U2 |
| C0 | shuffled targets, expert permutation and capacity-floor routes | causal negatives | U2 |
| K0 | U2 without peer teaching | promoted core | K1–K4 |
| K1 | U2 + symmetric KL | ordinary mutual teaching | K0 |
| K2 | U2 + directional teaching, no reject | direction only | K1 |
| K3 | U2 + full RDPT | candidate auxiliary | K0–K2 |
| K4 | wrong-payload and adaptive-transfer controls | role specificity | K3 |

Every promoted variant reports fused and per-branch mAP/Rank-1, parameter count,
activated FLOPs, peak memory, gradient utilization and three-expert routing
entropy. Reliability variants additionally report per-condition BCE/Brier/ECE
overall and by expert, modality, expert-modality, camera and identity-frequency
group; total/direct/relay/edge signed effects and helpful/neutral/harmful
strata; overdispersion and empirical concentration coverage; query-only versus
query-gallery agreement; proxy-to-deployed target transfer; and cheap-vs-full
agreement. Peer variants report rejection rate and private/shared CKA.

## 10. Claim and SOTA decision

The primary fair target is RGBNT201, static CLIP-B/16, 256×128, no reranking,
no test-time training. A SOTA statement requires the frozen model to exceed
both 84.1% mAP and 87.2% Rank-1 under the same protocol, with three fixed seeds
and an experiment-integrity audit. DINOv3, multi-stage external distillation and
test-time-training results remain separate tracks. Failure to cross the fair
target does not invalidate useful ablation findings, but it forbids a SOTA
claim.
