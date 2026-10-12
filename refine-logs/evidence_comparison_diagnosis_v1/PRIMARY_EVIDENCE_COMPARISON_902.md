# Primary evidence: frozen local comparison versus additive ReID similarity

Checked 2026-10-12. Scope: paper equations and author source, read only. No authors' programs were executed, no dependencies installed, no neural inference or training run, and no project code changed. The project observations below are supplied premises, not independently verified measurements.

## Decision the sources can inform

The reported local residual discriminates independently, while adding it to the global vector damages retrieval. Auxiliary key geometry improves, yet actual value readouts may remain similar across slots. These observations motivate a fixed comparison of the **same frozen evidence**, rather than importing a named architecture or adding another training loss. They do not identify compression as the cause, prove local evidence is complementary, establish novelty, or imply any score increase.

The clean first control is existing additive cosine versus cosine of concatenated global and weighted residual blocks, using exactly the tensors and coefficient already entering the addition. Multi-vector MaxSim is a second question only if actual local readouts are already available. It changes how those readouts are retained and compared; it must not be described as solely removing additive cross terms.

## Verified primary sources and immutable code pins

| Source | Verified identity and version | Code / license boundary |
| --- | --- | --- |
| [Value-Aware Loss Function for Model-based Reinforcement Learning, AISTATS 2017](https://proceedings.mlr.press/v54/farahmand17a.html) | Farahmand, Barreto, Nikovski; PMLR 54:1486–1494. [Main PDF](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf), [supplement](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a-supp.pdf); corroborated by [authors' MERL publication record](https://www.merl.com/publications/TR2017-049). | These inspected publication records link papers, not an implementation. No author code commit or software license was verified for this exact paper. Main PDF states author copyright; a separate reuse license was not verified in this audit. This is not evidence that author code cannot exist elsewhere. |
| [ColBERT, SIGIR 2020](https://arxiv.org/abs/2004.12832v2) | Khattab, Zaharia; arXiv v2, 2020-06-04. [Author full text](https://arxiv.org/html/2004.12832v2). | [Author repository](https://github.com/stanford-futuredata/ColBERT) links this paper. Original source pin **24f5ed52c24978690727c058917d1325b5eeafb9**, 2020-06-02, initial commit. Additional maintained-v1 pin **706a7265b06c6b8de1e3236294394e5ada92134e**, 2022-06-17, branch `colbertv1`; this is later code, not the paper-time snapshot. [Pinned MIT software license](https://github.com/stanford-futuredata/ColBERT/blob/24f5ed52c24978690727c058917d1325b5eeafb9/LICENSE). Paper arXiv record uses the [nonexclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html), distinct from the software license. |
| [SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) | Geifman, El-Yaniv; PMLR 97:2151–2159. [Main PDF](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf). | PMLR directly links [author code](https://github.com/geifmany/selectivenet). Pin **a6d0a8fd33dae61da910b61a2aae93102d2d4869**, 2019-04-24. Its complete tracked tree has no `LICENSE` or `COPYING`; no software license verified. Paper author copyright does not assign a software license. |

Pins were obtained with `git ls-remote`, then source/history read with `git show`, `git ls-tree`, and `git log`. GitHub's unauthenticated REST API was rate limited; no authenticated access was used. Scratch source repositories are under `C:/Users/gb/.codex_tmp/research_sources902/`; they were not imported or executed. Modern ColBERT `main` is explicitly not used as the SIGIR-2020 implementation.

## What each paper actually contributes

### Value-aware loss: relevance of a downstream operator, not attention values

With transition model \(\widehat P\), true transition \(P^*\), value-function class \(\mathcal F\), and state-action distribution \(\nu\), Eq. (7) defines

\[
L_{\mathrm{VAML}}(\widehat P)=\mathbb E_{(x,a)\sim\nu}\!\left[\sup_{V\in\mathcal F}\left|\mathbb E_{P^*(\cdot|x,a)}V-\mathbb E_{\widehat P(\cdot|x,a)}V\right|^2\right].
\]

The supremum is **inside** the state-action expectation. Eq. (9) substitutes observed next states and empirical sampling; Eq. (10) minimizes over the model class. “Value” is the RL value function, not the `V` tensor of attention. The motivation concerns how a model's errors affect Bellman updates; it does not diagnose ReID fusion. Its finite-sample result has model-class and sampling assumptions, not a guaranteed improvement for a different task. [Main paper, Sections 2–3, Eqs. (3), (7), (9)–(10), Theorem 2](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf).

**Inference for this project:** judge key/readout geometry by the fixed retrieval comparison that consumes it. This is an analogy motivating a diagnostic, not an application of VAML's theorem or a reason to add its training objective.

### ColBERT: independent encoding followed by directional local matching

For normalized query tokens \(Q_i\) and document tokens \(D_j\), its cosine score is

\[
S(Q,D)=\sum_i\max_j Q_i^\top D_j.
\]

Its alternative uses negative squared Euclidean distances. The sum is over query embeddings; this is asymmetric and permits many query embeddings to select the same document embedding. Query augmentation retains embeddings at mask-token positions; document punctuation filtering is a different operation. Gallery encoding can be precomputed, while query–gallery interactions remain at scoring time. This architecture was trained for passage ranking: independent encoding does not mean untrained frozen features automatically work under this score. The paper's speed findings belong to its IR workloads and implementations, not to ReID. [Paper, Sections 3.1–3.4](https://arxiv.org/html/2004.12832v2).

**Inference for this project:** late comparison can avoid mixing global and local coordinates. ColBERT does not justify unrestricted cross-part matching, ReID visibility interpretation, or a guaranteed gain.

### SelectiveNet: jointly trained abstention with a coverage objective

The paper defines coverage \(\phi(g)=\mathbb E[g(x)]\), selective risk \(R(f,g)=\mathbb E[\ell(f(x),y)g(x)]/\phi(g)\), and optimizes

\[
L=\alpha\{\widehat R(f,g)+\lambda[\max(0,c-\widehat\phi(g))]^2\}+(1-\alpha)\widehat R(h).
\]

The prediction, sigmoid selection, and auxiliary heads share a representation; the auxiliary head serves training. Section 5 calibrates a selection threshold using an **independent unlabeled validation set**, at the percentile for a chosen target coverage. Coverage calibration is not a guarantee of correctness or evidence usefulness. This is end-to-end selective prediction, not a no-training local-evidence gate; abstaining from a whole prediction differs from keeping a global retrieval prediction while excluding a local source. [Paper, Sections 2, 4–5](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf).

**Inference for this project:** report the fraction of pairs with interpretable local evidence alongside conditional outcomes. Do not transplant a selection sigmoid or calibrate on evaluation outcomes.

## Source-level scoring, loss, and mask paths

### ColBERT original 2020 implementation

- [`src/model.py`, lines 25–65](https://github.com/stanford-futuredata/ColBERT/blob/24f5ed52c24978690727c058917d1325b5eeafb9/src/model.py#L25): `forward` independently encodes each side then calls `score`. Query and document outputs are projected and L2 normalized. The cosine path uses matrix multiplication, maximum over documents, then query sum; the L2 path uses negative squared distance followed by the same reductions. Document mask zeros punctuation/padding embeddings before normalization.
- [`src/training/data_reader.py`, lines 32–69](https://github.com/stanford-futuredata/ColBERT/blob/24f5ed52c24978690727c058917d1325b5eeafb9/src/training/data_reader.py#L32): query/positive/negative triples produce two scalar scores; cross entropy targets column zero. It trains the encoder and projection through that ranking score. This is not an auxiliary key-diversity loss.
- [`src/indexing/encoder.py`, lines 20–29 and 41–53](https://github.com/stanford-futuredata/ColBERT/blob/24f5ed52c24978690727c058917d1325b5eeafb9/src/indexing/encoder.py#L20): offline index creation removes mask-false rows and can store fp16 vectors. [`src/evaluation/ranking.py`, lines 10–26](https://github.com/stanford-futuredata/ColBERT/blob/24f5ed52c24978690727c058917d1325b5eeafb9/src/evaluation/ranking.py#L10) repads indexed documents with zeros before `score`.

**Concrete mask limitation (algebraic inference from these paths):** a zero padded document vector has cosine dot product zero. If all valid matches for a query vector are negative, padding can win the max. Zero multiplication is therefore not equivalent to excluding an invalid location. For L2, a normalized query's distance to a zero vector is 1, so invalid padding can also beat a more distant valid vector. No assertion is made about how often this occurred in the paper's data.

### Maintained v1 deployment path, explicitly later than 2020

[`colbert/modeling/colbert.py`, lines 43–68](https://github.com/stanford-futuredata/ColBERT/blob/706a7265b06c6b8de1e3236294394e5ada92134e/colbert/modeling/colbert.py#L43) keeps the same scoring and zero-mask behavior. [`colbert/ranking/index_ranker.py`, lines 94–98](https://github.com/stanford-futuredata/ColBERT/blob/706a7265b06c6b8de1e3236294394e5ada92134e/colbert/ranking/index_ranker.py#L94) multiplies similarity by a document-length mask before max, retaining the zero-floor issue. [`colbert/ranking/faiss_index.py`, lines 64–118](https://github.com/stanford-futuredata/ColBERT/blob/706a7265b06c6b8de1e3236294394e5ada92134e/colbert/ranking/faiss_index.py#L64) searches token vectors and maps hits to unique passage IDs. [`colbert/ranking/rankers.py`, lines 28–43](https://github.com/stanford-futuredata/ColBERT/blob/706a7265b06c6b8de1e3236294394e5ada92134e/colbert/ranking/rankers.py#L28) scores that candidate set and sorts. Candidate retrieval and local comparison are distinct stages; finite candidate depth does not give exhaustive-gallery ranking.

### SelectiveNet publication code differs from the paper equation

[`models/cifar10_vgg_selectivenet.py`, lines 134–149, 214–232, 268–275](https://github.com/geifmany/selectivenet/blob/a6d0a8fd33dae61da910b61a2aae93102d2d4869/models/cifar10_vgg_selectivenet.py#L134) builds softmax prediction, sigmoid selection, and auxiliary softmax outputs. Its `selective_loss` passes **g-weighted targets** into categorical cross entropy and adds \(32\max(0,c-\operatorname{mean}(g))^2\). There is **no explicit division by mean(g)** in that loss, unlike the paper's selective risk. The accuracy metric does divide by the count selected with `g > 0.5`; compilation applies the two loss weights. The other two published model files repeat this loss pattern. Backend/version dependencies were not pinned by this repository; exact runtime loss semantics were not executed here.

[`selectivnet_utils.py`, lines 23–38 and 65–70](https://github.com/geifmany/selectivenet/blob/a6d0a8fd33dae61da910b61a2aae93102d2d4869/selectivnet_utils.py#L23) derives a percentile from `model.predict()` and uses a strict threshold. In the CIFAR class, [`predict`, lines 166–169](https://github.com/geifmany/selectivenet/blob/a6d0a8fd33dae61da910b61a2aae93102d2d4869/models/cifar10_vgg_selectivenet.py#L166) defaults to `x_test`. The utility therefore does not expose the paper's independent validation calibration data path. Do not copy this helper as validation-safe calibration. These discrepancies warrant precise descriptions, not a claim that the released code implements every paper equation exactly.

## Fixed controls for the project: mathematical design, not measured results

### A. Use the exact residual that currently enters addition

Let \(g_x\) and \(r_x\) be the actual tensors immediately before existing additive fusion, and \(a\) its already fixed coefficient. Preserve their current preprocessing, checkpoint, IDs, camera exclusions, gallery, query order, tie handling, and precision. Do not separately renormalize branches unless the existing path already does so.

Reference global score is \(s_g(q,d)=\cos(g_q,g_d)\). Reproduce

\[
s_{\mathrm{add}}(q,d)=\frac{g_q^\top g_d+a(g_q^\top r_d+r_q^\top g_d)+a^2r_q^\top r_d}{\|g_q+ar_q\|\,\|g_d+ar_d\|}.
\]

Use one fixed late scalar comparison, equivalently normalized block concatenation:

\[
s_{\mathrm{block}}(q,d)=\cos([g_q;ar_q],[g_d;ar_d])
=\frac{g_q^\top g_d+a^2r_q^\top r_d}{\sqrt{\|g_q\|^2+a^2\|r_q\|^2}\sqrt{\|g_d\|^2+a^2\|r_d\|^2}}.
\]

This retains the exact same residual and coefficient, removes cross-channel dot products, and removes within-image global–residual alignment from the normalization. It is not a pure numerator-only intervention. If both branches already have unit norm, \(s_{\mathrm{block}}=(s_g+a^2s_r)/(1+a^2)\), with \(s_r=\cos(r_q,r_d)\). The squared coefficient follows from the algebra; replacing it with a newly tuned score weight changes the control. If the production coefficient varies by image, use its actual \(a_q,a_d\): the numerator local factor becomes \(a_qa_d\), with the corresponding per-image denominators.

Inspect additive numerator and norm components directly for attribution. No additional fusion coefficients, temperatures, thresholds, checkpoint choices, or parameter sweep are needed. A better block score would support an **operational comparison-path explanation** at this frozen checkpoint; it would not prove where information was lost inside the encoder.

### B. Match actual frozen readouts only if the project already exposes them

Let \(Z_x=\{z_{xi}\}\) be actual value readouts, not auxiliary keys or a key-only projection. Use unit directions \(u_{xi}\) for defined nonzero readouts and only actually valid locations. For unrestricted set comparison define

\[
M_{q\to d}=\frac1{|I_q|}\sum_{i\in I_q}\max_{j\in I_d}u_{qi}^\top u_{dj},\qquad
M_{\mathrm{sym}}=\tfrac12(M_{q\to d}+M_{d\to q}).
\]

The normalized average keeps the score within the cosine range and avoids a raw token-count multiplier. It is a **ReID design choice**, not the original ColBERT sum. Symmetry uses the same interaction matrix's row and column maxima; it does not enforce a one-to-one correspondence. With \(Q=\{e_1,e_2\}\), \(D=\{e_1,e_1\}\), and orthogonal unit \(e_1,e_2\), forward mean is 0.5, reverse mean is 1, and symmetric mean is 0.75. A shared repeated target still satisfies several source slots.

If existing slot identities encode corresponding body parts, a fixed same-slot average over verified common valid slots is the more direct semantic comparison; unrestricted MaxSim can match different body parts or common background. Decide correspondence from the existing representation, before examining retrieval outcomes. Swapping direction is a diagnostic of asymmetry, not a license to choose the direction that scores best.

For an already unit-normalized global/residual implementation, one fixed combined comparison can replace the residual cosine term in the preceding block formula with the predeclared local comparator \(M\): \((s_g+a^2M)/(1+a^2)\). This retains a predetermined branch weight but changes pooling/matching. If raw readouts and the added residual are different objects, label this as **same frozen extractor, different retained representation**; the block control in A remains the exact same-residual comparison. Do not silently substitute keys for values or claim that both comparisons use identical evidence.

### C. Separate key geometry, actual readout diversity, and retrieval usefulness

For existing attention \(A\) and values \(V\), actual outputs obey \(Z=AV\). Slot difference is \(z_i-z_j=(A_i-A_j)V\). Separated keys do not force this quantity to be large; distinct attention rows can also produce similar outputs for the actual values. Conversely, all slots can repeat one image-discriminative vector: small within-image diversity does not establish that the entire local residual lacks identity information.

If all unit readouts in one image equal \(u_q\), and all in another equal \(u_d\), then every valid local interaction is \(u_q^\top u_d\). Both directional means and symmetric MaxSim reduce to that one-vector cosine. MaxSim cannot manufacture additional slot information. Audit existing readout directions and pairwise score matrices, rather than inferring value diversity from key loss or attention entropy alone. No compression mechanism is established by this calculation.

### D. Distinguish absent evidence from conflicting evidence

The mathematical max must exclude invalid targets; multiplying their similarities by zero does not do so. This is a concrete source-code concern above, not a hypothetical robustness feature. A pair with no eligible local comparison has an undefined local evidence score, not a negative identity vote. Report its availability separately. Do not infer physical visibility from a text-token mask, high attention entropy, low MaxSim, or a learned selection probability. No new visibility head or missing-data policy is prescribed here.

For fixed labeled audit pairs define \(\Delta_g=s_g(q,p)-s_g(q,n)\) and \(\Delta_l=s_l(q,p)-s_l(q,n)\), where \(p,n\) are valid positive/negative gallery samples chosen by a fixed rule. A valid local margin against the global ordering is a disagreement; it becomes useful correction only if the ground-truth ordering and full evaluation support it. Low or missing local support is “unknown”; valid evidence favoring the wrong identity is “conflict/error”. SelectiveNet's accept/reject output does not supply this distinction.

## Measurement and deployment boundaries

Use full frozen evaluation once for the predefined scores. Report mAP/Rank-1 under identical exclusions, paired per-query changes, and counts of global errors corrected versus global successes damaged. Local-only separability and favorable margins among global errors are diagnostic evidence; neither alone proves a usable combined score improves. Log actual score components and readout identity so evaluation cannot accidentally compare keys to residuals. Do not select a weight, comparator, or checkpoint using the evaluation outcome.

| Comparison | Cached gallery storage | Exact comparison cost per pair | Deployment consequence |
| --- | --- | --- | --- |
| Current single fused vector | \(O(Nd)\) | \(O(d)\) | One vector per gallery image. |
| Separate global/residual blocks | \(O(N(d_g+d_r))\) | \(O(d_g+d_r)\) | Still one concatenated vector for cosine indexing; larger vectors and transfers. If both caches already exist, no new encoder pass is required. |
| Local sets with \(m_q,m_d\) readouts | \(O(Nm_dd_l)\) | \(O(m_qm_dd_l)\), plus reductions | Interaction matrices cost \(O(m_qm_d)\) temporary storage. Symmetry can reuse the matrix. A single-vector similarity index is not an exact representation of general MaxSim. |
| Fixed global-candidate local reranking | Same local cache, scores for fixed \(K\) candidates | \(O(Km_qm_dd_l)\) per query | Cannot recover positives excluded from the candidate set. Report candidate recall and reranking scope; do not report candidate-restricted mAP as exhaustive-gallery mAP. |

These are elementary cost bounds for the proposed comparisons, not latency benchmarks. Local features, masks, IDs, transfer bandwidth, and index construction remain real costs even when gallery encoding is offline. The later-v1 ColBERT candidate and ranking paths above illustrate that separation; neither its later optimizations nor its passage-search speed numbers establish ReID deployment cost.

## Minimal recommendations

1. First compare reproduced additive cosine against the exact same-residual block cosine at the existing coefficient. This answers the present comparison-path question with the fewest changes.
2. Treat actual readout matching as a separately named fixed diagnostic. Use existing semantics to choose aligned slots or a predeclared symmetric set comparator. Do not import ColBERT's zero masks, asymmetric sum, tokenizer, or encoder training recipe.
3. Record corrections, damage, evidence availability, and actual value-readout dispersion. Keep key geometry and value-readout usefulness as different measured claims.
4. Do not add VAML or SelectiveNet training objectives for this no-retraining study. The three sources offer precedents and concrete implementation boundaries; **complementarity, novelty, causal explanation, and expected improvement remain unestablished**.
