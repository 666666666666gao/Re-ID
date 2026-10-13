# Raw-role objective geometry: primary-source evidence

2026-10-13. Source-only research; no model imports, NN forward, training, installation, SSH, or production changes by this research agent.

**Supported conclusion:** A train-mode BN classification task plus raw Euclidean Triplet does not constrain a common translation of its input features, under the assumptions below. This is an output-level property of those task terms. It does not prove an independently reachable parameter-flat direction, actual drift, identity/noise content, or a harmful effect on final retrieval.

## 1. Actual author source and the conditional symmetry

Signal is pinned to `cd1b0a672d1fe642e7608731cb4899a19dda7d51`. The four installed files supplied by the parent match the public pinned author files byte-for-byte; the comparison and SHA256 values are saved privately in `SIGNAL_INSTALLED_PUBLIC_COMPARISON.json`.

- The direct and auxiliary identity heads are `BatchNorm1d` followed by a bias-free `Linear`. The BN bias is frozen. Training computes classification from BN output and returns the pre-BN raw features. Its `training=False` branch returns raw `ori`, or raw concatenated `ori` and `vars_total`, without a retrieval BN. These are author-source facts, not verification of the current TriFusion invocation. [Signal head, lines 41–79, 122–169, 223–228](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/modeling/make_model.py)
- `TripletLoss.__call__` defaults to `normalize_feature=False`; `make_loss` calls it without changing that flag. Distances are Euclidean, then label-based hard mining and margin/soft-margin ranking are applied. The instantiated `center_criterion` is not added in this file's `softmax_triplet` return expression. [Triplet, lines 16–31 and 106–135](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/layers/triplet_loss.py), [loss assembly, lines 37–73](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/layers/make_loss.py)

PyTorch documents per-channel batch mean/variance during training and running mean/variance during evaluation by default. With `track_running_stats=False`, evaluation also uses batch statistics. [Official BatchNorm1d reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html)

**Algebraic derivation, rather than an author claim:** Fix one BN statistics group and labels. For any common vector \(v\in\mathbb R^d\), replace every pre-BN feature \(c_i\) by \(c_i+v\). Then

\[
\mu_B'=\mu_B+v,\quad \sigma_B'^2=\sigma_B^2,
\quad \operatorname{BN}_B(c_i+v)=\operatorname{BN}_B(c_i).
\]

Thus train-BN logits, and therefore ordinary or label-smoothed CE, are unchanged for **every** common direction. Also

\[
\|(c_i+v)-(c_j+v)\|_2=\|c_i-c_j\|_2,
\]

so raw Triplet distances, hard-mined pairs and ranking loss are unchanged. This follows directly from the inspected author's distance and mining code; the official Triplet reference likewise defines its distance from feature differences. [Official TripletMarginLoss reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginLoss.html)

The assumptions are that the same translation reaches all samples within each BN statistics group and all samples compared by Triplet; BN actually uses current batch statistics; and Triplet sees raw translated features. The `training=` argument of Signal's `forward` is not itself the BN module mode. PyTorch's `train(mode)`/`eval()` controls that mode; freezing parameters is a separate operation. [Official Module reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.train)

Important limits:

- With fixed evaluation buffers, write \(D=\operatorname{diag}(\gamma/\sqrt{\mathrm{running\_var}+\epsilon})\). Translation changes BN output by \(Dv\). For classifier matrix \(W\), softmax probabilities remain identical iff \(WDv\in\operatorname{span}(\mathbf1)\), equivalently \((w_k-w_1)^TDv=0\) for every class. This is a smaller conditional subspace, not arbitrary translations. Translating the running mean together with features restores cancellation; a fixed checkpoint does not perform that transformation automatically.
- Training can update running buffers even when its current logits are invariant. Sample-, identity-, modality- or subgroup-dependent shifts generally do not cancel across compared groups. Normalization before Triplet, sample-dependent fusion gain, or other feature terms can invalidate the stated symmetry.
- The representation-level symmetry does not establish a realizable independent parameter direction. Architecture constraints, affine-free outputs, shared parameters, parameter weight decay and other objectives must be checked separately. No runtime behavior was inferred from the source snapshots.

## 2. Two direct precedents, and what they do not establish

| Mechanism | Primary evidence and exact relevance | Boundary |
|---|---|---|
| BNNeck: choose and verify the retrieval feature space explicitly | Luo et al. distinguish pre-BN \(f_t\) for Triplet from post-BN \(f_i\) for ID classification, and select post-BN \(f_i\) with cosine retrieval in inference. The pinned author implementation returns raw `global_feat` to Triplet during training, offers BN-before/after test output, and applies optional test feature L2 normalization. [Paper, §III-E and §IV-C.5](https://arxiv.org/pdf/1906.08332), [head, commit `3da7e6f03164a92e696cb6da059b1cd771b0346d`, lines 145–172](https://github.com/michuanhaohao/reid-strong-baseline/blob/3da7e6f03164a92e696cb6da059b1cd771b0346d/modeling/baseline.py#L145), [test metric, lines 33–49](https://github.com/michuanhaohao/reid-strong-baseline/blob/3da7e6f03164a92e696cb6da059b1cd771b0346d/utils/reid_metric.py#L33) | A feature-space precedent, not a proof that BN retrieval or centering is correct for Signal/TriFusion. BN is per-channel standardization, not per-sample L2 normalization. |
| NormFace: train the normalized angular representation with scaled cosine classification | Wang et al. explicitly identify the unnormalized training inner-product versus cosine test metric gap (§3.1). They normalize features and class weights, remove classification bias and introduce a scale because unit cosine logits have limited range (§3.3). Their pinned author graph is `Normalize(feature)` → weight-normalized bias-free `InnerProduct` → learned `Scale` → `SoftmaxWithLoss`. [Paper, §3.1–3.3](https://arxiv.org/pdf/1704.06369), [author graph, commit `8438887d33023bf35f16ba5e8c21a422c49cd520`](https://github.com/happynear/NormFace/blob/8438887d33023bf35f16ba5e8c21a422c49cd520/prototxt/scaled_cosine_softmax.prototxt) | A direct normalization/metric mismatch precedent. It is face verification, not frozen-global additive residual ReID; it provides no TriFusion gain estimate or automatic transplant justification. |

For the supplied deployment expression \(u_i=\operatorname{Normalize}(g_i+a c_i)\), keep \(g_i\) and scalar gain \(a\) fixed. Away from zero norm, its response to a common residual translation is

\[
\delta u_i = \frac{a}{\|g_i+a c_i\|_2}(I-u_i u_i^T)v.
\]

This independent geometric derivation explains why a translation invisible to the conditional raw-role task can still change deployed angles. It does **not** determine whether a particular translation helps or harms ranking. Applying angular supervision only to standalone \(\operatorname{Normalize}(c)\) is not the same as supervising the actual fused deployment vector. Neither inspected precedent implements this exact frozen-global-plus-residual setup.

The supplied 94.09%/82.59% common-energy diagnostics were not independently recomputed here. A large mean/common component alone establishes neither identity irrelevance nor nuisance content, nor a cause of final errors. The supported next evidence link is to verify the current task inputs/modes against the assumptions and connect their observable geometry to final identity ranking; no method rollout is supported by this research alone.

Private supporting snapshots and integrity manifest: `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/objective934/research/`. These contain the pinned Signal, BNNeck and NormFace source files, official BN/Triplet documentation excerpts, fetching scripts, per-file SHA256 manifest and installed/public comparison. The BNNeck/NormFace commits were resolved with HTTPS `git ls-remote`, then raw files were fetched by those full commits; GitHub's unauthenticated commits API returned HTTP 403 rate-limit before that pin was obtained. Installed Signal files remain separately under the parent's `objective934/author_source/` directory.
