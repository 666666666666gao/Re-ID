# Current objective: discrimination is not an explicit complement requirement

Scope: source/math analysis of the existing loss, with the six fixed-best results from §41.919 as the empirical boundary. No training, model forward, new comparator, new score, seed or coefficient was run. This note does not locate the unique cause of the observed retrieval losses.

The authoritative source is `modeling/trifusion/region_use_objective.py:18-40`; the call and ordinary author tasks remain in `tools/run_region_use_probe.py:58-62`. The auxiliary is added to the existing full objective; the conclusions below concern the auxiliary alone.

For one eligible query, enumerate its legal positive-negative relations by t. Write its detached global margin as g_t, regional margin as r_t, s=√128, and w_t=1 (uniform) or σ(−s g_t) (ambiguity). Apart from the outer eligible-query mean:

    L_i = Σ_t w_t softplus(−s(g_t+r_t)/2) / Σ_t w_t.

1. With global and weights fixed, the derivative with respect to a regional margin is:

       ∂L_i/∂r_t = −(s/2)(w_t/Σ_u w_u) σ(−s(g_t+r_t)/2).

   It is negative for every legal relation with positive weight, including a relation that global already ranks correctly. This is a margin-level derivative; regional margins share model parameters, so it does not assert that every margin can be independently increased or that parameter gradients always follow this direction. This auxiliary asks for better regional identity margins. It does not by itself specify that those margins must contain information absent from global. The ordinary fused classification/metric tasks still act elsewhere in the full training objective.

2. Per-query normalization removes a common multiplier on all relation weights of that query. The ambiguity condition changes the relative weighting of its relations, not the query's overall inclusion weight. When all legal global margins for a query are equal, its common sigmoid weight cancels and the ambiguity and uniform objectives are equal, even when that common margin is positive. Under ideal unit-vector cosine geometry, margins are in [−2,2] and the sigmoid at the worst bound is larger than the denominator clamp1e−12 for a nonempty relation set. This is a real-arithmetic statement; it is not a certification of CUDA/AMP rounding or the observed margin distributions.

3. Zero logged `global_wrong_triplets` means no sampled legal margin was nonpositive. It does not mean the auxiliary or role gradients are zero, nor that the weights have ceased to act. Logged repeated training exposures are not unseen-identity query/gallery tests. No exact ambiguity-weight distribution was saved, so none is inferred here.

4. The earlier private analysis `SOURCE_LOSS_DUPLICATION_BOUND_912.json/.md` already exhibited a source-identity regular-simplex construction: copying the same global identity vector into every region can make this auxiliary small without adding information. Reuse that bounded construction; do not label it a new experiment or assert that the trained regions actually take that form. It concerns a feasible auxiliary representation, not a realized architecture solution or the optimum of the full loss.

The fixed-best empirical check is now available: regional MaxSim R and the actual auxiliary's fixed half(global+R) score J are below same-model G on all six cases. That rules out claiming an already stronger local teacher from this specific comparator. It does not prove that every region is useless, that all local comparators fail, or that additive compression is the unique bottleneck.

Consequences for a future proposal, if training is resumed:

- State which actual evidence-use gap is being changed. A sharper attention map or smaller standalone loss is not the target result.
- Distinguish “different features,” “source identity discrimination” and “net improvement over the existing identity relation.” Explicitly examine whether copying global can satisfy the claimed supplementary responsibility.
- Use existing repair/keep, responsibility, prototype and ranking experiments as direct prior controls. A familiar loss under a new name is insufficient.
- Check transfer and damage together: same-model global/fused and full-positive AP, not only first-rank repairs or branch-alone mAP.
- A source-feature suppression method may encourage other features, but this would still need evidence that those features complement clean global and survive unseen identities. Treat it as a mechanism hypothesis, not an expected gain.

Next action is primary-source verification of an external method addressing only the evidence-learning gap. Current training and six-best diagnosis remain stopped; this note authorizes no new experiment.
