# TriFusion RGBNT201 V13 target-learnability diagnostic

## Outcome

The zero-training diagnostic explains why V13-Q1 failed. The fixed-temperature
pointwise distillation target is almost uniform over the nine expert-modality
actions, while the preferred slot changes materially across identity folds.
The evidence supports replacing the pointwise utility-KL objective with a
fold-local retrieval objective. It does not support changing the temperature,
rerunning V13, accessing dev, or claiming a metric gain.

## Protocol boundary

- Source commit: `e6774432aba906cbb27913eb213984fbbc6b8678`.
- Exact paired-cache SHA-256:
  `1cc499a1acb7b12336f19de0e74ad4ef452dae8b2aa8299e4a16e2d619e15e27`.
- RGBNT201 fit-only records: 571 queries, 21 identities, three identity folds.
- Router training: false; optimizer steps: 0.
- Dev accesses: 0; official-test accesses: 0.
- Evidence SHA-256:
  `b125be261bef5f3923f26a93d94a37a2d7dd60aed256b70dcc227660907431c6`.

## Raw results

| Quantity | Value |
|---|---:|
| Utility mean | 0.0001897726 |
| Utility standard deviation | 0.0014492861 |
| Utility minimum / maximum | -0.0048559904 / 0.0047319531 |
| Top-1 minus Top-2 utility, 5th percentile | 0.0000277758 |
| Top-1 minus Top-2 utility, 25th percentile | 0.0001514554 |
| Top-1 minus Top-2 utility, median | 0.0003407001 |
| Top-1 minus Top-2 utility, 75th percentile | 0.0006257296 |
| Top-1 minus Top-2 utility, 95th percentile | 0.0013096035 |
| V13 distillation temperature | 0.05 |
| Target normalized entropy, mean | **0.9998319745** |
| Target normalized entropy, min / max | 0.9991291165 / 0.9999912977 |
| Mean maximum target probability | **0.1152755395** |
| Uniform nine-way probability | 0.1111111111 |
| Fold preferred slots | 2 / 0 / 2 |
| Fold 0 vs 1 slot-mean rank correlation | **-0.5000** |
| Fold 0 vs 2 slot-mean rank correlation | 0.4000 |
| Fold 1 vs 2 slot-mean rank correlation | 0.0500 |
| Identity-majority winner accuracy, macro | 0.4911836088 |
| Identity-majority winner accuracy, micro | 0.4991243433 |
| Within-identity centered utility cosine | 0.5229061246 |
| Between-identity centered utility cosine | 0.0406050235 |
| Max absolute residual-norm/utility correlation | 0.1138224006 |
| Max absolute direct-residual-cosine/utility correlation | 0.3880828917 |

## Findings

1. **Observation:** at temperature 0.05, normalized target entropy is
   0.999832 and the mean maximum probability is only 0.11528 versus the
   uniform 0.11111. **Interpretation:** the KL target supplies almost no
   nine-way preference signal. **Implication:** V13-Q1 is not evidence that a
   deployment-feature Router cannot help; it is evidence that this pointwise
   target is poorly conditioned. **Next experiment:** remove the pointwise
   utility-KL rather than sharpen it with a post-hoc temperature scan.

2. **Observation:** the median Top-1/Top-2 raw utility gap is 0.0003407 and
   fold slot rankings correlate from -0.50 to 0.40. **Interpretation:** absolute
   per-sample slot labels are both close and fold-dependent. **Implication:** a
   pooled action classifier can improve one fold while sacrificing another,
   exactly as V13-Q1 did on fold 2. **Next experiment:** compute retrieval risk
   separately in each teacher coordinate system and minimize the worst
   training-fold regret.

3. **Observation:** within-identity utility similarity is 0.5229, but an
   identity-majority winner predicts only 49.91% of rows. **Interpretation:**
   identity context carries signal, yet no stable single action represents an
   identity. **Implication:** the learning target should preserve query-to-
   gallery relations rather than collapse each query to one nine-class label.
   **Next experiment:** backpropagate through the composed, normalized retrieval
   embeddings and their cross-camera positive/negative distances.

4. **Observation:** residual norm has at most 0.1138 absolute correlation with
   utility, while direct-residual cosine reaches 0.3881 in only the strongest
   slot. **Interpretation:** simple local quality magnitude is insufficient to
   reproduce actual retrieval utility. **Implication:** quality supervision can
   remain a corruption-response control, but it cannot be the main identity
   routing objective. **Next experiment:** keep the existing quality loss and
   replace only the identity objective.

## Claim boundary

This diagnostic identifies a target-conditioning and cross-fold stability
problem. It does not establish causality, a new deployable checkpoint, a dev
improvement, an official-test result, or SOTA. V13 remains sealed as
`Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE`.

