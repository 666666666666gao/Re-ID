# Source-bound metric geometry analysis

Completed locally at 2026-10-05 05:38:45 CST. Eight relevant source files match their SHA entries in the existing scientific339 scope. This work executes scalar mathematics only. It does not construct a model, replay a training batch, obtain official metrics, or modify the running experiment.

The actual entry keeps raw fused features for author classification, but supplies the same normalized joint1536 vector to every role Triplet. The global author objective continues to use its original raw features. `TripletLoss()` uses Euclidean hard mining and `SoftMarginLoss` with the default zero hard factor. The author loss wrapper sums heads. Vehicle role heads therefore include three copies of the same joint Triplet; they are not averaged.

For unit vectors, Euclidean distance is between zero and two. With hardest positive distance p and nearest negative distance n, the per-anchor metric term is log(1 + exp(p - n)). In exact-real arithmetic, p - n is at least -2, so the term is at least **0.126928011043**. Three unweighted copies are at least **0.380784033129**. These are loose universal geometry bounds; actual multi-identity batches may impose a higher lower bound. They are not tolerances for the float32 implementation, not bounds on cross entropy, and not a new empirical training measurement.

An ideal four-vector scalar example holds all feature directions and Euclidean ordering constant while multiplying every raw feature by a positive scale:

| Raw scale | Raw soft-margin mean | L2 soft-margin mean |
|---:|---:|---:|
| 1 | 0.139706883262 | 0.139706883262 |
| 2 | 0.022232035792 | 0.139706883262 |
| 5 | 0.000075774495 | 0.139706883262 |
| 10 | 0.000000005742 | 0.139706883262 |

Thus a smaller absolute raw Triplet loss can occur with identical retrieval ordering. Comparing raw and normalized loss magnitudes does not establish which model learned better retrieval.

For nonzero h outside the normalization epsilon branch, z = h / ||h|| and the metric gradient is `(I - zz^T) upstream / ||h||`. Its dot product with h is zero. The recorded scalar example gives a radial dot product of approximately 5.55e-17. The normalized metric term alone therefore provides no direct reward for uniform radial growth. Increasing the correction relative to a detached global can change the fused direction, but this derivation does not identify the cause of measured correction growth or CMC damage. Raw classification and optimizer effects are separate paths and are not covered by the tangent-gradient conclusion.

The nonzero softplus derivative with respect to the margin also does not prove that every parameter receives a nonzero gradient: normalization, hard mining, feature derivatives and cancellation matter. This analysis neither explains nor repairs the original MSVR310 native M0 failure.

The original failures, current queue, source contract, epoch selection and performance gates remain unchanged. Decisions about another scientific intervention still await the full available endpoints and fixed-best evidence. No margin, scale, seed or objective adjustment is authorized by this analysis.
