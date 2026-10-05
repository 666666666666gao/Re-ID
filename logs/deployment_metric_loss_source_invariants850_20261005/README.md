# Source-bound interpretation of the deployed role Triplet

Six exact local source copies match the current339-source map, including the pinned author loss implementation. This is a source/algebra analysis, with no remote query or model execution.

For vectors in the unit Euclidean ball, d_ap and d_an lie between0 and2 in ideal real arithmetic. The configured no-margin Triplet uses mean log(1+exp(d_ap-d_an)), so each term is bounded below by log(1+exp(-2)), approximately0.126928. The numerical distance clamp makes this conservative; floating-point roundoff is not being tested here, and this is not an attainable global minimum for an entire multi-identity batch.

The deployment-metric wrapper supplies the same joint1536 normalized vector to every author head. The author objective sums head losses, so with H heads its metric contribution is H times the configured Triplet weight times the common Triplet loss. For three heads the unweighted lower bound is approximately0.380784; this does not set or change a training weight.

Therefore, old raw-feature observations of zero Triplet loss cannot be used as an activity criterion for this normalized soft-margin objective. Nonzero loss also does not establish useful gradients or successful retrieval learning. Saved head losses combine weighted CE and Triplet, and do not recover their separate histories. Neither raw-versus-normalized aggregate loss values nor this mathematical bound identify the cause of the observed best-to-last mAP degradation. Keep the current experiment and fixed selected checkpoints unchanged.
