# Saved-source positive-relation coverage diagnostic

Registered before this diagnostic is executed, 2026-09-08. The role-set Q1 run is sealed; its independent terminal audit is still running. This document does not register another training run or change its result.

Question: among the source positive positions whose distance exceeds at least one negative, how many are strictly below the single hardest-positive distance, and how does this split change when only different-scene positives are considered?

Inputs: all six source training distance streams, their complete 260-step metadata/training files, existing CPU bindings, and the source OOF protocol. No held-out ranking arrays, official images, model checkpoints, model forwards, or optimization. All binary inputs remain remote. The worker prints text statistics only.

The current batch has 64 anchors; historical candidates are not added as anchors. Positive masks follow actual identity labels and exclude the anchor's own current position. Other repeated views of the same record remain present and are counted separately. Negatives use different identity, including same-scene negatives. The different-scene subset only restricts positive positions; anchors with no such positives remain part of source accounting and are not deleted from other anchors' candidates.

A strict inversion is d(positive) > d(negative). Equal distances are counted separately and do not count as strict inversions. Strictly nonmaximal positives satisfy d(positive) < max(d(all true positives)); all maximum-distance ties are excluded from that category. Thus the category has no direct positive-distance derivative in the actual single-hardest-positive fused term, but this is not a claim of zero total encoder gradient or absence of supervision from other anchors/heads.

Report all steps, prewarmup steps1-65, postwarmup66-260, and final65steps separately for every fold/endpoint. Count source anchor exposure, positive positions/records, cross-scene positive availability, strict inversions, affected positive positions, strictly nonmaximal affected positions, historical affected positions, and negative-distance ties. Count when every maximum-distance positive is same-scene despite different-scene positives being available. These are repeated training exposures, not independent sample sizes and not a new ReID benchmark score.

Checks: validate CPU-bound sizes/SHA and source identities/scenes; read every saved distance element in stream order; independently match each row's existing expanded_wrong_order_anchors; compare sorted-negative inversion counts with explicit positive-by-negative comparisons on each endpoint's first actual batch; enforce count decompositions. No synthetic performance data. Runtime expected under one minute, NumPy CPU only, no remote files written.

Interpretation remains descriptive. Finding nonmaximal inversions does not establish that a different loss would generalize, that the current max objective cannot eventually fix them, or that hard positives are noisy. No loss family, temperature, margin, candidate count, or new training is selected by this diagnostic.
