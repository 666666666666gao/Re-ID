# Independent result-to-claim response

Verdict: `no`, confidence `high`.

V9's engineering contract is supported: Phase-A and Router stayed frozen,
Signal and Phase-B are exact prefixes, the two-round relay executes and is
approximately orthogonal, and the registered B64/K8 60-epoch run completed
with dev0 during training and official0.

The scientific claim fails. V9 fused is 56.5339 mAP / 57.2121 Rank-1, below
Phase-B by 1.8711 mAP / 2.1818 Rank-1 and below exact Signal by 1.4770 mAP /
0.2424 Rank-1. It is 8.4661 mAP below the 65 gate. Beating three degraded V9
expert outputs does not demonstrate synergy. Beta saturation is an observation,
not a causal explanation without an ablation.

Seal V9. Do not run ablations, multiple seeds, official test, or beta/epoch/LR/
residual/checkpoint scans. Any new representation hypothesis must first pass an
identity-disjoint fit-only positive-retrieval-utility gate and suppress harmful
additions using train-side evidence before a new final-only dev read.
