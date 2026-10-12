# Frozen best: five-score evidence-use diagnosis

All six original selected bests were evaluated once, with no optimizer updates or model/buffer changes. Training remains stopped. The six original fused distance matrices replayed with maximum absolute difference 0; original metrics agreed within the registered tolerance. Only physical GPU0/1 on2026 were used. The controller completed at2026-10-13T06:42:30.084846+08:00 and the actual supervisor EXIT was0 at06:42:30.530957. Both controller and supervisor, and all six children, were absent at the06:47 observation.

Scores below use the same best and the same complete legal query/gallery. G: same-model global1536 cosine. C: correction1536 cosine. F: original1536 additive deployment. R: symmetric meanMaxSim over the actual final16 regions, separately in all nine role/modality spaces. J: the fixed existing training score0.5*(G cosine+R); its coefficient was not fitted.

| Dataset | Condition | Epoch | G mAP | C mAP | F mAP | R mAP | J mAP | F−G | J−G |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RGBNT201 | uniform | 8 | 74.2967 | 50.6262 | 74.7880 | 53.2971 | 70.9668 | +0.4914 | -3.3299 |
| RGBNT201 | ambiguity | 8 | 74.2967 | 50.0877 | 74.6299 | 52.8910 | 71.4255 | +0.3332 | -2.8711 |
| RGBNT100 | uniform | 26 | 83.9654 | 44.3389 | 83.2858 | 48.2150 | 72.4028 | -0.6797 | -11.5626 |
| RGBNT100 | ambiguity | 26 | 83.9654 | 41.5517 | 83.3459 | 46.3867 | 70.8068 | -0.6195 | -13.1587 |
| MSVR310 | uniform | 38 | 50.5421 | 9.7260 | 50.5418 | 12.1669 | 33.0338 | -0.0004 | -17.5084 |
| MSVR310 | ambiguity | 38 | 50.5421 | 9.7327 | 50.5458 | 12.1850 | 32.9001 | +0.0036 | -17.6421 |

Full mAP/R1/R5/R10 values for all30 scores are in SCORES.csv; rows never mix epochs.

| Dataset | Condition | F−G R1 | F first repairs/new errors | Correction/G norm ratio | G→F degrees | Independent global epoch | Same-G minus independent same-epoch mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | uniform | +0.0000 | 6/6 | 0.059785 | 3.3432 | 8 | +0.0000000000 |
| RGBNT201 | ambiguity | +0.1196 | 8/7 | 0.053811 | 3.0372 | 8 | +0.0000000000 |
| RGBNT100 | uniform | -1.0496 | 10/28 | 0.197230 | 10.7396 | 26 | -0.0000052092 |
| RGBNT100 | ambiguity | -0.8163 | 9/23 | 0.188399 | 10.3586 | 26 | -0.0000052092 |
| MSVR310 | uniform | -0.1692 | 0/1 | 0.005513 | 0.3157 | 38 | +0.0000006966 |
| MSVR310 | ambiguity | -0.1692 | 0/1 | 0.005494 | 0.3146 | 38 | +0.0000006966 |

1. In all six cases the existing fixed training-pairwise comparator underperforms the same-model global in mAP and Rank-1. This diagnosis does not establish a stronger local teacher or evidence that additive compression is the primary bottleneck. It constrains this exact region/comparator combination; it does not exclude every local comparison or prove that the regions contain no complementary information.
2. RGBNT201 has a small positive additive mAP increment, RGBNT100 loses mAP and Rank-1, and MSVR310 is almost unchanged in mAP while losing one first-rank query. No stable three-dataset deployed benefit has been established.
3. The current same-model global metrics match the independent global history at the selected epoch up to the original floating-point reporting tolerance. In RGBNT100 the relevant comparison is independent global epoch26, not its independently selected epoch7 best. Therefore the gap to independent best must not all be described as shared-backbone harm. Text metric agreement is not proof that all features or weights are identical.
4. Correction and regional standalone discrimination do not guarantee useful complement. Their lower standalone mAP is also insufficient to prove that all added evidence is useless. The report includes same-model paired repairs, new errors and AP contributions so net benefit is assessed directly.

| Dataset | Condition | Repair AP contribution | New-error contribution | Kept-correct contribution | Kept-wrong contribution | F−G total |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | uniform | +0.053314 | -0.037725 | +0.421672 | +0.054124 | +0.491385 |
| RGBNT201 | ambiguity | +0.054045 | -0.058370 | +0.328295 | +0.009266 | +0.333237 |
| RGBNT100 | uniform | +0.051114 | -0.185625 | -0.499958 | -0.045216 | -0.679685 |
| RGBNT100 | ambiguity | +0.021697 | -0.141626 | -0.466941 | -0.032664 | -0.619533 |
| MSVR310 | uniform | +0.000000 | -0.004123 | -0.004547 | +0.008318 | -0.000352 |
| MSVR310 | ambiguity | +0.000000 | -0.004123 | -0.000691 | +0.008445 | +0.003630 |

Research direction: study whether added evidence changes actual final identity relations usefully. External methods address one demonstrated gap at a time. These results do not justify automatically launching late interaction, distillation, a new gate, gain/seed searches, or another training queue. Current training and NN diagnosis are complete; retain the original bests, first strict results, pinned diagnostic distances and historical failures.

Scope: single seed42, consumed development benchmarks. The pairwise R/J scores require pairwise local comparison and are not the1536 deployment result or a new main result. Source and terminal receipts establish execution checks; raw image labels and distance sorting have not been independently re-audited by this summarizer. A fresh same-family provisional integrity review is separate.
