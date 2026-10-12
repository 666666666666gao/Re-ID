# Saved query comparison: ambiguity minus uniform

Four completed seed42 full50 mAP-best endpoints, two datasets. Official benchmark consumed in development. Saved GT-scored query rows only, not independent binary/image replay. No NN/optimizer/GPU/SSH/report rerun or new training. Global rank1 strata describe benchmark difficulty; they are not the training ambiguity weights or a causal partition. No claims about RGBNT100, full cohort, statistical significance, teacher, inference comparator, or unique failure cause.

| Dataset | Best U / A | Delta mAP | Delta R1 | AP improved / worsened | Worst 5% AP delta |
|---|---:|---:|---:|---:|---:|
| RGBNT201 | 8 / 8 | -0.15814806 | +0.11961722 | 222 / 271 | -5.25770302 |
| MSVR310 | 38 / 38 | +0.00398260 | +0.00000000 | 32 / 29 | -0.01099949 |

## RGBNT201

| Global stratum | Queries | Within-stratum delta AP | Contribution to overall delta mAP |
|---|---:|---:|---:|---:|
| global_rank1_correct | 660 | -0.14442777 | -0.11402192 |
| global_rank1_wrong | 176 | -0.20959915 | -0.04412614 |

## MSVR310

| Global stratum | Queries | Within-stratum delta AP | Contribution to overall delta mAP |
|---|---:|---:|---:|---:|
| global_rank1_correct | 402 | +0.00566876 | +0.00385591 |
| global_rank1_wrong | 189 | +0.00039617 | +0.00012669 |

This descriptive comparison does not establish why weighting changed the outcomes. Any final decision awaits the original RGBNT100 endpoints; no rescue tuning or additional seeds are triggered.
