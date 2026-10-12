# Four original actual-region results — draft claim scope

Only the first four original906 endpoints are complete: RGBNT201 and MSVR310, uniform and ambiguity, one seed42, each real8 M0 then fresh50 then first strict same-mAP-best evaluation. RGBNT100 uniform was still3/50 in the actual22:03 snapshot, its ambiguity endpoint unattempted. Do not describe six ends as complete.

| Dataset | Condition | Best epoch | mAP | R1 | Delta mAP vs raw semantic | Delta mAP vs independent global |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | uniform | 8 | 74.7880 | 78.9474 | +0.4132 | +0.4914 |
| RGBNT201 | ambiguity | 8 | 74.6299 | 79.0670 | +0.2550 | +0.3332 |
| MSVR310 | uniform | 38 | 50.5418 | 67.8511 | -0.0004 | -0.0004 |
| MSVR310 | ambiguity | 38 | 50.5458 | 67.8511 | +0.0035 | +0.0036 |

These are the original static1536-vector results, not regional late-interaction scores. All R1/R5/R10 follow the same weight. Complete legal query/gallery and unmodified camera/scene filtering are used. Controls are the pinned raw-semantic824 and independently trained shared-adapter global794. Fixed-model per-query analysis is not training-seed uncertainty.

201uniform vs independent global repairs6/new-errors6; total mAP gain is +0.4914, with +0.4217 contributed by654 queries whose first rank remained correct. Ambiguity repairs8/new-errors7 and +0.3332 mAP; it is not a larger mAP gain than uniform. MSVR repairs0/new-errors1 vs independent global for both, with one-query R1 decline of0.1692 points. MSVR differences near zero are not evidence of stable efficacy or statistical insignificance.

201auxiliary loss first→last uniform0.577491→0.002347, ambiguity0.583456→0.006464; full50 eligible exposure57416/169536 and11089680 legal triplets per condition. Best→last mAP declines3.151862 and2.669513. MSVR first→last0.655440→0.266886 and0.664683→0.304410, eligible21708/45184 and2793000 triplets. These establish actual task/support activity descriptions, not a unique reason for limited final gains. Exposure counts are not unique identities/samples.

No same-model global/residual/regional decomposition is asserted here. No stronger local teacher, compression-loss explanation, background cause, mask correspondence truth, significance, full three-dataset success or new SOTA claim. Historical local full Signal20180.3029/85.1675 andMSVR53.2424/72.4196 are external project references, not re-run in this consumer; neither new condition exceeds them. The original all-three goal remains unmet. Do not rescue failure with gain/seed/extra comparator searches.

Producer wall vs training-loop seconds are separate:201 uniform2428.699669 vs1646.581627; ambiguity2422.959254 vs1645.164808; MSVR1433.092709 vs516.258550 and1430.653969 vs515.309923. Training-loop totals exclude subsequent per-epoch evaluation/save. Original report4 ran22:03–22:04 on CPU with empty CUDA visibility, zero model inference and optimizer updates. Original NN queue remained live on26 physical0/1 in the latest actual snapshot.
