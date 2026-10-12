# Fixed evidence-use diagnostic, 2026-10-12

Question: at the existing raw-semantic selected bests, does the way current residual and region evidence enter similarity limit its useful contribution? This is a no-training diagnostic before adding a comparator/module. It is not a compression-cause hypothesis accepted as fact.

Inputs: all three sealed824 raw-semantic bests, unchanged full original query/gallery, camera exclusions or MSVR scene exclusions, selected epochs8/38/5. Reuse public initialization and existing two-card architecture, seed42. Do not select another epoch, train a classifier, tune mix/gain/topk/margin/temperature, rerun closed reports or use new seeds. Protected bests stay in place.

Perform one forward per query/gallery record, keeping actual global g, readout residual c, original fused f, and final CNN/Transformer/Mamba region tensors that the actual readout consumes. The region tensors are not auxiliary keys, geometrically labeled parts or visibility GT. Preserve model state and no parameter gradients/optimizer updates. Reproduce each original selected fused mAP/CMC within1e-5 before claiming any comparator result.

Seven fixed comparisons, all full-gallery, none chosen by outcome:

1. Current global cosine.
2. Current additive fused cosine, with exact existing gamma.
3. Residual standalone cosine (diagnosis only).
4. Exact same-residual block cosine of [g; gamma c]. This changes both GC cross terms and normalization; it is a3072-vector diagnostic, not an original contribution or1536 static result.
5. Same-slot mean cosine of unit final region vectors, separately within each role/modality. Same slot is an index control, not ground-truth part correspondence.
6. Symmetric mean MaxSim within each of the nine role/modality groups: average source-to-target and target-to-source maxima, then equal group mean. Many-to-one; no cross-role or cross-modality coordinate matching, learned masks or visibility inference. All16 final output slots participate, with finite nonzero norm asserted.
7. A fixed-energy regional score replacing only residual cosine in comparison4 by comparison6. Use existing gamma squared times norm(c_q)norm(c_g) and the unchanged block denominator; no additional fusion coefficient. This uses the same frozen extractor with a different retained representation, not the exact same1536 residual.

Record all metrics and per-query/identity changes, first-rank repair/new-error/kept-correct/kept-wrong total mAP contributions, query AP-delta percentiles and worst5% mean. Also measure actual region direction dispersion. Region separability and scalar score difference are distinct. No low score/entropy is labeled physical absence; unknown/conflict support is untested.

Primary interpretation is same-residual block versus additive; local-only and fixed-energy region diagnostics then test the usefulness of retaining actual regions. Better local comparison would justify a future comparison/learning study under identity-isolated validation; it does not establish a stronger trained method, novelty or a unique compression cause. If these fixed comparisons do not improve net retrieval, do not automatically add a matcher/gate; return to evidence learning. No automatic rollout/training follows.

Exact outputs/costs: one fixed diagnostic invocation per dataset, zero optimizer updates; original model/checkpoint/arrays remain immutable. Save g/c/f and regions once per split to avoid future encoding reruns, plus textual metrics/query summaries. No dense score matrices saved. FP32 cache estimate1,254,113,280B plus64MiB serialization/text allowance, plus original2GiB reserve: storage gate is the measured protocol-derived sum, not lowered after failure. All model/arrays stay on26. Local matching chunks16queries×128gallery bound temporary memory and do not select candidates. Estimate runtime from the first completed endpoint before scheduling observations. The MSVR author scorer writes re.txt in its working directory; scoring runs in the new endpoint output directory, with model/input construction unchanged and all sealed paths absolute.

Resources: server26 physicalGPU0/1 only, one pair, existing warm conda, no installs/preemption/power/temperature actions or GPU2/3/25 queries. The diagnostic coordinate process checks the two allowed cards are free before NN admission; occupied cards stop admission, never preempt. Source/plan review and exact dependency seal precede the one launch. Completed source review is not NN verification. Missing/failed endpoints remain missing, no fallback or hidden retry.

Closest source paths and paper/code differences are in PRIMARY_EVIDENCE_COMPARISON_902.md. ColBERT's original asymmetric sum and zero-mask path are not transplanted. Value-Aware RL/SelectiveNet inform only purpose and risk boundaries; no their training objective or guarantee is imported. Official sets have been consumed for development; these diagnostics add exposure and are not untouched confirmation.
