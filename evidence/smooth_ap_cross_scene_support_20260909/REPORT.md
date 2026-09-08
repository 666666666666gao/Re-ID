# Registered source queue: cross-scene positive support

Executor descriptive replay, 2026-09-09. No new model, forward, optimizer, image or current Q1 score read. The pinned source protocol/metadata/Smooth-AP config hashes and all 780 T0 queue counts/ages were checked. One shared schedule is counted once, not doubled for paired endpoints. Complete per-fold, per-identity and all 780 per-batch rows are in the accompanying JSON. Raw labels/metadata/T0 were already archived; the JSON binds them by SHA.

| Post-warmup (steps 66–260) | fold0 | fold1 | fold2 | total |
|---|---:|---:|---:|---:|
| Anchor exposures | 12480 | 12480 | 12480 | 37440 |
| Have cross-scene positive in pool | 5144 | 5176 | 5056 | 15376 |
| Globally eligible, missing from pool | 56 | 48 | 32 | 136 |
| Source identity has only one scene | 7280 | 7256 | 7392 | 21928 |
| Cross-scene positive positions | 27864 | 28716 | 28334 | 84914 |
| Same-scene positive positions | 66880 | 67204 | 66250 | 200334 |
| Batches with zero cross-scene anchors | 1 | 1 | 2 | 4 |

Cross-scene eligible anchors account for 41.0684% of exposures; cross-scene positive positions account for 29.7685% of all positive positions. Among globally eligible exposures, only 0.8767% lack a cross-scene positive in the pool. Historical candidates rescue zero additional anchor-eligibility events in this registered schedule; this does not mean history adds no positive positions or difficult examples. Every source fold has 40 cross-scene identities; no model capability is inferred from identity/scene membership.

Eligible-anchor count histogram across 585 post-warmup batches: 0:4, 8:54, 16:102, 24:172, 32:147, 40:79, 48:24, 56:3. Zero-eligible steps are fold0/180, fold1/221, fold2/133 and232. These are true labels under the candidate cross-scene rule, not missing positives in the currently running all-identity Smooth-AP, which retains current same-identity positive views.

Conditional implication: a later cross-scene ranking term would not cover all 64 anchors. For a fixed batch and fixed nonempty eligible set of size n, mean over eligible anchors equals (64/n) times a zero-filled mean over all 64. Here that normalization multiplier ranges from 8/7 to8 for nonempty batches; the aggregate exposure ratio 37440/15376 is not the actual per-step gradient multiplier. Choosing either convention changes the objective definition and must be explicit before a new paired experiment. No convention is selected or registered by this diagnostic.

The four empty batches are direct evidence requiring an explicit empty-eligible-set definition if that future loss is implemented. Same-identity same-scene positions must be excluded from that ranking term, never converted to negatives; those records and all other identities retain other supervision and candidate distractor roles. More than half the anchor exposures are globally single-scene, so merely refreshing or enlarging the queue cannot provide a real other-scene positive for them.

Limits: repeated positions, not independent images; finite registered queue, not an unrestricted gallery; label support, not loss/gradient contribution or retrieval gain. No evidence here that easy same-scene positives dominate the actual Smooth-AP parameter gradient. No official test reading, seed sweep, training-mask change, added loss or promotion. Finish the existing full Q1/CPU and source-gradient/ranking analysis before selecting the next single hypothesis.
