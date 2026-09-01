# Result-to-Claim Reviewer Prompt

Intended claim: the shared-CLIP CNN+Transformer+Mamba architecture with staged exchange and quality-aware routing reaches the registered RGBNT201 target `85.3 mAP / 87.9 Rank-1`, supports a SOTA-level main result, and may proceed to ablations.

Evidence supplied to the independent reviewer:

- RGBNT201 `postfreeze-final`, seed 42, epoch 60, one official evaluation.
- fused `59.14784166853979 mAP / 63.27751196172249 Rank-1`.
- CNN `59.156120842932026 / 63.75598086124402`.
- Transformer `59.12187416192253 / 62.67942583732058`.
- Mamba `58.8748068990539 / 62.44019138755981`.
- Fused target gap: `-26.15215833146021 mAP / -24.62248803827751 Rank-1`.
- One dataset and one seed; no baseline reproduction or multi-seed by user instruction.
- Query/gallery symmetry claim failed; repair was audit-only with no optimizer step or official re-evaluation.

Requested fields: claim support, supported/unsupported statements, missing evidence, revised claim, next experiments under user constraints, and confidence.
