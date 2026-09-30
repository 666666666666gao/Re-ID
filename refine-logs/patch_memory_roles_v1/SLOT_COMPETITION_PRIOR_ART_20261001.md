# Slot competition: primary-source boundary, 2026-10-01

This is a mechanism review of the four accepted Patch-memory endpoints, not a new experiment registration or evidence that a proposed modification works. RGBNT100 has not yet been accepted. The existing six-end training contract, sources, checkpoint selection and gate remain unchanged.

## What the current code and measurements establish

`PatchMemoryRoles.sample_context()` forms scores of shape batch × modality ×16slots ×128Patches. `softmax(dim=-1)` normalizes Patches independently for each slot. Local mode masks each slot to its nearest nine Patches; full mode exposes all128. Global context conditions queries, not values. Both modes retain the same Q/K/V/output parameters and initialization.

Complete-query measurements before anchor addition and role processing found selected-content cosine means of0.6265/0.9864 (local/full, RGBNT201) and0.6554/0.9857 (MSVR310). These establish similar sampled content in full mode. They do not establish identical final roles, physical-part correspondence, or a unique cause of the paired negative mAP changes. Local masking itself limits attention overlap.

## Primary sources and verified implementations

| Canonical ID | Source | Relevant mechanism | Evidence boundary |
|---|---|---|---|
| arxiv:2006.15055 | [Object-Centric Learning with Slot Attention, NeurIPS2020](https://proceedings.nips.cc/paper_files/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf), Algorithm1 | Normalize over slots for each input, then over inputs for a slot weighted mean; iterative GRU and residual MLP | Object discovery and set prediction; not a three-spectrum ReID result |
| arxiv:2209.14860 | [Bridging the Gap to Real-World Object-Centric Learning, ICLR2023](https://arxiv.org/html/2209.14860v2) (DINOSAUR) | Group pretrained features with Slot Attention and reconstruct feature targets; its frozen-ViT variant uses frozen input/target features | Feature reconstruction is part of this method, not a universal requirement for slot competition. Other encoder variants also exist; no RGBNT/MSVR evidence |
| arxiv:2409.13475 | [PLOT, ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03209.pdf), Sec.3.2–3.5; [supplement](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03209-supp.pdf) | Competitive part slots for text/image retrieval, shared initial slots, part identity/retrieval losses and token-feature reconstruction; text-conditioned part similarity aggregation | Direct retrieval precedent. Experiments are CUHK-PEDES/ICFG-PEDES/RSTPReid, not RGB/NIR/TIR. Shared slot indices alone are not a mathematical correspondence guarantee |

Author code is pinned to the revisions actually inspected, not described as historical paper releases:

- Slot Attention: `google-research/google-research@d36068b845da4c2b24927fee2cea1e6ef98dadda`, [SlotAttention.call](https://github.com/google-research/google-research/blob/d36068b845da4c2b24927fee2cea1e6ef98dadda/slot_attention/model.py). With input×slot logits, its last-axis softmax is slot competition; this differs from our slot×Patch logits. The locally downloaded source SHA is96c2b12d8b28c22fd2605eccf9027bda304f38bb9332f8c1620898f471c67ec4; source remains outside the project repository.
- DINOSAUR: `amazon-science/object-centric-learning-framework@0a97292cb0dbb173777d8137ac0032957b4f0a6c`, [SlotAttention.step](https://github.com/amazon-science/object-centric-learning-framework/blob/0a97292cb0dbb173777d8137ac0032957b4f0a6c/ocl/perceptual_grouping.py), [frozen feature extractor](https://github.com/amazon-science/object-centric-learning-framework/blob/0a97292cb0dbb173777d8137ac0032957b4f0a6c/ocl/feature_extractors/timm.py), [feature-reconstruction configuration](https://github.com/amazon-science/object-centric-learning-framework/blob/0a97292cb0dbb173777d8137ac0032957b4f0a6c/configs/experiment/projects/bridging/_base_feature_recon.yaml). Grouping attention and decoder mask normalization are different operations.
- PLOT: `jicheol93/PLOT@cf5690165b1063b8dc0b6302641d78a7b65d059c`, [SlotAttention_LearnableSlots.forward](https://github.com/jicheol93/PLOT/blob/cf5690165b1063b8dc0b6302641d78a7b65d059c/model/slot_model.py), [PLOT model](https://github.com/jicheol93/PLOT/blob/cf5690165b1063b8dc0b6302641d78a7b65d059c/model/build_local_slot_detachRecon.py). B×slot×token scores use `softmax(dim=1)` followed by token normalization. The single initial `self.slots` is shared, while image/text attention modules are independent. Reconstruction targets are detached. HEAD differs from the paper in PartID concatenation, additional local objectives, inference scaling and an RSTP learning-rate script; do not call it a verbatim paper implementation.

## What can and cannot follow

Slot competition is an established alternative to independent query attention, including a direct person-retrieval precedent. It cannot be claimed as our original mechanism. Changing a normalization axis alone does not reproduce the iterative models and their supervision, guarantee diverse values when input features are similar, or demonstrate useful identity evidence. Attention diversity and sampled-content diversity must also be distinguished from retrieval improvement.

No proposed competitive model was implemented, registered or trained during this review. First complete all six current endpoints, reproduce RGBNT100 diagnostics with the unchanged helper, and judge the original gate. Any later intervention requires its own matched control and complete training; no threshold, seed, auxiliary loss or architecture change is justified by a particular consumed official query.

Retrieval/extraction used primary paper and author-code sources, mechanically deduplicated by the three arXiv IDs. Fresh read-only source shards supported extraction; they are not independent scientific approval. No local paper library or arXiv helper was available for this focused lookup. No third-party code was executed and no environment was changed.
