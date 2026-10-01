# Pinned MDReID readback for next evidence interpretation

Readback date2026-10-01; author main3525ac2da1a2a90a5a160c930fac674b4f226f6c. No author model execution or experiment source changes.

- Actual three dataset parsers equal current TriFusion dataset parsers after LF/CRLF normalization only. RGBNT201 uses train_171 and test for both query/gallery; RGBNT100 uses rgbir bounding_box_train/query/bounding_box_test; MSVR310 uses bounding_box_train/query3/bounding_box_test. Exact paper-result image-list hashes remain unavailable, so compatible parser code does not prove identical historical paper inputs.
- MSVR metric active removal is same identity and same scene/time-segment, not camera. General metric uses same identity and same camera. Current official code scoring retains complete gallery; no filtering all distractor-only identities.
- CLIP visual adds one image token when ADD_SHARE; class_token and extra image_token separately represent the two streams. MDReID.make_model forms modality-specific globals and shared-token features; combo includes original and shared evidence. This is modality-specific/shared content learning, not the current private-adapter-writeback switch.
- EnhancedTripletLoss detaches specific/shared comparison features and uses max hardest-positive distance across batch and min hardest-negative distance across batch. Normalization defaults false; engine invokes it without override. Thus it is not a per-query mAP gain or same-positive/same-negative margin comparison. With unnormalized features, scale participates; simply copying it into normalized1536 additive embeddings changes the objective.
- Source RGBNT201 config: CLIP ViT-B/16, camera SIE, B64/K8, Adam, baseLR.00035, warmup10,50epochs,seed1555, CE weight.25/triplet1, separate/shared/combo supervision. Current panel uses warmed ReID seed42/AdamW/warmup5/its existing supervision. These budget/init/loss differences must be separated from claimed structural effect.

Sources pinned to the author commit:
https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/data/datasets/RGBNT201.py
https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/data/datasets/msvr310.py
https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/utils/metrics.py
https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/modeling/clip/model.py
https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/modeling/make_model.py
https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/layers/triplet_loss.py
https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/engine/processor.py
https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/configs/RGBNT201/MDReID.yml

No novelty claim from using shared/private tokens or KDL. Current nine-end comparison must complete, then interpret paired repair/new-error/cost evidence before registering any successor. Do not retune current campaign from these external settings.
