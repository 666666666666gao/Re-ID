# Frozen visual initialization comparison

Status: PREPARED; no M0 or training has run under this plan.

The complete global-token panel has nine accepted endpoints and fails its original advancement rule. All six candidate-control mAP differences are below 0.06 percentage points. Shared-global to fused gains remain small, and all nine full50 trajectories deteriorate after their selected best despite lower training loss. These observations do not uniquely identify the representation source as the cause.

One hypothesis: the dataset-finetuned, then frozen visual backbone constrains the transferable information available to the new adapters and roles. Test changing only the 152 public CLIP visual tensors to their public pretrained values; leave the trained camera embedding and all other baseline state unchanged. This is an initialization control, not a new module or a wholly public-CLIP-only system. The lineage difference was already documented in section 41.666; the controlled performance comparison has not been run.

## Six fresh endpoints

RGBNT201, RGBNT100 and MSVR310 each receive `reid_visual` and `public_visual`, seed42, fresh role-stage full50. Both use the existing STATIC global-token architecture, full128 patch memory, independent FP32 attention, M1 at blocks4/8/12, CNN–Transformer–production Mamba, 1536D regional readout, context queries, no auxiliary ID and no M3. Static context, parameter count, adapters, roles, classifier initialization, original loader, optimizer, learning-rate schedule, warmup, CE smoothing and Triplet margin remain unchanged. Compare every trainable initial tensor bitwise; frozen initial model hashes are expected to differ. No loss, sampling or readout changes.

`public_visual` is a copy of the saved pure-ReID baseline checkpoint with exactly its `clip_vision_encoder.base.*` tensors replaced from the existing official `ViT-B-16.pt`. There are exactly 152 such tensors with identical names. Only positional embeddings need the existing author convention: keep the CLS position and bilinearly resize the 14x14 patch grid to 16x8 or 8x16. Cast each public tensor to the saved tensor dtype. All seven/nineteen other checkpoint entries, including trained camera embeddings, remain bitwise equal. Preserve the original three pure baseline checkpoints. No SIM/AlignM, text, segmentation, external data, new pretraining or environment changes.

## Execution and reporting

Run actual eight-batch production M0 before each fresh full50 and strict evaluation. Frozen baseline unchanged during training, finite real gradients for every trainable tensor, strict reload and full query/gallery checks remain mandatory. No auto-retry, partial endpoint, seed search or early cancellation based on retrieval scores. A genuine failed stage stops pending launches and drains already active work with all failure receipts preserved. Queue uses actual free GPUs without preemption and a 240-second harvest interval.

Use exactly the existing official protocol JSONs and complete gallery: RGBNT201/RGBNT100 same-ID same-camera filtering, MSVR310 same-ID same-time filtering; retain gallery-only identities. Per endpoint, select one official fused-mAP-best checkpoint after completing all50 epochs; all metrics come from that checkpoint. Official development selection is disclosed and is not an untouched-test estimate. Report fused, jointly trained global and total correction without calling either a separately trained control or pure-local evidence.

All six accepted endpoints are required before deciding the hypothesis. Advancement requires public-minus-ReID mAP >0 and Rank-1 >=0 on all three datasets, and >=0.5 mAP on RGBNT201 and MSVR310. A failed rule remains failed; do not retune, change the threshold or add seeds to rescue it. This gate is not the baseline/SOTA goal. Even a passing initialization effect is not algorithmic novelty or a stability claim.

Record the public weight SHA, original and reset checkpoint SHAs, all152 changed/copied tensors, positional convention, all unchanged nonvisual states, full runtime closure, all18 real stage exits, full50 logs, same-trainable initialization witnesses, complete metrics, negative flips, identity AP distribution, actual time and disk usage. Upstream pure-ReID training and retained camera state are explicitly additional lineage/cost. Inputs and checkpoints stay remote; transfer only code, text, JSON and generated plots.
