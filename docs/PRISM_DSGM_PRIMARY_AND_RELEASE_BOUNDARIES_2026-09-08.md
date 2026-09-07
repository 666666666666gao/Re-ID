# PRISM / DSGM: primary tables and release boundaries

This closes the primary-table gap recorded in SOTA_PRIMARY_REFRESH_2026-09-06.md. It does not change the active MSVR freshness experiment or its gates. Scores below are author-preprint mAP / Rank-1 percentages, not local reproductions or internal Q1 results.

| Method | RGBNT201 | RGBNT100 | MSVR310 | Verified primary location |
|---|---:|---:|---:|---|
| PRISM | 80.5 / 84.0 | 86.1 / 97.8 | 47.6 / 64.8 | Tables I–II, PDF pp.6–7 |
| DSGM | 82.6 / 87.0 | 89.4 / 98.2 | 64.6 / 76.0 | Tables I–II, PDF p.7 |

Both HTML tables and separately downloaded/rendered PDF pages were inspected: [PRISM author preprint v1](https://arxiv.org/html/2607.23451v1), [DSGM author preprint v1](https://arxiv.org/html/2607.29207v1). These are the verified versions; equivalence to a publisher's final version was not established here.

## Resources and training described in the papers

PRISM uses CLIP plus offline semantic masks: OpenPifPaf-derived human masks and SAM2 vehicle masks. Prompt-S6 conditions state-space input/output projections using auxiliary tokens; semantic pruning precedes progressive fusion. Its implementation section specifies B64/K8 for RGBNT201/MSVR310 and B128/K16 for RGBNT100; training lasts60/60/50 epochs respectively. The visual encoder has learning rate5e-6, other modules3.5e-4. These conditions differ from frozen-Signal residual-only training. [PRISM sections III-A/B and IV-A2](https://arxiv.org/html/2607.23451v1#S4.SS1.SSS2).

DSGM uses GPT-4o descriptions and SAM2 masks alongside CLIP. Text injection, mask-guided local/global interaction and hierarchical expert fusion therefore use additional semantic resources. Its dataset section describes RGBNT201 as141 training/30 validation/30 test identities; its training section specifies60 epochs there,50 on both vehicle datasets, B64/K8 for RGBNT201/MSVR310 and B128/K16 for RGBNT100. That section prints module learning rate3.5e-6. These statements require the release qualifications below. [DSGM sections IV-A/B](https://arxiv.org/html/2607.29207v1#S4.SS2).

## What the pinned released code actually says

PRISM author HEAD: `0067f6d895c522afa2c4f30515b33bc4300fe680`; DSGM: `6566f78636440c70c85700a45bbf2f48f18260b1`. Both selected LICENSE files are MIT. Thirty-one commit-pinned source/config/license files were fetched and hashed; this is selected static inspection, not a complete runnable-release audit.

- PRISM's [MSVR310 YAML](https://github.com/zw-absin/PRISM/blob/0067f6d895c522afa2c4f30515b33bc4300fe680/configs/MSVR310/PRISM.yml#L24) sets `NUM_INSTANCE: 4`, while the paper states8. B64/60epochs agree. The [RGBNT201 loader](https://github.com/zw-absin/PRISM/blob/0067f6d895c522afa2c4f30515b33bc4300fe680/data/datasets/RGBNT201.py#L26) reads `train_171`, with query/gallery from `test`.
- DSGM's [RGBNT201 YAML](https://github.com/zw-absin/DSGM/blob/6566f78636440c70c85700a45bbf2f48f18260b1/configs/RGBNT201/DSGM.yml#L29) sets `BASE_LR: 0.00035` and50epochs. A YAML base rate alone does not prove every optimizer parameter group's effective rate; no optimizer execution was performed.
- DSGM's [factory](https://github.com/zw-absin/DSGM/blob/6566f78636440c70c85700a45bbf2f48f18260b1/data/datasets/make_dataloader.py#L23) selects `RGBNT201_Text`; that [actual class](https://github.com/zw-absin/DSGM/blob/6566f78636440c70c85700a45bbf2f48f18260b1/data/datasets/RGBNT201_Text.py#L32) also reads `train_171`. A directory name does not verify its actual identity count. This prevents inferring the main table's actual training population solely from the paper's141/30/30 description.
- DSGM README commands name `train_net.py`/`test_net.py`, whereas its pinned tree has `train.py`/`test.py`. This is an entry-point documentation mismatch; author code was not run or modified.

## Project implications and evidence

Keep both methods in the resource-qualified literature inventory. Their scores do not replace existing stronger references or prove a complete leaderboard. Additional semantic resources, paper/release discrepancies and unavailable run receipts remain explicit; no internal TriFusion score is subtracted from these reports. Any future reproduction needs a declared contract and actual loader/optimizer/evaluator verification first. No masks, weights or new method were installed into the current experiment.

The [verification receipt](../evidence/prism_dsgm_primary_20260908/primary_verification.json) records both PDF hashes, pages, all31 code-file URLs/hashes, and the retrieval failures. GitHub tree API hit403; partial-clone checkout also failed. Commit-pinned raw text retrieval subsequently completed. These were local literature-access issues, unrelated to remote training.

The running source freshness experiment remains execution-bound toab67d4c/configf3a0634. Continue allsix endpoints and fullCPU before scientific interpretation or the next intervention. WholeGoal active/unmet.
