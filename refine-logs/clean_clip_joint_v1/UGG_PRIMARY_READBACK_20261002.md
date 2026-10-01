# UGG-ReID primary-source readback, 2026-10-02

Source-only work while the registered clean-public-CLIP six-endpoint campaign continues. This readback does not change its source, evaluation protocol, selection rule or J1 threshold, and does not launch an author reproduction or N1/N2/N3.

## Published reference and resource boundary

Wan et al., *UGG-ReID: Uncertainty-Guided Graph Model for Multi-Modal Object Re-Identification*, arXiv:2507.04638v1 (7 July 2025). The author README states acceptance at NeurIPS 2025; this readback inspected the arXiv version and author repository, not a separately retrieved version of record.

| Dataset | Author main-table mAP / R1 | R5 / R10 |
|---|---:|---:|
| RGBNT201 | 81.2 / 86.8 | 92.0 / 94.7 |
| RGBNT100 | 88.0 / 98.1 | Not reported in that table |
| MSVR310 | 60.1 / 78.0 | Not reported in that table |

The paper uses CLIP, patch graphs and uncertainty-guided experts. It reports Adam, 40 epochs and one RTX 4090. Its RGBNT201 ablation rises from 72.2 to 81.2 mAP (+9.0). These are author reports, not matched-budget local results. [Primary paper, Tables 1–3 and §4.1](https://arxiv.org/html/2507.04638v1).

## Pinned executable-source inventory

Author repository: [wanxixi11/UGG-ReID](https://github.com/wanxixi11/UGG-ReID/tree/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1), commit `eaf1e8e50d04f34ee3e471440f70d335cc67b2c1`, retrieved 02:49:56 CST. The recursive Git tree was not truncated. Fourteen selected source files (108,922 bytes) were fetched and matched against their Git blob SHA-1; SHA-256 values are retained in `SOURCE_CHECK.json`.

The tree contains actual model, loader, optimizer and scorer implementations. Its only YAML configuration is `configs/RGBNT201/UGG.yml`; there are no checked-in `.pth`, `.pt` or `.ckpt` weights and no license file in the inspected tree. The README names vehicle configurations absent from that tree. Those commands are not a completed vehicle reproduction recipe. No third-party implementation was copied into TriFusion.

## Evaluation and training facts

- `data/datasets/msvr310.py` uses `query3` and all `bounding_box_test` identity directories. It reads identity, view and time/scene fields from filenames. Its normalized text equals the current local TriFusion parser. RGBNT100 also matches; RGBNT201's `__init__` and `_process_dir` match at AST level despite file-text differences. This is a source comparison, not a runtime witness of the author's original file lists or gallery counts.
- `utils/metrics.py::eval_func_msrv` excludes same identity **and same scene/time label**. The camera-only alternative is commented out. `R1_mAP.compute` normalizes features and computes squared Euclidean distances. `engine/processor.py` selects this scorer for MSVR310 and uses the camera scorer otherwise. The general scorer supports reranking, but the published RGBNT201 YAML sets it to `no`.
- `configs/RGBNT201/UGG.yml` specifies CLIP ViT-B/16, GPGR and UGMoE enabled, global/local fusion, 256×128 images, batch 64, K=8, 40 epochs, warmup 10 and Adam at 3.5e-4. `solver/make_optimizer.py` assigns 5e-6 to trainable non-adapter parameters whose name contains `base` under its unfrozen-CLIP condition. This source condition has not been checked by constructing the author's optimizer.
- `modeling/make_model_ugg.py::UGG.forward(return_pattern=3)` concatenates the three-modal original global representation and the three-modal fusion representation. With the inspected CLIP width 512, the declared output is 3072 dimensions. A matched 1536-dimensional adaptation would differ from this author configuration.
- `engine/processor.py` records all CMC values at a single highest-mAP epoch and uses `>=` for ties. Its best-checkpoint save call is commented out. The YAML's checkpoint period is 50 but its run length is 40, so the inspected ordinary periodic-save branch does not save a formal terminal within that run. `test_net.py` loads a hard-coded author-local RGBNT201 weight path instead of the YAML `TEST.WEIGHT`.
- `GeneralFusion.forward` calls Gaussian reparameterization before the `self.training` branch; the helper uses `torch.randn_like`. Thus the graph path includes a random draw during evaluation. Its scale is a learned parameter initialized to zero; this source-only readback does not establish its trained value or the realized metric variance. Do not silently replace this path with the mean and call it an exact author reproduction.

Code evidence: [MSVR parser](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/data/datasets/msvr310.py), [metrics](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/utils/metrics.py), [processor](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/engine/processor.py), [configuration](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/configs/RGBNT201/UGG.yml), [graph/fusion implementation](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/modeling/GPGR_UGMOE/GPGR_UGMoE.py).

## Consequence for this project

UGG-ReID adds a concrete CLIP paper reference on MSVR310 and a more substantial source package than a README-only comparator. Its filtering code supports the same time-based exclusion definition, but original file lists, vehicle settings, author weights, evaluation randomness and checkpoint recovery remain unqualified for a local matched-protocol result. It is not yet a runnable, verified SOTA target on our server.

N1 remains the next candidate after the current full-six report and fresh audit. UGG's graph and routing are not proposed as a new module or a restart of the previously failed Router line. MDReID, DeMo and MODAL remain the required direct N2 neighbors; N3 still requires actual batch-level legal cross-environment support beyond the existing dataset-level counts.
