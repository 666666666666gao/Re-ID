> **TriFusion research fork.** This branch adds a shared-CLIP
> CNN + Transformer + Mamba collaborative RGB–NIR–TIR ReID system, together
> with identity-disjoint reliability calibration, fixed-endpoint evaluation,
> recovery receipts, and fail-closed evidence verification. See the
> [current complete handoff](docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md)
> for the exact status, commands, metrics, claim boundaries, and document index.
> The frozen seed-42 final result is **59.1478 mAP / 63.2775 Rank-1**
> (fused), below the registered target; see the
> [result report](results/TRIFUSION_RGBNT201_FINAL_SEED42_2026-09-01.md) and
> [integrity audit](EXPERIMENT_AUDIT.md). This result does not support a SOTA
> claim and does not pass the project gate for ablations.
> The later V4 held-out-dev run also completed 60/60 epochs: fused
> **43.4031 mAP / 42.7879 Rank-1**, below Mamba **44.0659 / 43.5152** and the
> frozen 65 mAP dev gate. The exact Signal 3072D direct+SIM baseline with
> camera SIE subsequently reached **58.0109 mAP / 57.4545 Rank-1** on the
> project's fixed 141-fit/30-dev split. A fused successor is admitted only if
> it beats that same-checkpoint baseline. Signal's published **80.3/85.2** is
> an official-test result and remains upstream-only here.
> The Signal-preserving V5 core and runner pass exact checkpoint parity,
> real B32/K4 capacity, and 100-step fixed-batch overfit gates on the RTX3090.
> Its complete seed-42 60-epoch held-out-dev run selected epoch 51: the exact
> Signal baseline is **58.0109 mAP / 57.4545 Rank-1**, while fused is
> **58.0168 / 57.4545** and CNN is **58.0181 / 57.4545**. Fused therefore
> fails the frozen requirement to beat every expert and reach 65 mAP; no
> official test or ablation was run. A read-only checkpoint diagnostic shows
> the fused distance matrix is effectively unchanged from baseline
> (correlation `1.0`, Top-10 overlap `99.988%`).
> The resulting main-only V6 correction passed all readiness gates and completed
> its single seed-42 60-epoch held-out-dev run. The selected epoch 8 gives fused
> **58.7321 mAP / 57.5758 Rank-1**, above the exact Signal baseline
> **58.0109 / 57.4545** but below CNN **59.1022 / 59.6364** and the frozen
> 65 mAP gate. A read-only diagnostic confirms that V6 now changes retrieval
> geometry, but its high-entropy router gives the strongest CNN expert the
> lowest weight. The V6 claim gate therefore fails; official test, ablations,
> and multiple seeds remain closed.
> A read-only ground-truth Oracle then reaches **63.6089 mAP**, **4.5067**
> points above the strongest fixed V6 branch, with positive leave-one-out
> contribution from CNN, Transformer, and Mamba. The main-only V7 correction
> therefore retains all three residual experts and fixes synchronized triplet
> geometry, hierarchical modality/expert routing, matched-token residuals and
> bounded sample energy. Exact Signal parity, real B64/K8 two-view capacity
> (222/222 gradients, 11.486 GiB peak reserved) and fixed-batch overfit pass.
> Its complete seed-42 60-epoch held-out-dev run selected epoch 1: fused is
> **58.3293 mAP / 57.9394 Rank-1**, above the exact Signal baseline
> **58.0109 / 57.4545** but just below Mamba **58.3476 / 57.8182** and below
> the 65 mAP gate. The joint phase then degrades fused mAP to 57.7550 by epoch
> 60. A read-only diagnostic finds a nearly uniform Router (entropy 0.99791),
> while residual-only Oracle complementarity remains 3.6118 mAP. V7 therefore
> fails the main gate; no SOTA, official-test, ablation or multiple-seed claim
> exists. See the [V7 result report](results/TRIFUSION_RGBNT201_V7_DEV_SEED42_2026-09-02.md).
> A subsequent optimizer-free frozen-router probe rejects the obvious V8
> follow-up. All 571 cross-camera-eligible fit queries select CNN, so an
> identity-disjoint 18-to-3 utility teacher cannot learn expert diversity and
> only matches the 55.27% dev majority policy. Restoring equal residual energy
> raises the deployable frozen result to **59.6188 mAP / 59.1515 Rank-1**, but
> it remains 5.3812 below the dev gate. V8 must therefore improve expert
> representations rather than only retrain the Router; see the
> [probe report](results/TRIFUSION_RGBNT201_V8_FROZEN_ROUTER_PROBE_2026-09-02.md).
> The replacement V8 Phase-A now branches after frozen CLIP block 8 and sends
> CNN, Transformer, and Mamba through the same frozen pretrained tail with
> role-specific residual heads. Exact preflight, real B64/K8 capacity and
> 100-step overfit pass. A 20-epoch final-only probe gives a deployable fixed
> fused **58.0972 mAP / 56.8485 Rank-1**, which is not a main gain, but the
> ground-truth branch Oracle reaches **64.7850 / 65.9394** and every expert has
> unique wins and positive leave-one-out contribution. Independent review is
> `partial` and the integrity audit is `WARN`: this authorizes only one
> frozen-expert, fit-only Router feasibility phase, not HFER, official test,
> ablations, 65 mAP or SOTA. See the
> [Phase-A report](results/TRIFUSION_RGBNT201_V8_EXPERT_FORMATION_PHASE_A_2026-09-02.md)
> and [V8 audit](EXPERIMENT_AUDIT_V8_PHASE_A.md).
> V8 Phase-B then replaced saturated OOF AP labels with continuous identity
> margins and trained only a frozen-expert hierarchical Router from fit-only
> identity-disjoint folds plus controlled single-modality degradation. Its
> single frozen seed-42 dev evaluation gives fused **58.4050 mAP / 59.3939
> Rank-1**, above the exact Signal baseline **58.0109 / 57.4545** and above
> CNN/Transformer/Mamba **57.6071 / 56.3031 / 56.6260 mAP**. This is a narrow
> deployable gain, but it remains **6.5950 mAP below** the frozen 65 gate; the
> OOF learned-vs-fixed advantage is also only `0.000314` mean margin. Phase-B
> is therefore sealed as positive but not promoted: HFER, official test,
> ablations, multiple seeds and SOTA claims remain closed. See the
> [Phase-B report](results/TRIFUSION_RGBNT201_V8_OOF_MARGIN_ROUTER_PHASE_B_2026-09-02.md)
> and [independent audit](EXPERIMENT_AUDIT_V8_PHASE_B.md).
> V9 then tested a distinct representation-level hypothesis: freeze Signal,
> the pretrained-tail experts and Phase-B Router; execute two receiver-specific
> orthogonal peer-relay rounds; and append a triadic interaction embedding while
> preserving the complete Phase-B vector as an exact prefix. Engineering gates
> and the 60-epoch B64/K8 run pass, but the sole final-only dev result is a clear
> negative: fused **56.5339 mAP / 57.2121 Rank-1**, below exact Signal by
> **1.4770 mAP** and below Phase-B by **1.8711 mAP**. It misses the 65 gate by
> **8.4661 mAP**. V9 is sealed without official test, ablations, multiple seeds
> or tuning scans; see the [V9 terminal report](results/TRIFUSION_RGBNT201_V9_DEV_SEED42_2026-09-02.md)
> and [independent audit](EXPERIMENT_AUDIT_V9.md).
> Dataset files, pretrained weights, checkpoints, and remote artifacts are not
> distributed in this repository. The original DeMo project and attribution are
> preserved below.

<p align="center">

  <h1 align="center">DeMo: Decoupled Feature-Based Mixture of Experts for Multi-Modal Object Re-Identification</h1>
  <p align="center">
    <img src="results/logo.png" alt="Description of the image" style="width:54%;">
  <p align="center">

[//]: # (  <p align="center">)

[//]: # (    <img src="https://github.com/924973292/TOP-ReID/assets/89966785/e56e96f1-aa08-47f6-b34d-ae3b7d110060" alt="Description of the image" width="400" height="395">)

[//]: # (  <p align="center">)
  <p align="center">
    <a href="https://scholar.google.com/citations?user=WZvjVLkAAAAJ&hl=zh-CN" rel="external nofollow noopener" target="_blank"><strong>Yuhao Wang</strong></a>
    ·
    <a href="https://dblp.org/pid/51/3710-66.html" rel="external nofollow noopener" target="_blank"><strong>Yang Liu</strong></a>
    ·
    <a href="https://ai.ahu.edu.cn/2022/0407/c19212a283203/page.htm" rel="external nofollow noopener" target="_blank"><strong>Aihua Zheng</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=MfbIbuEAAAAJ&hl=zh-CN" rel="external nofollow noopener" target="_blank"><strong>Pingping Zhang*</strong></a>
  </p>
<p align="center">
    <a href="https://arxiv.org/pdf/2412.10650" rel="external nofollow noopener" target="_blank">AAAI 2025 Paper</a>

<p align="center">
    <img src="results/Overall.png" alt="RGBNT201 Results" style="width:100%;">
</p>

**DeMo** is an advanced multi-modal object Re-Identification (ReID) framework designed to tackle dynamic imaging quality variations across modalities. By employing decoupled features and a novel Attention-Triggered Mixture of Experts (ATMoE), DeMo dynamically balances modality-specific and modality-shared information, enabling robust performance even under missing modality conditions. The framework sets new benchmarks for multi-modal and missing-modality object ReID.

## News
- We released the **DeMo** codebase and paper! 🚀 [Paper](https://arxiv.org/pdf/2412.10650)
- Great news! Our paper has been accepted to **AAAI 2025**! 🎉
---

## Table of Contents
- [Introduction](#introduction)
- [Contributions](#contributions)
- [Results](#results)
- [Visualizations](#visualizations)
- [Reproduction](#reproduction)
- [Citation](#citation)

---

## **Introduction**

Multi-modal object ReID combines the strengths of different modalities (e.g., RGB, NIR, TIR) to achieve robust identification across challenging scenarios. **DeMo** introduces a decoupled approach using Mixture of Experts (MoE) to preserve modality uniqueness and enhance diversity. This is achieved through:
1. **Patch-Integrated Feature Extractor (PIFE)**: Captures multi-granular representations.
2. **Hierarchical Decoupling Module (HDM)**: Separates modality-specific and shared features.
3. **Attention-Triggered Mixture of Experts (ATMoE)**: Dynamically adjusts feature importance with adaptive attention-guided weights.

---

## **Contributions**

- Introduced a decoupled feature-based MoE framework, **DeMo**, addressing dynamic quality changes in multi-modal imaging.
- Developed the **Hierarchical Decoupling Module (HDM)** for enhanced feature diversity and **Attention-Triggered Mixture of Experts (ATMoE)** for context-aware weighting.
- Achieved state-of-the-art performance on RGBNT201, RGBNT100, and MSVR310 benchmarks under both full and missing-modality settings.

---

## **Results**
### Multi-Modal Object ReID
#### Multi-Modal Person ReID [RGBNT201]
<p align="center">
  <img src="results/RGBNT201.png" alt="RGBNT201 Results" style="width:100%;">
</p>

#### Multi-Modal Vehicle ReID [RGBNT100 & MSVR310]
<p align="center">
    <img src="results/RGBNT100_MSVR310.png" alt="RGBNT100 Results" style="width:100%;">
</p>

### Missing-Modality Object ReID
#### Missing-Modality Performance [RGBNT201]
<p align="center">
    <img src="results/RGBNT201_M.png" alt="RGBNT201 Missing-Modality" style="width:100%;">
</p>

#### Missing-Modality Performance [RGBNT100]
<p align="center">
    <img src="results/RGBNT100_M.png" alt="RGBNT100 Missing-Modality" style="width:100%;">
</p>

### Ablation Studies [RGBNT201]
<p align="center">
    <img src="results/Ablation.png" alt="RGBNT201 Ablation" style="width:100%;">
</p>

---

## **Visualizations**

### Feature Distribution (t-SNE)
<p align="center">
    <img src="results/tsne.png" alt="t-SNE" style="width:100%;">
</p>

### Decoupled Features
<p align="center">
    <img src="results/Decoupled.png" alt="Decoupled Features" style="width:100%;">
</p>

### Rank-list Visualization
<p align="center">
    <img src="results/rank-list.png" alt="Rank-list" style="width:100%;">
</p>

---

## **Reproduction**

### Datasets
- **RGBNT201**: [Google Drive](https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH)  
- **RGBNT100**: [Baidu Pan](https://pan.baidu.com/s/1xqqh7N4Lctm3RcUdskG0Ug) (Code: `rjin`)  
- **MSVR310**: [Google Drive](https://drive.google.com/file/d/1IxI-fGiluPO_Ies6YjDHeTEuVYhFdYwD/view?usp=drive_link)

### Pretrained Models
- **ViT-B**: [Baidu Pan](https://pan.baidu.com/s/1YE-24vSo5pv_wHOF-y4sfA)  (Code: `vmfm`)
- **CLIP**: [Baidu Pan](https://pan.baidu.com/s/1YPhaL0YgpI-TQ_pSzXHRKw) (Code: `52fu`)

### Configuration
- RGBNT201: `configs/RGBNT201/DeMo.yml`  
- RGBNT100: `configs/RGBNT100/DeMo.yml`  
- MSVR310: `configs/MSVR310/DeMo.yml`


### Training
```bash
conda create -n DeMo python=3.8.12 -y 
conda activate DeMo
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
cd (your_path)
pip install -r requirements.txt
python train_net.py --config_file configs/RGBNT201/DeMo.yml
```
### Notes
- This repository is based on [MambaPro](https://github.com/924973292/MambaPro). The prompt and adapter tuning on the CLIP backbone are reserved (the corresponding hyperparameters are set to `False`), allowing users to explore them independently.  
- This code provides multi-modal Grad-CAM visualization, multi-modal ranking list generation, and t-SNE visualization tools to facilitate further research.  
- The hyperparameter configuration is designed to ensure compatibility with devices equipped with less than 24GB of memory.   
- Thank you for your attention and interest!

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=924973292/DeMo&type=Date)](https://star-history.com/#924973292/DeMo&Date)

---

## **Citation**

If you find **DeMo** helpful in your research, please consider citing:
```bibtex
@inproceedings{wang2025DeMo,
  title={DeMo: Decoupled Feature-Based Mixture of Experts for Multi-Modal Object Re-Identification},
  author={Wang, Yuhao and Liu, Yang and Zheng, Aihua and Zhang, Pingping},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
