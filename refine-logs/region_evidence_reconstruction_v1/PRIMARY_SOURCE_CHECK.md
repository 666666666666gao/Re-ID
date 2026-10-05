# SAGA-ReID primary-source check — 2026-10-05

Read-only source investigation; no model execution, reproduction, SSH, campaign changes, or next-source selection.

Author code is pinned to **a694589ce4f5d81b729b306ac0b69b1cef15224f**, commit “Added code”, 2026-04-27 08:12:50 UTC. Eleven downloaded files match the pinned tree's Git blob SHA-1; private SHA-256/size evidence, API responses and paper HTML are retained at `C:/Users/gb/.codex_tmp/independent_evidence_draft/reconstruction_primary_check857/`. `visual_modules.py` SHA-256 is `85d91e827e4baf3b1f5bee2ba38044923ad0fee2b49ed1ef9afb05a0fba20840`. [Author commit][commit], [pinned tree][tree].

The paper describes patch-query/anchor-key-value attention with residual attention and FFN, shared text-space anchors concatenated with image-conditioned domain anchors, maximum-anchor attention pooling, and frozen-backbone refinement training. It excludes projected CLS from fusion (`w_p=0`). Its §4.6 rejects enforced semantic correspondence: anchors are not verified physical parts. These are paper descriptions and claims, not reproduced TriFusion evidence. [Paper §§3–4.6][paper].

What the released implementation actually does:

- **Direction and residuals:** `q=LN(patches)`, `k=v=LN(anchors)`; learned MHA Q/K/V and output projections precede `T←T+attention`, then `T←T+FFN(LN(T))`. Two blocks, eight heads and dropout 0.1 are configured. Patch tokens are residually updated, not replaced by pure anchor mixtures. [Actual block][block], [Market configuration][market].
- **Anchors:** shared text contexts run through a frozen text encoder and trainable 512→768 projection. Default contexts contain **4 trainable phrase-initialized tokens + 12 frozen phrase-token embeddings**, rather than entirely learnable CLIP-ReID-mean contexts. Fixed body/clothing/accessory phrases are resources, not per-image captions. Dynamic anchors are **three additional vectors**, not deltas added to shared anchors. [Text anchor construction and phrases][text], [domain MLP: token mean→Linear/GELU/Linear][domain], [concatenation][concat].
- **Readout:** raw block-9 patches exclude CLS and precede `ln_post`. Final head-averaged attention is maximized across anchors, normalized across patches, and pools the updated tokens. The 768→768 “projection” is identity; BN feeds classifiers, while evaluation takes the raw refined feature. [Backbone token extraction][tokens], [pooling/head/eval code][readout].
- **Training:** Stage 1 learns training-ID contexts; Stage 2 fine-tunes the image model with ID/triplet/I2T. Stage 3a/3b train visual modules on a frozen backbone; both refinement losses omit I2T by default. Stage 4 is disabled. Market/Occ-Market use four classifier heads and classifier decorrelation; Duke uses one. Training labels construct losses/text banks; inference feature extraction does not receive identity labels. [Stage 1–2][s12], [Stage 3 calls][s3], [Market][market], [Duke][duke], [Occ-Market][occmarket].
- **Fusion/normalization:** released features are 768 refined + 768 CLS + 512 projected CLS. Market and Duke inherit weights **(2, 0.2, 0.05)**; Occ-Market explicitly uses **(10, 1, 0)**. Each branch is L2-normalized and scaled by √weight. The evaluator also normalizes because string `"no"` is truthy; without NFC, constant total squared norm preserves weighted-cosine ranking. This source-path observation is not an execution check. [Fusion][fusion], [default weights][defaults], [evaluator][metrics], [Occ-Market][occmarket].

The proposed `X_hat = X + Attn(X,A,A)`, `A = A_shared + Delta(Pool(S))`, followed by the existing independent 16-region reader, is a **visual-only project adaptation**. Its direction and attention residual agree with SAGA. Its additive shared/delta construction, omitted FFN/text resources, region readout and retained raw global/role objectives differ. Keeping 1536D is the project's interface constraint; it coincides numerically with the paper's two 768D branches but differs in composition and from the released 2048D feature. SAGA's learned anchors do not supply evidence that those 16 regions become physical parts.

**Analytic design boundary, not a diagnosed error:** if `Delta` is one identical offset broadcast to every anchor and keys/values use affine projections, its key-logit shift is equal across anchors and cancels in softmax; its value contributes common image context. Per-slot deltas are materially different. Anchor-dependent normalization or other nonlinear transformations invalidate that simple cancellation argument. Specify the delta shape and transforms before interpreting the experiment; this observation warrants no current runtime branch.

Minimum direct controls to register **after the current capacity report**:

| Arm | Direct comparison |
|---|---|
| Current independent reader on original `X` | Existing source/checkpoint reference. |
| Patch→anchor attention only, then the same reader | Tests the proposed attention residual without FFN. |
| Attention + residual FFN, then the same reader | Separates FFN's added capacity from attention-only behavior; matches the actual block structure. |
| Same attention parameters, but repeat `Pool(X)` as every query; retain residual `X` | Needed for a future local-complementarity claim: preserves anchor generator and attention parameter shapes while removing patch-specific routing. It is not evidence that current TriFusion errors are caused by broadcast. |

Hold source initialization, raw global/role losses, region count, output dimension, training/selection budget and evaluation protocol fixed. Report actual trainable parameters and cost. Compare **shared-only vs image-conditioned** anchors only if claiming conditioning is necessary. These comparisons can test local routing and FFN/conditioning contributions; improvement over the current reader alone cannot identify reconstruction rather than extra capacity. No mAP prediction is warranted.

Both requested primary sources were available. The pinned repository tree contains no checkpoint/run-log evidence and no LICENSE/COPYING file; GitHub's repository metadata reports `license: null`. Public availability does not establish an explicit repository reuse license. No published number or RGBNT result was independently reproduced. Current source345/capacity campaign and its live jobs are outside this investigation; next source selection remains pending its final report. [Pinned tree][tree], [repository metadata][meta].

[paper]: https://arxiv.org/html/2604.22190v1
[commit]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/commit/a694589ce4f5d81b729b306ac0b69b1cef15224f
[tree]: https://api.github.com/repos/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/git/trees/a694589ce4f5d81b729b306ac0b69b1cef15224f?recursive=1
[meta]: https://api.github.com/repos/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID
[block]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/visual_modules.py#L77-L144
[domain]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/visual_modules.py#L16-L44
[text]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/train.py#L616-L744
[concat]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/train.py#L895-L931
[tokens]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/model/clip/model.py#L415-L488
[readout]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/train.py#L830-L1021
[s12]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/train.py#L375-L550
[s3]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/train.py#L1575-L1725
[fusion]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/train.py#L1833-L1893
[defaults]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/train.py#L119-L129
[metrics]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/utils/metrics.py#L91-L130
[market]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/configs/market1501.json
[duke]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/configs/occ_dukemtmcreid.json
[occmarket]: https://github.com/ipl-uw/Structured-Anchor-Guided-Aggregation-for-ReID/blob/a694589ce4f5d81b729b306ac0b69b1cef15224f/configs/occ_market1501.json
