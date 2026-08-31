# Baseline selection and license audit

> Audit date: 2026-08-31  
> Scope: pinned upstream source revisions used or considered by TriFusion-ReID.  
> This is an engineering provenance review, not legal advice.

## Decision

The single **high-metric, repository-level open-source baseline** selected for
the user's requirement is **Signal**, pinned at
[`cd1b0a672d1fe642e7608731cb4899a19dda7d51`](https://github.com/010129/Signal/tree/cd1b0a672d1fe642e7608731cb4899a19dda7d51).
That revision contains an MIT `LICENSE` and reports 80.3% mAP / 85.2% Rank-1
on RGBNT201. The released log and real B64/K8 loader have been audited, but the
checkpoint bytes have not been obtained; therefore those numbers remain an
**upstream fixed-path test-log result**, not a local reproduction. The primary
fair row will be a locally trained, pre-registered fixed epoch-50 Signal
checkpoint, not `Signalbest.pth` selected on official-test mAP.

One repository cannot honestly fill every engineering and empirical role, so
the remaining roles are kept explicit:

- **implementation scaffold:** DeMo `b4f323a`, MIT, already integrated and
  exercised on the complete training graph;
- **strongest local checkpoint anchor:** MDReID `3525ac2`, locally reproduced
  at 82.0868% mAP / 85.1675% Rank-1, but without a repository-level license;
- **strongest source-visible training comparator:** PEFT-BoA `d2b198b`, whose
  public log gives fixed epoch-120 82.2% / 85.8% and test-selected epoch-80
  82.7% / 86.1%, but whose repository has no repository-level license.

Thus “baseline” in paper tables must carry a role label. Signal satisfies the
requested high-metric open-source baseline; DeMo is not misrepresented as the
strongest method, and an unlicensed GitHub repository is not misrepresented as
open source.

## What counted as open source

For every local checkout, the audit inspected the exact Git tree at the pinned
commit, not only the GitHub “Public” badge. A repository is classified as
open-source here only when that tree contains an explicit repository-level
license granting modification and redistribution rights. Public readability,
a training command, a checkpoint link, or an acknowledgement of another
project is insufficient.

Several unlicensed repositories contain isolated inherited files with BSD or
MIT header comments. Those notices govern the named files; they do not license
the repository authors' remaining code. Some even refer to a root `LICENSE`
that is absent from the pinned tree. Consequently the audit does not infer a
blanket license from those fragments.

## Pinned local-source audit

| Repository | Commit and reported RGBNT201 mAP / R1 | Repository license at that commit | Classification | TriFusion use |
|---|---|---|---|---|
| [Signal](https://github.com/010129/Signal/tree/cd1b0a672d1fe642e7608731cb4899a19dda7d51) | `cd1b0a6`; 80.3 / 85.2 upstream log | MIT; `LICENSE` SHA-256 `cc38a11e…f4e5` | Open source | **Selected high-metric licensed baseline**; retrain fixed e50 before claiming a local number |
| [DeMo](https://github.com/924973292/DeMo/tree/b4f323a430b32e3a1637c3e7acb25868cb52e9cd) | `b4f323a`; 79.0 / 82.3 paper | MIT; `LICENSE` SHA-256 `318ca45d…b65d` | Open source | Implementation scaffold and matched fixed-e50 baseline |
| [MambaPro](https://github.com/924973292/MambaPro/tree/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149) | `f9ee6f6`; 78.9 / 83.4 paper figure | MIT; same license bytes as DeMo | Open source | Licensed lineage/reference; not the strongest baseline |
| [MDReID](https://github.com/stone96123/MDReID/tree/3525ac2da1a2a90a5a160c930fac674b4f226f6c) | `3525ac2`; local 82.0868 / 85.1675 | No repository-level license in the pinned tree | Source-visible, not established open source | Isolated checkpoint evaluation only; do not copy code into TriFusion |
| [PEFT-BoA](https://github.com/fffunly/PEFT-BoA/tree/d2b198be634ac4f9f5744eebf6e0a6604e490deb) | `d2b198b`; fixed e120 82.2 / 85.8; test-selected e80 82.7 / 86.1 | No repository-level license in the pinned tree | Source-visible, not established open source | Isolated training comparator; do not copy code into TriFusion |
| [MFRNet](https://github.com/stone96123/MFRNet/tree/ec54a1302321cda4b5fad9ca1c0878dabf0b46b6) | `ec54a13`; 80.7 / 83.6 test-selected | No repository-level license in the pinned tree | Source-visible, not established open source | Isolated official-checkpoint comparator; no code reuse |
| [UGG-ReID](https://github.com/wanxixi11/UGG-ReID/tree/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1) | `eaf1e8e`; 81.2 / 86.8 paper | No repository-level license in the pinned tree | Source-visible, not established open source | Paper/concept comparator and clean independent reimplementation only |

The exact commits, tree objects, tracked-file counts, remotes, license paths,
license hashes, and inherited file-level notices are frozen in
`evidence/baseline_license_audit_20260831.json`.

## Stronger-looking candidates that do not replace Signal

- [ProxyTTT](https://github.com/liuzhaojun-zwd/ProxyTTT/tree/92fb0fa33d74813566e06820e56e8d8f48ca1205)
  exposes code and an 85.0 / 88.5 checkpoint result, but that result performs
  two epochs of test-time training and belongs to a different inference track.
  Its static `w/o TTT` paper row is 82.3 / 84.7, and the pinned repository tree
  shows no repository-level license. It is neither the static licensed
  baseline nor an unconditional SOTA target.
- [PRISM](https://github.com/zw-absin/PRISM/tree/0067f6d895c522afa2c4f30515b33bc4300fe680)
  is MIT licensed, but its RGBNT201 route consumes precomputed foreground masks
  during training and testing. Its 80.5 / 84.0 result is retained in the
  extra-resource track, not used as a same-input baseline.
- RoDI-CLIP remains the paper-reported static CLIP target at 84.1 / 87.2, but
  its fixed repository contains no runnable model code or checkpoint. It is a
  target threshold, not a reproducible baseline.

## Binding reuse and reporting policy

1. New TriFusion implementation may derive from DeMo, MambaPro, or Signal only
   while preserving their MIT notices and documenting copied/modified files.
2. MDReID, PEFT-BoA, MFRNet, UGG-ReID and ProxyTTT stay in isolated upstream
   checkouts. Use them for private reproducibility evaluation or implement ideas
   independently from papers; do not copy or redistribute their unlicensed
   repository code as part of TriFusion.
3. A code license does not automatically license pretrained weights or dataset
   bytes. Checkpoints and datasets retain separate provenance and redistribution
   boundaries.
4. Until Signal fixed-e50 is locally trained and evaluated, write
   `Signal 80.3/85.2 (upstream fixed-path log; not locally reproduced)`.
5. Keep test-selected, fixed-epoch, checkpoint-parity, TTT and extra-resource
   columns separate. No result in this audit proves TriFusion is SOTA.

## Reproduction consequence

The next baseline work item is an isolated Signal environment/adapter followed
by a real-model 8 GiB capacity gate and a seed-1234 fixed-e50 run that never
uses official-test metrics for checkpoint selection. The current GPU preflight
was 1,123 MiB used, above the registered `<500 MiB` launch threshold, so no GPU
job was started during this audit.

