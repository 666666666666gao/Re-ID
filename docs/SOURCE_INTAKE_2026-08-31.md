# Source intake: multimodal ReID trend review

## Frozen source

- Windows path: `E:\调研综述趋势_2026-08-31_01-59.md`
- WSL2 path: `/mnt/e/调研综述趋势_2026-08-31_01-59.md`
- Bytes: `91632`
- Last modified (UTC): `2026-08-30T17:59:45`
- SHA-256: `7063a578635e485b2168bba06abad9fa27ea51e91f793d69952a890041f537ba`

The source file is an input record and is not copied into or modified by this
repository. Its literature numbers are treated as leads rather than ground
truth; primary-source verification lives in
`docs/RESEARCH_AUDIT_2026-08-31.md`.

## Requirements distilled from the source and user direction

1. RGB, NIR and TIR are synchronized sensor modalities; CNN, Transformer and
   Mamba are architecture experts. Architecture and modality must not be bound
   one-to-one.
2. The three architecture branches must each be independently discriminative
   and must all participate in the paper's main model. The user's explicit
   requirement for deep three-way collaboration overrides the review's optional
   recommendation to make Top-1/Top-2 sparse routing the primary design.
3. Collaboration must occur before the final head so that information from one
   architecture can improve the native representation learned by the other two;
   a late ensemble alone is not sufficient.
4. Local texture/boundaries, global token relations and bidirectional
   long-range state propagation are the intended CNN, Transformer and Mamba
   roles. The implementation must measure whether these roles remain distinct.
5. Reliability must be sample-, modality- and architecture-dependent. It must
   react to blur, occlusion, exposure, noise and missing modalities instead of
   degenerating to a fixed `RGB -> CNN` mapping.
6. Before attributing gains to collaboration, the experiment suite must include
   three single branches, mean/concat controls, a parameter-matched larger late
   fusion control, branch error overlap and an oracle-complementarity audit.
7. Complete-modality retrieval, six missing-modality masks, controlled quality
   degradation, active FLOPs, latency, memory, expert utilization and
   cross-dataset generalization are all part of the evidence contract.
8. Results using stronger pretraining, generated text, segmentation masks or
   test-time parameter updates must be separated from the static CLIP-B/16
   visual track. A SOTA statement is permitted only after same-protocol,
   multi-seed evidence exceeds the corresponding public target.

## Project mapping

| Source requirement | Authoritative project artifact |
|---|---|
| Dataset acquisition and integrity | `tools/audit_*.py`, `../artifacts/*_audit_20260831.json` |
| Reproducible WSL2/CUDA/Mamba environment | `environment.yml`, `requirements-lock.txt`, `scripts/build_mamba_sm89.sh` |
| High-metric measurable baseline | `tools/reproduce_mdreid.py`, `../artifacts/mdreid_rgbnt201_eval_20260831.json` |
| Open-source implementation-base calibration | `tools/run_demo_baseline.py`, `scripts/run_demo_rgbnt201_seed42.sh` |
| Three-way collaborative method and claim gates | `docs/METHOD_SPEC_V1.md` |
| Frozen train-only development protocol | `protocols/rgbnt201_dev_v1.json` |
| Full ablation and promotion matrix | `refine-logs/EXPERIMENT_PLAN.md`, `refine-logs/EXPERIMENT_TRACKER.md` |
| Primary-source novelty and SOTA tracks | `docs/RESEARCH_AUDIT_2026-08-31.md` |

## Main-method interpretation

The paper model is therefore the dense collaborative HFER + URGC + RDPT
configuration: all three experts run, exchange role-preserving information at
multiple depths, share one calibrated reliability posterior for relay and
fusion, and teach peers only in reliability-supported directions. Optional
sparse execution may be studied later as an efficiency extension, but cannot
replace the requested full three-branch main result.
