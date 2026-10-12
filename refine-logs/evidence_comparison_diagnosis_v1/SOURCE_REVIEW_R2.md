# Same-evidence diagnostic: source/plan review R2

**Verdict: SOURCE_ONLY_PASS for the planned absolute-path launch. No remaining source changes are required within that scope.**

Reviewed on 2026-10-12. Requested reviewer: `gpt-6-astra`, reasoning `max`; actual model/reasoning identity **UNATTESTED**. Review independence **same-family**, acceptance **provisional**. This follow-up independently checked the actual revised source against R1. It is not neural/runtime verification or experimental acceptance.

## Exact reviewed changes and preservation

- Runner `tools/diagnose_same_evidence_comparison.py`: SHA-256 `71fe059e26744c670661cab6fb3bbaaf13da56a5567adf78b8079e2ba0db9ad5`.
- Plan `refine-logs/evidence_comparison_diagnosis_v1/EXPERIMENT_PLAN.md`: SHA-256 `0ea17e67772806f1f6fbbb2082c07fda4b5b380efa1da5195c17a064976e9bf4`.
- Preserved R1 runner: SHA-256 `c32d13272cda614f5367bd20155f9487b3b434479c5b5383a3e4ea5e7c68f373`, verified against `DIAGNOSTIC_R1_SOURCE.py.txt`.

Removing the new comment and `os.chdir(output.resolve())` exactly reconstructs R1 runner text. Removing only the plan's CWD disclosure and reverting its cache number reconstructs the original R1 plan hash. Sixteen other supplied files and all eighteen supplemental runtime snapshot files retain their reviewed R1 hashes. R1's Markdown/JSON remain unchanged with their original **REVISE** verdict.

## R1 issue resolution and path scope

**R1 is resolved for the declared absolute output path.** Revised runner lines 149–150 change the endpoint process cwd immediately before any `base.score` call. The pinned author's `eval_func_msrv` opens `re.txt` relative to cwd (`comparators/Signal-cd1b0a6/utils/metrics.py:38` and `:71`); it now writes inside the fresh endpoint directory. Model/initialization/checkpoint construction, both feature extractions, cache saves, original-distance loading, and metadata comparison occur before the CWD change, at runner lines 123–148. No author source modification was introduced.

The path safety argument has an explicit boundary: **the concrete launch must pass an absolute `--output-dir`**, alongside the plan's absolute sealed input paths. `output` at line 121 retains the Path supplied by the CLI; `output.resolve()` at line 150 resolves the argument to `chdir` but does not reassign `output`. With an absolute output path, later cache stat/hash reads at lines 187–188 and the `DIAGNOSIS.json` write at line 191 continue to address the original endpoint correctly. Source paths use absolute `ROOT`; historical artifact keys were directly checked to be absolute POSIX paths. The model's source/weight/protocol paths are constructed before scoring.

A relative `--output-dir` would instead be reinterpreted after cwd changes and is **outside this bounded launch review**. No broad CLI compatibility claim is made, and no extra fallback or exception machinery is requested. The actual new launch command and final seal were not supplied as completed artifacts; their absolute-path coverage remains a prelaunch check already required by the plan.

Each dataset runs in its own subprocess and the parent waits for it (`coordinate`, lines 206–220). Changing an endpoint's cwd therefore does not change the coordinator's cwd or another endpoint's paths. The seven MSVR evaluations still replace `re.txt` within their own output directory; the declared per-query/metric results remain in `DIAGNOSIS.json` and do not rely on preserving separate author rank-text files.

## Arithmetic and unchanged scientific behavior

Plan line 23 now correctly states **1,254,113,280 B** for 13,608 records with three 1536-dimensional features and `3 × 3 × 16 × 128` region values, all FP32. Adding 64 MiB allowance and 2 GiB reserve gives **3,468,705,792 B**. This is a prelaunch correction of the earlier conservative estimate, not an observed capacity failure or gate relaxation.

Exact text-delta checks establish that the following are unchanged: the final-region hook and shapes; raw global/correction/additive tensors; normalization; group-preserving regional reductions; same-residual and fixed-energy algebra; all seven full-gallery scores; official camera/scene exclusions and metric-parity gate; per-query repair/damage summaries; checkpoint/initializer selection; model-state checks; inference count; optimizer absence; and serial GPU0/1 admission controls. No scoring, neural computation, fitting, or selection change was introduced. The existing R1 scalar-algebra and source-construction findings continue to apply to unchanged code; no new runtime result is inferred from them.

| A–F check | R2 status |
| --- | --- |
| A. Ground truth provenance | PASS, unchanged frozen dataset records. |
| B. Normalization | PASS, unchanged feature-norm definitions and ordinary AP/CMC. |
| C. Result existence/provenance | WARN/pending: no new scores, checkpoint rehash, completed launch seal, or runtime reproduction established by this review. |
| D. Active path | PASS within absolute-path launch scope; known MSVR output is contained. |
| E. Scope/missingness | PASS source-only: three fixed historical seed42 models, seven predetermined full-gallery comparisons; failed endpoints remain missing, physical absence/conflict untested. |
| F. Evaluation type | Planned dataset `real_gt` retrieval; dispersion remains descriptive, algebra checks remain implementation checks. |

No SSH, source imports, torch/CUDA, neural execution, installation, source edits, or new agents were used for R2. The follow-up used only local reads/hashes/text comparison and PowerShell integer arithmetic, with writes restricted to the private review directory.

## Remaining runtime obligations

Before launch, bind these revised hashes into the complete new-schema source/artifact seal, verify the exact absolute-path command and storage gate, and perform the planned permitted GPU admission check. Subsequently require actual additive reproduction within `1e-5`, valid tensor/state checks, all endpoint results, and unchanged input hashes. SOURCE_ONLY_PASS does not establish improvement, compression causality, novelty, physical visibility, or a successful neural run.

Evidence: `R2_DELTA_CHECKS.json` and `check_r2_delta.ps1`. R1 remains preserved rather than overwritten or relabeled.
