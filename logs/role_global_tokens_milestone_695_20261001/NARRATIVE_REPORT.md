# Nine-end global-token comparison: complete, original advancement FAIL

All nine fresh seed42/full50 endpoints and all27 M0/train/evaluate subprocesses completed with real exit0. The final observer recorded the terminal state at 2026-10-01T10:30:00.474989+08:00 and itself completed0 at10:30:00.613595. The persistent report waiter invoked the prepared CPU report exactly once,10:28:23.147195–10:28:49.475757, exit0. No completed training, observer or report was restarted.

The immutable intake contains103 original text/JSON/source/log/plot files, preserved10:39:02.897889, each individually SHA/byte verified. Archive1427567bytes, SHA256 `f82e7b39c8a20fd0aa0646abaf8e9ad1ec361edb5ded45412a3e2f2fb0666b8f`. The complete raw SUMMARY is4078932bytes, SHA256 `d750b7a651f745ff95b70a05cee12a245bf828114d5ac4fd4c64fd29e906e16b`. Checkpoints, distance tensors and dataset images remain remote. Original215 bound runtime sources remain unchanged.

## Full50, one fused-mAP-best checkpoint per row

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | static | 2 | 72.5308 | 74.0431 | 82.7751 | 87.9187 |
| RGBNT201 | token | 2 | 72.5621 | 73.9234 | 82.4163 | 87.7990 |
| RGBNT201 | direct | 2 | 72.5220 | 73.9234 | 82.6555 | 87.9187 |
| RGBNT100 | static | 1 | 85.1035 | 95.0437 | 95.6851 | 96.0350 |
| RGBNT100 | token | 1 | 85.1178 | 94.9854 | 95.6851 | 95.9767 |
| RGBNT100 | direct | 1 | 85.1020 | 95.1020 | 95.6851 | 96.0350 |
| MSVR310 | static | 10 | 52.4849 | 68.0203 | 83.4179 | 89.3401 |
| MSVR310 | token | 10 | 52.4877 | 68.1895 | 83.4179 | 89.0017 |
| MSVR310 | direct | 10 | 52.4341 | 68.1895 | 83.5871 | 89.1709 |

| Dataset | Control to token | ΔmAP | ΔR1 | Repairs | New first-match errors | Identity macro ΔAP |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | static to token | +0.0313 | -0.1196 | 1 | 2 | +0.0320 |
| RGBNT201 | direct to token | +0.0401 | 0 | 2 | 2 | +0.0426 |
| RGBNT100 | static to token | +0.0143 | -0.0583 | 0 | 1 | +0.0111 |
| RGBNT100 | direct to token | +0.0158 | -0.1166 | 0 | 2 | +0.0131 |
| MSVR310 | static to token | +0.0028 | +0.1692 | 8 | 7 | -0.0363 |
| MSVR310 | direct to token | +0.0535 | 0 | 7 | 7 | -0.1269 |

All six comparisons fail the originally registered rule. Do not add seeds, retune weights or change that threshold to rescue this version. Tiny positive mAP changes do not establish useful global-conditioned interaction.

## What changed, and what did not

The total correction alone becomes more discriminative in the token condition: mAP37.0356/61.5935/18.4742 for201/100/MSVR, versus static33.1332/56.7783/6.0656. However, fused-minus-its-jointly-trained-global mAP is only+0.0706/+0.0398/+0.0479. For MSVR the token fusion repairs8 but introduces9 first-match errors relative to that same checkpoint's global, and identity macro ΔAP is−0.2057. This differs from the candidate-control comparison above. A more discriminative correction does not by itself ensure a useful addition to the shared representation.

`joint_local` is the normalized total correction and carries global information in token/direct; it is not pure-local evidence. The checkpoint's global was jointly trained and is not an independent global-only control. These decompositions and label-based repair counts are explanatory, not deployable selectors or oracle gains.

All nine best-to-final mAP trajectories deteriorate while mean CE+Triplet decreases. All three RGBNT100 runs have6559 logged steps and zero Triplet throughout,19677steps in total. This confirms lack of active hinge support in those runs, not a unique explanation or optimizer-update share. No new objective is selected by mining individual official failures.

Actual campaign interval9385.471587s (2.607h); summed endpoint intervals30026.496473s (8.341h). The sum is not elapsed wall time and neither includes upstream pure-ReID training. Endpoint costs reflect GPU concurrency/evaluation and do not prove equal compute. Matching initial state and parameter count do not equate effective capacity.

## Claim boundary and successor

Full official query/gallery and source-label camera/time filtering remain unchanged, including MSVR gallery-only distractors. One seed and official-best development selection do not establish robustness, untouched-test significance, novelty or SOTA. The overall three-dataset goal remains unmet. A fresh same-family/provisional experiment audit is archived separately once it actually returns.

The next registered comparison changes only the frozen visual initialization: saved ReID visual tensors versus public CLIP visual tensors, with trained camera/nonvisual state and new trainable initialization preserved. See `refine-logs/visual_start_roles_v1/EXPERIMENT_PLAN.md`. This tests a previously documented lineage distinction; it is not a new module, a wholly public-only system, or a proven cause. Preparation/initialization witnessing is distinct from actual production M0/training. All original pure-baseline weights are retained.

## Actual returned integrity review

The fresh native `gpt-6-astra`/max reviewer returned WARN on 2026-10-01T02:52:39Z. This is a same-family provisional review, not independent acceptance. The nine complete endpoints and all six registered advancement failures are supported. It checked 27,266 protocol records, all 3,142 legal query masks, all 27 distance matrices and 108 aggregate metrics; maximum aggregate discrepancy was 0.000002851897 percentage points. It also parsed 30,624 formal steps, all 450 epoch summaries, 27 stage logs, CPU-loaded nine best and nine M0 checkpoints plus three baselines, and verified 215 frozen source hashes. No new neural forward, GPU execution, production construction or report rerun occurred during review.

All nine first M0/full batches match exactly. The additional first-eight-step bitwise scalar probe failed for all nine endpoints, with maximum later discrepancy 0.000149250031; the cause is not localized. This supplementary equality was not a registered completion gate. Preserve this limitation and do not claim bitwise training reproducibility. Single-seed official-best selection also precludes stability or untouched-test significance claims. Raw returned Markdown/JSON and response are archived under `.aris/traces/experiment-audit/2026-10-01_role_global_tokens_695`; the original returned JSON keeps `trace_path: null`, with actual archival metadata separate. No scoring correction or training replay is requested by the review.
