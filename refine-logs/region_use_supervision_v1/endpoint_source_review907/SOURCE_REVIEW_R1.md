# Endpoint907 source integrity review R1

Verdict: **FAIL — two minimal source-contract fixes required.** This is a same-family/provisional source review, not result acceptance. Requested reviewer: Astra / max; actual native serving model and effort: **UNATTESTED**.

Reviewed new report SHA256: `2723dc8b1f53d10a7943d3103467c9e7a9bcafed32f67a71195ac8f9559a955a`. Plan SHA256: `7f58c0bf4fbb784d019a22d6e318c4f343892c0a8be0cedb36bde778e94ec936`. All 28 captured source/text inputs are listed in `REVIEWED_INPUT_MANIFEST_R1.json`; original new inputs remain unchanged in `inputs/`.

## Findings

1. **F1, P2 — exact equality conflicts with the actual producer tolerance.** `inputs/tools/report_region_use_endpoint.py:45` requires exact receipt/history metric equality. `inputs/tools/run_foundation_recipe.py:301` and `inputs/tools/queue_foundation_recipe.py:96` require absolute differences below 1e-5. The isolated scalar fixture with selected mAP 50.0 and strict receipt 50.000005 passes the producer assertion and fails R1. Use that existing tolerance; do not change training, evaluation, selection or kernel settings.
2. **F2, P2 — the old raw batch-order digest is available but unchecked.** `inputs/tools/report_region_use_endpoint.py:55` reads the old file and line 56 only compares bytes. The current artifact seal already pins each dataset's complete raw comparator order. Matching deliberately changed text fixtures pass R1 despite a seal mismatch. Check the old path's sealed SHA before reading/comparing. No production tampering is alleged.

## A–F audit

| Check | Source finding |
|---|---|
| A — GT | PASS within source scope. `run_correspondence_roles.py:88` takes identities/cameras/scenes from sealed protocol metadata, and `:104` saves them with complete distances. Protocol counts/hashes and all legal-query positive counts passed stdlib checks. No prediction-derived GT. |
| B — score math | PASS within source scope. `train_msvr310_signal_oof.py:223` and `train_rgbnt100_signal_oof.py:253` implement complete legal AP/CMC; same-environment negatives remain. No own-model score normalization. |
| C — result/acceptance | FAIL F1. Ordered exit0 four-stage/COMPLETE/full50/best/checkpoint/distance checks exist (`report_region_use_endpoint.py:30–45`), but exact equality introduces an unintended stronger gate. Actual result acceptance is unperformed. |
| D — live call path | PASS within source scope. `run_region_use_probe.py:60,83,122` reaches the actual logger and foundation train/evaluate chain; report `:70` calls inherited descriptor/paired and its actual scorers. |
| E — order/scope | FAIL F2. Formal counts, per-epoch counts and ordered identities are checked (`:48–60`), but the old order is not pinned at use. Scope is honestly one seed42 selected-best descriptive diagnosis. |
| F — type | Intended evaluation is real_gt; audit fixtures are simulation_only. No neural acceptance follows from source tests. |

The AP transitions repaired/new-error/still-correct/still-wrong contribute on the full query denominator and sum to delta mAP (`:73–79`). Inherited `paired:41` checks repairs minus new errors against Rank-1 delta. The tail is the smallest `ceil(.05*N)` signed AP changes (`:80–81`), including positive values if every query improves. There is no averaging over training seeds or implied own-global/local score.

Completed individual jobs are permitted while the broader campaign remains RUNNING. The fixtures accept this and reject the supplied actual unfinished first node; no whole-campaign terminal or fresh kernel gate is imposed.

## Timing and provenance boundaries

Producer wall time is the queue's train-stage interval after Popen through wait completion, including construction and per-epoch scoring/save, excluding GPU admission wait, prepare/M0 and separate strict evaluation. The training JSON interval starts after build/optimizer/loader, and ends before terminal receipt/wrapper writes. Epoch loop seconds are captured before subsequent evaluation/save (`run_foundation_recipe.py:200,212,246,283`; `queue_native_research.py:67–73`).

`run_native_partitioned.py:37–50` writes both cuda:0/cuda:1 allocated/reserved peaks. GPU1 reset includes optimizer setup; GPU0 resets again before the loop. Treat these as producer counters with those boundaries, not symmetric whole-process memory maxima.

The supplied last actual witness is RUNNING at 2026-10-12T19:46:37.472459+08:00. No new full50 endpoint is present. Source status/receipt consistency does not independently prove remote execution, binary contents, image GT, or performance.

The broader source-map inventory is documented in `LOCAL_SOURCE_SCOPE_LIMITS_R1.json`: 329/463 raw-exact local files, 107 unavailable, 27 CRLF/LF differences. Direct reviewed scorer/metadata sources and all three protocols are exact. The missing directly relevant upstream author scorer is `comparators/Signal-cd1b0a6/utils/metrics.py` (sealed SHA `91604acb7d978462c16904910a645d7d1697c15bf150da529099004c8b5eecb2`). This does not create a new production gate. `run_trifusion_experiment.py` is not treated as the sealed production version.

Validation: `SOURCE_ONLY_CHECKS_R1.json` records 12 checks, including both bug reproductions. The script uses stdlib and isolated AST statements only: zero torch/NumPy imports, tensor reads, models, optimizers, GPU queries, SSH calls or production invocations. Source and live queue were not mutated by the reviewer.
