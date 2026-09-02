# V15 Experiment Tracker

| Run ID | Milestone | Purpose | Split | Priority | Status | Next condition |
|---|---|---|---|---|---|---|
| V15-M0-TDD | M0a | CRDE/counterfactual public seams | synthetic | MUST | PASS: 17 adjacent tests | complete |
| V15-M0-REMOTE | M0b | exact preflight+B64/K8+overfit | fit-only | MUST | PASS: `1f2de44` | complete |
| V15-Q1-S42 | Q1 | complete-path OOF mechanism gate | V12 3 folds | MUST | FAIL: gate false | V15 sealed |
| V15-D1-S42 | D1 | all-fit final-only one dev | 141-fit/30-dev | CONDITIONAL | NOT_AUTHORIZED | not executed |

禁止项：D1、dev、official、消融、多seed、checkpoint selection，以及
LR/epoch/regret/edge scale/fold/threshold/reranking扫描。

Q1 terminal receipt: `evidence/trifusion_v15_q1_seed42_71152d3.json`.
`status=PASS` means execution completed; `gate.passed=false` and
`next_phase_authorized=false` are the scientific verdict.
