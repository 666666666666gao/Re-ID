# V15 Experiment Tracker

| Run ID | Milestone | Purpose | Split | Priority | Status | Next condition |
|---|---|---|---|---|---|---|
| V15-M0-TDD | M0a | CRDE/counterfactual public seams | synthetic | MUST | PASS: 17 adjacent tests | complete |
| V15-M0-REMOTE | M0b | exact preflight+B64/K8+overfit | fit-only | MUST | PASS: `1f2de44` | Q1 authorized |
| V15-Q1-S42 | Q1 | complete-path OOF mechanism gate | V12 3 folds | MUST | AUTHORIZED_NOT_STARTED | full pass → D1; failure → seal |
| V15-D1-S42 | D1 | all-fit final-only one dev | 141-fit/30-dev | CONDITIONAL | BLOCKED_BY_Q1 | ≥65+strict wins required |

禁止项：Q1前dev；D1前消融；任何阶段official、多seed、LR/epoch/regret/edge
scale扫描或reranking。

M0 terminal receipt: `evidence/trifusion_v15_m0_seed42_1f2de44.json`.
The earlier `evidence/trifusion_v15_m0_seed42.json` is retained as an invalid-
gate implementation receipt, not a scientific M0 failure. The corrected M0
uses the preregistered two-live-exchange capacity condition and subtracts the
fixed-comparator regret floor in the 100-step excess-loss ratio.
