# V15 Experiment Tracker

| Run ID | Milestone | Purpose | Split | Priority | Status | Next condition |
|---|---|---|---|---|---|---|
| V15-M0-TDD | M0a | CRDE/counterfactual public seams | synthetic | MUST | TODO | green → M0b |
| V15-M0-REMOTE | M0b | exact preflight+B64/K8+overfit | fit-only | MUST | BLOCKED_BY_M0a | pass → Q1 |
| V15-Q1-S42 | Q1 | complete-path OOF mechanism gate | V12 3 folds | MUST | BLOCKED_BY_M0b | full pass → D1; failure → seal |
| V15-D1-S42 | D1 | all-fit final-only one dev | 141-fit/30-dev | CONDITIONAL | BLOCKED_BY_Q1 | ≥65+strict wins required |

禁止项：Q1前dev；D1前消融；任何阶段official、多seed、LR/epoch/regret/edge
scale扫描或reranking。
