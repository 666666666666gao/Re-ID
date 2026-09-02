# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V13-T0 | M0 | public seam RED→GREEN | shared fusion/counterfactual/bootstrap/gates | synthetic CPU | worked examples | MUST | TODO | 已有用户`接缝同意` |
| V13-P0 | M0 | remote adjacent tests | V13+V8/V12 tests | remote CPU/GPU import | pytest | MUST | BLOCKED | 等T0 |
| V13-P1 | M0 | real fold0 preflight | paired collector small batch | 141-fit fold0 | shapes/SHA/dev0/official0 | MUST | BLOCKED | 等P0；GPU<500MiB |
| V13-Q0 | M1 | actual-path teacher qualification | V12 OOF teacher + all-fit Phase-A student | 141-fit 3fold | Delta health/transfer | MUST | BLOCKED | 等P1 |
| V13-Q1 | M2 | policy + OOF replay qualification | fixed Router config | 141-fit 3fold | Delta/Top1/AP/margin/bootstrap/quality | MUST | BLOCKED | 仅Q0 PASS |
| V13-D1 | M3 | main deployable result | final Router + same Phase-A SHA | 30-dev | mAP/R1 | MUST | BLOCKED | 仅Q1 PASS；一次 |
| V13-A1 | M4 | target deletion | residual-only target | frozen protocol | OOF/dev mAP | NICE | BLOCKED | 仅D1≥65且strict wins |
| V13-A2 | M4 | input deletion | fold raw input | frozen protocol | OOF/dev mAP | NICE | BLOCKED | 仅D1≥65且strict wins |
