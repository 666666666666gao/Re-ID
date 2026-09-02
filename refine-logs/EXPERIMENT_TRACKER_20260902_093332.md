# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V13-T0 | M0 | public seam RED→GREEN | shared fusion/counterfactual/bootstrap/gates | synthetic CPU | worked examples | MUST | PASS | real RED；V13 tests 7/7 |
| V13-P0 | M0 | remote adjacent tests | V13+V8/V12 tests | remote CPU/GPU import | pytest | MUST | PASS | commit `46b3e99`；19/19 |
| V13-P1 | M0 | real fold0 preflight | paired collector small batch | 141-fit fold0 | shapes/SHA/dev0/official0 | MUST | PASS | 8 samples；13.91s；dev0/official0 |
| V13-Q0 | M1 | actual-path teacher qualification | V12 OOF teacher + all-fit Phase-A student | 141-fit 3fold | Delta health/transfer | MUST | PASS | 571 queries；Oracle-minus-fixed `0.0014682`；transfer `+0.0008706` |
| V13-Q1 | M2 | policy + OOF replay qualification | fixed Router config | 141-fit 3fold | Delta/Top1/AP/margin/bootstrap/quality | MUST | TERMINAL_FAIL | fold0 Top1 FAIL；fold2 utility/AP/margin FAIL；四项 bootstrap LB<0 |
| V13-D1 | M3 | main deployable result | final Router + same Phase-A SHA | 30-dev | mAP/R1 | MUST | NOT_AUTHORIZED | Q1 FAIL；未refit、未访问dev |
| V13-A1 | M4 | target deletion | residual-only target | frozen protocol | OOF/dev mAP | NICE | NOT_AUTHORIZED | 主结果未成功 |
| V13-A2 | M4 | input deletion | fold raw input | frozen protocol | OOF/dev mAP | NICE | NOT_AUTHORIZED | 主结果未成功 |
