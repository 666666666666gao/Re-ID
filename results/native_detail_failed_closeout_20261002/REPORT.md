# Original N1 failed campaign closeout

**Original acceptance: FAILED; accepted5/6; invalid1; original success-report invocations0.**

| Dataset | Variant | Status | Best epoch | mAP | R1 | R5 | R10 |
|---|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | high | VERIFIED_COMPLETE | 10 | 68.3015 | 69.2584 | 78.7081 | 83.8517 |
| RGBNT100 | high | VERIFIED_COMPLETE | 15 | 79.3874 | 95.2187 | — | — |
| MSVR310 | high | VERIFIED_COMPLETE | 16 | 51.1501 | 68.0203 | — | — |
| RGBNT201 | low | VERIFIED_COMPLETE | 13 | 68.1329 | 71.0526 | 79.3062 | 84.5694 |
| RGBNT100 | low | VERIFIED_COMPLETE | 16 | 79.1748 | 93.7609 | — | — |
| MSVR310 | low | FAILED_EVALUATION_AFTER50 | — | — | — | — | — |

| Dataset | High minus | ΔmAP | ΔR1 | Repairs/new errors | Gate |
|---|---|---:|---:|---|---|
| RGBNT201 | roles | -1.5749 | -2.1531 | 40/58 | N1-A: FAIL |
| RGBNT201 | low | +0.1686 | -1.7943 | 38/53 | N1-B: FAIL |
| RGBNT201 | global_only | +0.1898 | -0.4785 | 44/48 | Descriptive only |
| RGBNT100 | roles | -1.5357 | +0.9329 | 45/29 | N1-A: FAIL |
| RGBNT100 | low | +0.2125 | +1.4577 | 40/15 | N1-B: PASS |
| RGBNT100 | global_only | +0.8961 | +1.3411 | 55/32 | Descriptive only |
| MSVR310 | roles | -0.8541 | -0.6768 | 17/21 | N1-A: FAIL |
| MSVR310 | low | — | — | — | UNAVAILABLE: original reload failure |
| MSVR310 | global_only | -0.5723 | +0.8460 | 19/14 | Descriptive only |

- Original six-end acceptance remains FAILED: five valid endpoints and one invalid strict-reload endpoint.
- Original success-report waiter exited without invocation; this separate CPU closeout does not replace it or create an accepted matrix.
- All six original training arms reached50; failed deployment metrics remain null, diagnostic reruns never substitute.
- N1-A failed on all three datasets; N1-B already fails201 and its MSVR comparison remains unavailable.
- N1 replaces CNN semantic values with image-native values while retaining semantic keys; it does not test preserving both evidence sources.
- N1-A jointly changes CNN value source, candidate grid from128 to512 positions, and capacity by93,248 parameters.
- N1-B matches parameters, initialization and512 candidates while changing input-detail computation; low upsampling does not restore lost detail.
- Old clean controls are pinned original results, not rerun or reselected. Global-only comparisons are descriptive, without a new gate.
- All valid formal metrics use real GT/full gallery/original environment filtering/no rerank and one mAP-best checkpoint per50epoch run.
- Official benchmarks were consumed in development/epoch selection; one seed and fixed-model identity bootstrap do not prove unbiased or multi-seed stability.
- Training costs include epoch evaluations; exclude construction, final reload, prior controls and separate failure diagnostics.
- No N2/N3 implementation, individual-role necessity, baseline/SOTA achievement or evidence of a unique failure cause.
