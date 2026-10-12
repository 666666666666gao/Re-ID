# Saved-query AP damage and identity distribution

Post-development description of original selected-best, single-seed saved query results versus fixed independent global. AP gain/loss signs are exact saved floating-point comparisons, not statistical significance. All queries and identities retained; identity macro is equal weight per identity and does not replace query mAP. Four Rank-1 outcome groups partition queries; unchanged Rank-1 does not imply unchanged AP. Gross loss sums only negative AP changes; the largest identity fraction uses that gross-loss denominator, not net mAP. No same-model branch causality, unique failure cause, raw-GT reauthentication, deployable GT gate, NN, training, candidate or method selection.

| Dataset | Condition | Queries / identities | Query mAP delta | Identity macro AP delta | Identities gain / loss / unchanged |
|---|---|---:|---:|---:|---:|
| RGBNT201 | uniform | 836 / 30 | +0.4914 | +0.4513 | 20 / 8 / 2 |
| RGBNT201 | ambiguity | 836 / 30 | +0.3332 | +0.2730 | 23 / 5 / 2 |
| RGBNT100 | uniform | 1715 / 50 | -1.2480 | -0.7624 | 21 / 27 / 2 |
| RGBNT100 | ambiguity | 1715 / 50 | -1.1879 | -0.7668 | 22 / 26 / 2 |
| MSVR310 | uniform | 591 / 52 | -0.0004 | -0.0085 | 25 / 16 / 11 |
| MSVR310 | ambiguity | 591 / 52 | +0.0036 | -0.0052 | 27 / 15 / 10 |

| Dataset | Condition | Rank-1 outcome | Queries | AP gain / loss / unchanged | All-query mAP contribution |
|---|---|---|---:|---:|---:|
| RGBNT201 | uniform | still_correct | 654 | 218 / 116 / 320 | +0.4217 |
| RGBNT201 | uniform | still_wrong | 170 | 104 / 66 / 0 | +0.0541 |
| RGBNT201 | uniform | repair | 6 | 5 / 1 / 0 | +0.0533 |
| RGBNT201 | uniform | new_error | 6 | 0 / 6 / 0 | -0.0377 |
| RGBNT201 | ambiguity | still_correct | 653 | 231 / 99 / 323 | +0.3283 |
| RGBNT201 | ambiguity | still_wrong | 168 | 91 / 77 / 0 | +0.0093 |
| RGBNT201 | ambiguity | repair | 8 | 8 / 0 / 0 | +0.0540 |
| RGBNT201 | ambiguity | new_error | 7 | 0 / 7 / 0 | -0.0584 |
| RGBNT100 | uniform | still_correct | 1614 | 566 / 740 / 308 | -1.0175 |
| RGBNT100 | uniform | new_error | 43 | 1 / 42 / 0 | -0.4777 |
| RGBNT100 | uniform | still_wrong | 34 | 13 / 21 / 0 | -0.0876 |
| RGBNT100 | uniform | repair | 24 | 23 / 1 / 0 | +0.3348 |
| RGBNT100 | ambiguity | still_correct | 1611 | 583 / 723 / 305 | -0.9127 |
| RGBNT100 | ambiguity | new_error | 46 | 1 / 45 / 0 | -0.5548 |
| RGBNT100 | ambiguity | still_wrong | 27 | 12 / 15 / 0 | -0.0692 |
| RGBNT100 | ambiguity | repair | 31 | 31 / 0 / 0 | +0.3489 |
| MSVR310 | uniform | still_correct | 401 | 110 / 90 / 201 | -0.0045 |
| MSVR310 | uniform | still_wrong | 189 | 86 / 47 / 56 | +0.0083 |
| MSVR310 | uniform | new_error | 1 | 0 / 1 / 0 | -0.0041 |
| MSVR310 | ambiguity | still_correct | 401 | 113 / 87 / 201 | -0.0007 |
| MSVR310 | ambiguity | still_wrong | 189 | 83 / 49 / 57 | +0.0084 |
| MSVR310 | ambiguity | new_error | 1 | 0 / 1 / 0 | -0.0041 |
