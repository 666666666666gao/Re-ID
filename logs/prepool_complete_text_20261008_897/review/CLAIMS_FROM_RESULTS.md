# H2a claims from completed results

**claim_supported: no · integrity_status: warn · confidence: high for the scoped decision.** Same-family review; acceptance remains provisional. Requested gpt-6-astra/max; actual model and effort UNATTESTED.

None of the three primary pairs meets the registered +0.5 mAP with Rank-1 nondecline criterion: **0/3 advances**. Original receipts support three real M0s, three fresh50 runs, first strict evaluations and one complete CPU report: 150 formal epochs and 6,484 formal updates. Completion is receipt-backed; no independent binary replication was performed.

| Dataset | H2a best epoch | Formal mAP / R1 | Primary CPU ΔmAP / ΔR1 | R1 repairs / new errors |
|---|---:|---:|---:|---:|
| RGBNT201 | 8 | 74.3864 / 79.1866 | +0.0116 / +0.3589 | 6 / 3 |
| MSVR310 | 38 | 50.5468 / 68.0203 | +0.0046 / +0.1692 | 1 / 0 |
| RGBNT100 | 26 | 83.5272 / 95.8017 | −0.5631 / −0.0583 | 30 / 31 |

Formal scores and original CPU paired-query differences are separate provenance layers; CMC rounding agrees within the unchanged 1e-5-point tolerance. Primary comparisons use sealed raw-semantic controls. The threshold is a development decision rule, not a significance or equivalence test.

| Claim | Judgment |
|---|---|
| C1: useful three-dataset increment under the registered rule | **No.** Two small observed improvements and one regression; none reaches +0.5 mAP. |
| C2: completed endpoints; changed proxies distinct from retrieval | **Yes, qualified.** Original bound records support completion and lower first-to-last auxiliary means. |
| C3: RGBNT100 own-global versus global-best proves backbone harm | **No.** Epoch26 is being compared with epoch7. All four own-global metrics equal the sealed global curve at epoch26. |
| C4: low mapped L1/loss proves identity correspondence | **No.** Uniform attention can give zero mapped L1; normalized K–K and actual Q–K measure different quantities. |
| C5: stable seeds, three necessary novel mechanisms or SOTA | **No.** Those evidence requirements remain unmet. |

RGBNT100 fused versus own-global at the same selected checkpoint loses 0.4382 mAP and 0.7580 R1 points in the CPU report, with 10 repairs/23 new errors. This describes the selected output, without identifying a causal failure. Same-epoch aggregate metric equality does not establish weight, feature or whole-trajectory identity. Near-uniform late RGBNT201 role averages do not establish universal role collapse.

The attached integrity audit is **WARN, zero blockers**. It preserves four limits: one seed and consumed official development scores; local text validation without binary rescoring; checkpoint-selection scope; and proxy interpretation. Canonical evidence_check.py remains UNRESOLVED. File/SHA checks establish existence only. Zero new parameters does not mean zero extra training computation.

Use the revised claim: “This fixed same-modal geometric auxiliary reduced its logged loss but did not meet the registered retrieval requirement in three complete seed42 development runs.” Close this implementation and retain all negatives. No further experiment is needed for that decision; B3 is not triggered. This does not reject every broader correspondence hypothesis.

Future mechanism, stability or SOTA claims require a separately registered identity-linked intervention, actual Q/K norm/logit evidence, paired complete-pipeline seeds and prospective held-out/resource-matched comparisons. Already closed controls remain closed. This review authorizes no new algorithm; the broader goal remains ACTIVE_UNMET.
