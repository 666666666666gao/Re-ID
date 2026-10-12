# Endpoint907 R2 supplement: exact official scorer source

Verdict: **source-only PASS; zero new blocking findings.** R1 FAIL, R2 source-only PASS and the original 28 snapshots are preserved. This closes only the missing direct read of the author scorer; it adds no gate and accepts no neural result.

Reviewed file: `production_sources/comparators/Signal-cd1b0a6/utils/metrics.py`. Raw SHA256 `91604acb7d978462c16904910a645d7d1697c15bf150da529099004c8b5eecb2` exactly matches the existing463 seal. The parent supplied the cold copy; this reviewer used no SSH or production query.

The active functions `eval_func_msrv` (lines 13–108) and `eval_func` (111–170) match the inherited project scorer semantics:

- MSVR excludes same identity **and** same scene (68); camera scoring excludes same identity **and** same camera (137). Different-identity distractors remain, including those in the query environment.
- Full sorted gallery/filtered `orig_cmc` enters AP (82–99, 142–161). The top-100 display variable and top-50 rank-list/CMC slices do not truncate AP. R1/R5/R10 lie within that CMC range.
- AP is precision at each positive rank averaged over legal positives. CMC saturates cumulative hits at one and averages query hits. These are the same quantities as the inherited AP/first-positive-rank scorers. No metric uses prediction-statistic normalization.
- The upstream no-positive branch skips a query; project scorers assert a positive. The already verified sealed protocols have positives for every one of the 836/591/1715 queries, so this branch does not reduce their scope. Those passing protocol checks were not rerun.
- `run_correspondence_roles.py:93–102` directly calls these functions and compares against the independent project scorers with the existing `<1e-5` bound. Optional visualization/reranking classes are not called by this path. Upstream CMC is float32, consistent with preserving the established tolerance.

`SOURCE_ONLY_CHECKS_R2_SUPPLEMENT.json` records eight hash/AST/source-semantic checks. No source module was imported, no scorer/model/report executed, and no tensor payload read. Actual full50 performance, checkpoint/distance contents and producer/receipt execution authenticity remain unaccepted. Same-family/provisional; requested Astra/max remains distinct from the unattested actual serving identity.
