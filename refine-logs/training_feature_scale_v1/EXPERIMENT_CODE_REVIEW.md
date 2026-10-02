# F2 code review - incomplete

Status: REVIEW_UNAVAILABLE / SAME_AGENT_REVIEW_IN_PROGRESS, not approval.
The same native gpt-6-astra/max review returned three actual capacity errors.
Original requests and errors are retained privately; the fourth continuation
is reviewing the demonstrated protocol correction, not a new model/verdict.
Partial source inspection is not a completed review. No executor PASS.

The reviewer identified a real prelaunch blocker: F1/249 canonical protocol
hashes refer to 2025 roots, while the preserved 2026 canonical protocols use
the original SEALED_SOURCE243 hashes. The minimal source_map correction checks
those three exact original hashes, retains strict F1/249 hashes for the other
246 sources, and records actual target hashes. Neither old manifest nor
canonical protocol is modified. Full review of the corrected source is pending.

Actual read-only target checks also found the sparse checkout had omitted the
old F1 manifest and three F1 reference files. The committed task directories
have now been restored and persisted in sparse checkout. Source and data
presence pass; these checks are not model/initialization/M0 results.

Local Codex policy says: "If the requested Codex reviewer cannot start, record
REVIEW_UNAVAILABLE with the actual error." It also requires continuing only
work that does not depend on that result. experiment-bridge Phase2.5 requires
the secondary source review before deployment. Same-family/provisional only.

No F2 initial forward, M0, formal training or scorer has run. The latest actual
GPU check found unrelated DeMo-DualAxis training on all four 2026 cards. Only
2026 GPU0-3/max4 is authorized; no preemption or new 2025 training.
