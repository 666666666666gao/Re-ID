# Independent experiment audit — TriFusion V10-Q0

Date: 2026-09-02

Overall verdict: **WARN**

Integrity status: **warn**

Scientific verdict: **FAIL_TO_QUALIFY / STOP_V10_Q0**

Low performance is not an integrity failure. The program completed, but the
scientific qualification gate failed.

| Dimension | Verdict | Conclusion |
|---|---|---|
| Ground truth / protocol / Oracle semantics | PASS | The loader uses only fit records with cross-camera identities. Real identity/camera GT drives retrieval. Oracle is diagnostic, not deployable. |
| Normalization / metric inflation | PASS | Fixed ImageNet input normalization, equal-block L2 composition and standard normalized Euclidean ReID scoring are executed; no score rescaling or post-result multiplier exists. |
| Result existence / consistency / provenance | WARN | JSON numbers and gate arithmetic are consistent. The result was untracked at audit time, and remote-only binaries cannot be rehashed locally. |
| Executed paths | PASS | Frozen Phase-B, strict DINO load, feature collection, token-shape check, metric, Oracle, state check and qualification gate are all on the main run path and covered by focused tests. |
| Scope / leakage / claim boundary | PASS with caveat | optimizer0, training false, dev0 and official0 are recorded. JSON `status=PASS` means probe completion only; `qualification_gate=false` is the scientific result. |
| Evaluation type | PASS | `real_gt_fit_only_diagnostic`; not synthetic, self-supervised proxy, simulation-only or human evaluation. |

## Claim impact

- Unsupported: DINOv2 provides fit-only complementarity to Phase-B. DINO has
  zero unique AP wins and zero Oracle gain.
- False under the fixed probe: equal-block concat improves Phase-B. It drops
  from 100.0000 to 92.2120 mAP.
- Not authorized: V10 implementation, capacity, training or dev access.
- Supported only as engineering evidence: frozen, strict-load, no-training,
  no-dev execution completed without mutating checkpoint states.

The result JSON must be tracked and V10 sealed. Do not use post-hoc DINO
resolution/block/multiplier/training-head/dev scans to rescue this failed gate.
