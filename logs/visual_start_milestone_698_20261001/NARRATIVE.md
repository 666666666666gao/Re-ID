# Four complete visual-start endpoints and source-bound training audit

The actual intake at **2026-10-01T12:19:12.063708+08:00** found **four verified complete full-50 endpoints** and **two live RGBNT100 training processes**. GPU0 public-start PID2077472 and GPU3 ReID-start PID2077361 existed with their original commands. GPUs1/2 were free because their registered endpoints had completed, not because a missing training process was restarted. No pending endpoint or additional seed was created.

| Dataset | Frozen visual start | Best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | Public CLIP visual | 34 | 70.1618 | 72.3684 | 82.4163 | 86.4833 |
| RGBNT201 | Trained ReID visual | 2 | 72.5247 | 73.9234 | 82.7751 | 87.9187 |
| MSVR310 | Public CLIP visual | 24 | 48.0166 | 66.1591 | 81.2183 | 86.9712 |
| MSVR310 | Trained ReID visual | 10 | 52.4159 | 67.8511 | 83.2487 | 89.0017 |

All rows are full50 followed by strict selected-checkpoint reload and complete legal-gallery scoring. Public-minus-ReID descriptive differences are **-2.3629119942 mAP / -1.5550255775 R1** on RGBNT201 and **-4.3993049678 / -1.6920506954** on MSVR310. These two complete pairs do not show the hoped-for benefit from resetting the frozen visual initialization. The third pair is incomplete; the all-six automatic analysis and registered final verdict have not run. No partial result was used to change a gate, source, seed, configuration or stop the remaining endpoints.

The independently executed source audit at **12:16:05.648115+08:00** verifies 13 selected source bindings against all221 unchanged runtime sources. Its JSON SHA256 is `ffc66ad160ba1ad5987ba3efa4beda45f0ec083e5a893ab3003a056169efd1de` and is preserved under `raw/.codex_tmp/visual_training_boundary_20261001/SOURCE_AUDIT.json`. It performs source/shape checks only: no model construction, neural forward, optimizer update or retrieval evaluation.

Established boundaries:

* Our current visual path explicitly freezes all Signal parameters. M1 still changes the downstream shared representation through its hooks, and gradients can propagate through later frozen blocks to adapters. The six existing production M0s establish gradient support for the new parameters; the audit does not invent a whole-forward gradient cutoff.
* The pinned saved shape metadata contains152 frozen visual tensors with86,140,416 elements per dataset. Current trainable parameters are2,871,242/2,685,386/2,846,666 for RGBNT201/RGBNT100/MSVR310. This is a shape count, not evidence that capacity alone is the bottleneck or that every tensor would receive useful gradients in a new training mode.
* Signal's pinned author defaults use `MODEL.FROZEN=False`. Its optimizer assigns non-adapter CLIP-base parameters5e-6 when not frozen. The current role optimizer instead selects only `requires_grad` parameters in one AdamW group with the common five-epoch warmup/cosine schedule. Current batch64/K8 also differs from native Signal's batch128/K16 on RGBNT100 and64/K4 on MSVR310. Fifty epochs alone does not equalize these protocols or computation.
* Current checkpoints exclude all `backbone.signal.*` entries and training asserts the entire Signal hash is unchanged. A future visual-update control cannot be implemented solely by toggling `requires_grad`: it must explicitly group the visual parameters, save/reload their changed state, and retain immutability checks for the genuinely frozen state. This is an implementation requirement, not a newly validated method.

No visual fine-tuning experiment was launched; its performance causality remains untested. Ordinary fine-tuning should not be relabeled as algorithmic novelty, and any future role claim must separate it from matched global-only adaptation.

The immutable evidence archive is **316163 bytes**, SHA256 `b2ba7a1373041649a5389404d8a10fd638530b8fa3ae5019702885ce383dd769`, with **65 raw files** independently checked for length and SHA. All221 runtime sources remain unchanged. Free disk was **93,399,752,704 bytes**; no dependency weights were retired. Models, distance arrays and dataset images remain remote.

The prior publication697 actually synchronized at **12:01:55.904758+08:00**, HEAD `5eb72bc7cdc8f016b15df6e23da316fc9423928d`,64 raw blobs, document SHA `c35a52335160df54ef0fa7b13843dd290771418d6f08fccda02cd314e2040dde`. Its actual proof is archived here. The existing13:15 observer and single completion-only CPU waiter remain the continuation handles; the report still waits for all six verified endpoints. The original three-dataset baseline/SOTA objective remains **ACTIVE / UNMET**.
