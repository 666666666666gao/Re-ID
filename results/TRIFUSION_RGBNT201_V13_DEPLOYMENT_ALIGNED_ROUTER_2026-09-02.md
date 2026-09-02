# TriFusion RGBNT201 V13 Deployment-Aligned Router

## Terminal verdict

`Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE`

V13-Q0 proved that actual deployed fusion-path counterfactual utilities are
non-degenerate and transferable enough to justify one Router qualification.
V13-Q1 did not prove that the frozen deployment-feature Router can reliably
learn those utilities or improve identity-OOF replay. No final refit,
checkpoint, dev evaluation, official test, or ablation was authorized.

## Protocol and implementation

- Remote NVIDIA RTX 3090 only; seed 42.
- RGBNT201 fixed 141-fit identity registry, three folds, 571 cross-camera
  eligible records.
- Router input: frozen all-fit V8 Phase-A `direct_modal/modal_residual`.
- Teacher target/replay: complete-path identity-OOF Signal/expert features.
- Router: `P(modality|x) × P(expert|modality,x)`, hidden width 128.
- Fusion: exact Signal prefix plus blockwise routed residual bank at fixed
  `alpha=0.2`.
- Router training: 100 epochs/fold, LR `3.5e-4`, utility temperature `0.05`,
  quality-loss weight 1.
- Q1 hard gate: all four metrics non-inferior in every fold and all four
  identity-cluster bootstrap 95% lower bounds strictly greater than zero.

Public-seam RED→GREEN completed. The committed remote tree at
`46b3e993b732c3afee63af9a56c75a62b3dbae21` passed 19/19 V13 plus adjacent
V8/V12 tests.

## P1 preflight

- 8 real fold-0 records; required tensor shapes passed.
- Phase-A SHA before/after:
  `ecfd7fbcf4496fe83f58bd8cda288f163e21b2328b81f65308df246c1698fb77`.
- Elapsed `13.907s`; peak allocated/reserved `1225.70/1440 MiB`.
- Training false, optimizer steps 0, dev0, official0.

## Q0 actual-path target qualification

| Quantity | Result |
|---|---:|
| Queries | 571 |
| CNN unique positive wins | 218 |
| Transformer unique positive wins | 196 |
| Mamba unique positive wins | 157 |
| RGB unique positive wins | 241 |
| NI unique positive wins | 109 |
| TI unique positive wins | 221 |
| Oracle mean utility | 0.0020423282 |
| Best fixed mean utility | 0.0005741757 |
| Oracle-minus-fixed | **+0.0014681525** |
| Read-only action-transfer gain | **+0.0008705698** |

All three action-transfer folds are non-inferior. The immutable-reference,
expert/modality diversity, Oracle gain, action-transfer, and access-boundary
gates pass. Q0 ran no training and accessed neither dev nor official test.

Paired cache SHA-256:
`1cc499a1acb7b12336f19de0e74ad4ef452dae8b2aa8299e4a16e2d619e15e27`.

## Q1 policy and OOF replay

| Fold | Fixed slot | Utility gain | Top-1 gain | Replay AP gain | Replay margin gain | Per-fold result |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | +0.0000726 | **-0.0210526** | +0.0041573 | +0.0005697 | FAIL: Top-1 |
| 1 | 2 | +0.0004395 | +0.0111732 | +0.0011984 | +0.0056176 | PASS |
| 2 | 6 | **-0.0003723** | +0.0742574 | **-0.0039748** | **-0.0023345** | FAIL: utility/AP/margin |

Aggregate identity-cluster bootstrap results use 21 identity clusters, 10,000
resamples, and fixed statistical seed 42:

| Paired learned-minus-fixed quantity | Observed mean | 95% lower bound | Gate |
|---|---:|---:|---|
| Expected utility | +0.0000302 | **-0.0004691** | FAIL |
| Top-1 correctness | +0.0227671 | **-0.0396049** | FAIL |
| Replay average precision | +0.0003529 | **-0.0081192** | FAIL |
| Replay margin | +0.0011247 | **-0.0028545** | FAIL |

The positive point estimates are not promoted because every registered lower
bound is negative.

## Quality, resources, and provenance

| Modality | Clean mass | Corrupted mass | Gate |
|---|---:|---:|---|
| RGB | 0.338163 | 0.111049 | PASS |
| NI | 0.331485 | 0.119141 | PASS |
| TI | 0.330352 | 0.115227 | PASS |

Missing-modality maximum mass is exactly zero. Phase-A state SHA is unchanged.
Q1 executed 300 Router optimizer steps in `34.260s`, with peak
allocated/reserved `2459.15/3400 MiB`. Dev and official access counts are zero.

Q1 config SHA is
`4baca75b39bbb442b0558a308dc2874818f53b9a026202dc62713268320baa08`; runner
SHA is `e181971f284473d528ce43796cda27411c15d6bcec04fe659ad5e39c73d544ec`;
source diff SHA is the empty-tree digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Result-to-claim boundary

Independent result-to-claim is `no/high`. Integrity audit is `WARN/warn`; its
warning concerns remote binary-artifact packaging and tracker staleness, while
ground-truth provenance, normalization, executed path, scope/leakage, and
evaluation classification pass.

The only supported statement is:

> V13 formed a Q0-qualified actual-path OOF utility target, but the fixed
> deployment-feature Router did not reliably learn it or produce reliable OOF
> replay gains.

Current deployable best remains V8 Phase-B at `58.4050 mAP / 59.3939 Rank-1`,
which is `6.5950 mAP` below the 65 mAP dev gate. V13 adds no dev metric and does
not support an official or SOTA claim.

## Evidence

- `evidence/trifusion_v13_deployment_aligned_preflight_seed42.json`
- `evidence/trifusion_v13_deployment_aligned_q0_seed42.json`
- `evidence/trifusion_v13_deployment_aligned_router_q1_seed42.json`
- `EXPERIMENT_AUDIT_V13.md`
- `RESULT_TO_CLAIM_V13.md`

