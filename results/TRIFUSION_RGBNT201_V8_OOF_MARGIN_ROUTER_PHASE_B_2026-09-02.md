# TriFusion V8 OOF-margin Router Phase-B result

Date: 2026-09-02

Hardware: one remote NVIDIA RTX 3090 24 GB

Seed: 42 only

Protocol: RGBNT201 fixed 141-fit / 30-dev identity split

Official-test access: 0

## Outcome

The frozen-expert hierarchical Router produces the first V8 deployable fused
output that strictly beats the exact Signal baseline and all three fixed
experts on held-out dev. The gain is nevertheless too small for the frozen
main gate: fused reaches 58.4050 mAP, not the required 65 mAP. The experiment
therefore stops before HFER, ablations, multiple seeds, or official test.

## Continuous OOF target repair

The original OOF per-query AP labels saturated near 100%. The replacement
target is the continuous identity margin

\[
d_{nearest\ negative}-d_{farthest\ positive}.
\]

It reuses the three already-trained identity-OOF expert checkpoints and does
not execute an optimizer step or access dev/official data.

| Target diagnostic | Result |
|---|---:|
| OOF queries | 571 |
| CNN / Transformer / Mamba unique slot winners | 38 / 350 / 183 |
| RGB / NI / TI unique slot winners | 215 / 59 / 297 |
| Best fixed slot mean margin | 0.1534070 |
| Slot Oracle mean margin | 0.3177096 |
| Oracle minus best fixed | +0.1643026 |

The target-diversity gate passed. This only authorized Router training; the
Oracle uses labels and is not a deployment result.

## Router training

The Router predicts

\[
w_{e,m}=P(m\mid x)P(e\mid m,x)
\]

and a bounded sample residual gate \(\alpha(x)\in(0,0.5]\). Three
identity-disjoint Router folds were each trained for 100 fixed epochs, followed
by one 100-epoch refit on all eligible fit identities. Phase-A expert and
Signal parameters were frozen throughout.

| OOF Router gate | Learned | Fixed/majority | Passed |
|---|---:|---:|:---:|
| Expected identity margin | 0.1020340 | 0.1017202 | yes |
| Top-slot accuracy | 17.8634% | 17.6883% | yes |

The pass is narrow and must not be described as strong OOF generalization.
All three actual single-modality blur tests decreased their own modal mass:

| Corrupted modality | Clean mass | Corrupted mass |
|---|---:|---:|
| RGB | 0.306154 | 0.117502 |
| NI | 0.298051 | 0.102016 |
| TI | 0.395795 | 0.166562 |

Missing-modality maximum mass is exactly 0. The run used 400 Router optimizer
steps, 3.4 GiB peak reserved memory, dev access 0, official access 0, and left
the complete Phase-A model state SHA unchanged. The combined checkpoint SHA is
`6f95f99a86763580c3bd8592974347825659a5336f9afec43062516d21fbfe02`.

## Frozen held-out-dev evaluation

The combined checkpoint was evaluated exactly once with no optimizer or model
selection. It emits baseline-only, fused, CNN, Transformer, and Mamba from the
same checkpoint.

| Output | mAP | Rank-1 | Rank-5 | Rank-10 | mAP delta vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline_only | 58.0109 | 57.4545 | 69.9394 | 76.6061 | — |
| fused | **58.4050** | **59.3939** | **71.2727** | 76.6061 | **+0.3941** |
| CNN | 57.6071 | 56.4848 | 70.9091 | **77.5758** | -0.4038 |
| Transformer | 56.3031 | 55.8788 | 69.6970 | 76.2424 | -1.7077 |
| Mamba | 56.6260 | 54.4242 | 68.8485 | 75.1515 | -1.3849 |

Fused strictly beats baseline/CNN/Transformer/Mamba in mAP, and improves over
baseline by +1.9394 Rank-1 points. It remains 6.5950 mAP below the 65 gate.
The promotion gate is therefore false and `next_phase_authorized=false`.

The learned dev Router has alpha mean/min/max
`0.329553/0.244704/0.411128`. Mean modal probability is
RGB/NI/TI=`0.322938/0.362106/0.314956`; mean conditional expert probability is
CNN/Transformer/Mamba=`0.162290/0.507904/0.329806`.

## Execution integrity and failed starts

- Trainer attempt `568c499` collected fit-only quality features, then stopped
  before the first Router optimizer step because the Signal repository had
  occupied the top-level `modeling` namespace. The two imports were corrected
  to the registered project `trifusion` namespace.
- Dev-evaluator attempt `208c679` stopped before runtime construction and
  before dev access because it imported `trifusion` too early. The import was
  moved after project namespace registration.
- The successful Router run is code identity `81dcb70`; the successful frozen
  dev evaluator is `4070d1e`.
- Successful dev access count is 1, official-test access count is 0, optimizer
  steps during dev evaluation are 0, and Phase-A/Router state SHAs are unchanged.

## Claim boundary

Independent result-to-claim review returned `partial` with medium confidence.
Supported narrowly: on this single seed42 held-out-dev protocol, the complete
fit-only OOF-margin, controlled-quality and hierarchical-routing configuration
produces a frozen fused output that improves the exact Signal baseline and
every fixed expert.

The gain cannot be attributed to the learned Router alone. The evaluator uses
the learned probabilities together with soft fusion and a sample-level alpha,
while the learned OOF advantage over the fixed policy is only `0.000314` mean
margin and `0.1751` percentage points of Top-slot accuracy. The evidence thus
supports a small deployment gain for the complete Phase-B configuration, not
the stronger claim that the Router has already learned to exploit most of the
available complementarity.

Not supported: 65 mAP, HFER effectiveness, official-test improvement, SOTA,
multi-seed robustness, or cross-dataset generalization. Because the unchanged
formal gate fails, HFER, ablations, multiple seeds, and official test remain
closed.

## Independent integrity audit

The V8 Phase-B audit returns `WARN`. Ground-truth provenance, normal ReID
normalization, live executed paths, fit/dev separation, and evaluation-type
classification all pass. The warning is limited to packaging: large
checkpoint/cache artifacts remain remote-only and cannot be independently
re-hashed from a fresh local clone. During sealing, the remote Phase-A and
combined checkpoint SHAs were re-checked and matched their receipts. See
`EXPERIMENT_AUDIT_V8_PHASE_B.md` and `.json`.

## Evidence

```text
evidence/trifusion_v8_oof_router_margin_targets_seed42.json
evidence/trifusion_v8_oof_margin_router_phase_b_seed42.json
evidence/trifusion_v8_oof_margin_router_dev_seed42.json
EXPERIMENT_AUDIT_V8_PHASE_B.md
EXPERIMENT_AUDIT_V8_PHASE_B.json
```

Remote large artifacts remain under
`/root/autodl-tmp/trifusion-v2/artifacts/` and are bound by the SHA values in
the evidence JSON files; they are not published in Git.
