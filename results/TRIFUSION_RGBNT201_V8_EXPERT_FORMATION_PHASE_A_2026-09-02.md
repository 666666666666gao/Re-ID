# TriFusion V8 pretrained-tail expert formation — Phase A

## Outcome

The Phase-A engineering and complementarity gates pass on the remote RTX 3090.
This result authorizes one frozen-expert, fit-only Router feasibility phase. It
is not a deployable main result and does not authorize HFER, ablations,
multiple seeds or official-test evaluation.

## Frozen experiment identity

- RGBNT201 fixed 141-fit/30-held-out-dev protocol;
- seed 42, B64/K8, AMP;
- exact frozen Signal 3072D baseline;
- branch point after pretrained CLIP block 8;
- frozen shared CLIP tail blocks 9/10/11 in every expert;
- CNN horizontal local-detail head, Transformer global-CLS relation head,
  Mamba spatial and aligned cross-modal long-range head;
- Router and HFER disabled;
- 20 epochs, final checkpoint only, zero dev evaluations during training;
- no reranking and zero official-test accesses.

## Engineering gates

| Gate | Result | Evidence |
|---|---:|---|
| exact preflight | PASS | 3072D tensor equality; 58.0109/57.4545 baseline parity; optimizer 0 |
| B64/K8 capacity | PASS | 8 steps; 203/203 gradients; 0 overflow; 6006 MiB reserved |
| fixed-batch overfit | PASS | 4.1156→0.6125; excess-loss ratio 0.000534≤0.1 |
| 20-epoch formation | PASS | 840 steps; 0 overflow; Signal SHA unchanged; 6214 MiB reserved |

Total/trainable parameters are `100,171,789 / 9,068,556`. The Phase-A run
took `933.24 s` on one RTX 3090.

## Raw fixed-output results

| Output | mAP | Rank-1 | ΔmAP vs baseline |
|---|---:|---:|---:|
| baseline_only | **58.0109** | **57.4545** | — |
| fixed equal-energy fused | 58.0972 | 56.8485 | +0.0863 |
| baseline + CNN residual | 57.6071 | 56.4848 | -0.4037 |
| baseline + Transformer residual | 56.3031 | 55.8788 | -1.7077 |
| baseline + Mamba residual | 56.6277 | 54.4242 | -1.3832 |

The fixed fused output is only a Phase-A diagnostic. Its tiny mAP gain and
lower Rank-1 do not demonstrate deployable collaboration.

## Query-wise diagnostic Oracle

| Diagnostic | Best fixed mAP | Oracle mAP | Oracle Rank-1 | Oracle gain |
|---|---:|---:|---:|---:|
| baseline + expert branch | 58.0109 | 64.7850 | 65.9394 | +6.7741 |
| residual-only experts | 53.8660 | 63.4813 | 66.9091 | +9.6153 |

| Expert | Branch unique AP wins | Branch leave-one-out ΔmAP | Residual unique AP wins | Residual leave-one-out ΔmAP |
|---|---:|---:|---:|---:|
| CNN | 201 | +1.2043 | 257 | +3.1128 |
| Transformer | 170 | +1.9592 | 232 | +4.9698 |
| Mamba | 138 | +0.8435 | 199 | +2.6370 |

The Oracle uses held-out labels and is not a deployable score. It establishes
that each expert contributes to different queries, but the branch Oracle is
still `0.2150 mAP` below the registered 65 mAP gate. Hard expert selection
alone therefore cannot meet the gate.

## Findings

1. **Observation:** all experts have many unique wins and positive
   leave-one-out margins. **Interpretation:** the pretrained-tail restructuring
   repaired the V7 fit-domain winner collapse. **Implication:** a Router now has
   a non-degenerate expert signal to learn. **Next step:** freeze the experts
   and train only a fit-only hierarchical Router.
2. **Observation:** fixed soft fusion remains near baseline and reduces
   Rank-1. **Interpretation:** complementary evidence is not yet converted into
   an inference-time representation. **Implication:** Oracle cannot be reported
   as the method result. **Next step:** require learned fused output to beat
   baseline and every fixed expert before enabling HFER.
3. **Observation:** branch Oracle is below 65 even though its gain is large.
   **Interpretation:** selection alone is insufficient. **Implication:** after
   Router feasibility, typed low-rate exchange must create additional useful
   representation rather than merely choose a branch.

No mean, standard deviation or significance claim is reported because the
project is intentionally restricted to one seed.

## Claim and integrity boundary

Independent result-to-claim review: `partial / medium`. Independent V8 audit:
`WARN`, with all metric/GT/leakage checks passing; the warning is limited to
packaging large remote artifacts by SHA/path. See
`EXPERIMENT_AUDIT_V8_PHASE_A.md`.

Supported wording:

> On fixed seed-42 RGBNT201 held-out dev, the V8 Phase-A pretrained-tail
> residual experts exhibit query-level complementarity, warranting one
> frozen-expert, fit-only Router feasibility phase.

Unsupported: learned Router quality, HFER effectiveness, deployable 65 mAP,
official-test improvement, SOTA, causal expert-role attribution or broad
generalization.

## Evidence

```text
evidence/trifusion_signal_preserving_v8_expert_formation_preflight_seed42.json
evidence/trifusion_signal_preserving_v8_expert_formation_capacity_seed42.json
evidence/trifusion_signal_preserving_v8_expert_formation_overfit_seed42.json
evidence/trifusion_signal_preserving_v8_expert_formation_probe_seed42.json
```

Remote artifact root:

```text
/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v8_expert_formation_probe_seed42_abbf33d
```
