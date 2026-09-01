# TriFusion V9 train-only readiness

Date: 2026-09-02

Hardware: one remote NVIDIA RTX 3090 24 GB

Seed: 42 only

Code identity: `b40b171beb0ec14e0f79c4e7c32580ee8cf3bb2b`

Dev access: 0

Official-test access: 0

## Registered representation hypothesis

V9 freezes exact Signal, the V8 pretrained-tail CNN/Transformer/Mamba experts,
and the Phase-B quality Router. Each expert receives two rounds of messages
constructed only from the other two experts. Every message is projected onto
the receiver's orthogonal complement before quality-gated injection. A
triadic head then combines the three enhanced experts and all three pairwise
products into a new identity residual. The complete 7680D Phase-B embedding is
the exact prefix of the 9216D V9 fused representation.

## Readiness gates

| Gate | Result |
|---|---:|
| Public-seam TDD plus adjacent V8 regression | PASS; 12 tests |
| Real train-only preflight | PASS |
| Total / trainable parameters | 107,301,905 / 6,840,577 |
| Maximum relay/receiver absolute cosine | 2.98e-8 |
| Signal / Phase-A / Router state unchanged | yes |
| Real capacity | PASS; B64/K8; 8 steps |
| Capacity gradient tensors | 59 / 59 |
| Capacity AMP overflow | 0 |
| Capacity peak allocated / reserved | 1426.80 / 2020 MiB |
| Fixed real-batch overfit | PASS; 100 steps |
| Overfit loss | 3.78850 → 0.61228 |
| Label-smoothing excess-loss ratio | 0.000518 ≤ 0.10 |
| Overfit gradient tensors / overflow | 59 / 59; 0 |

All three gates use only the 141-fit training side. No dev loader is iterated
and no official-test data is accessed. The readiness result authorizes the
single registered seed42 60-epoch training run and one final-checkpoint dev
evaluation; it is not evidence of retrieval improvement.

## Frozen main gate

The final V9 fused mAP must be at least 65 and strictly exceed the same
checkpoint's exact Signal baseline, frozen V8 Phase-B fused output, and the
V9 CNN/Transformer/Mamba collaborative expert outputs. Failure closes official
test, ablations, multiple seeds, and SOTA claims.

## Evidence

```text
evidence/trifusion_v9_preflight_seed42.json
evidence/trifusion_v9_capacity_seed42.json
evidence/trifusion_v9_overfit_seed42.json
```
