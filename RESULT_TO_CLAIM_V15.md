# TriFusion V15 Result-to-Claim

- `claim_supported`: **no**
- `confidence`: **high**
- `integrity_status`: **warn_packaging_only; Q1 internal integrity passed**
- `routing_action`: `FAIL_TO_PROMOTE_SEAL_V15_NO_D1`

## What the results support

V15 is trainable and protocol-clean under the fit-only identity-OOF protocol:
three folds completed, all 110/110 trainable tensors received gradients in
every fold, frozen state remained unchanged, overflows were zero, and access
counters remained dev0/official0. Mamba alone has a positive aggregate matched
gain of `+0.2898 mAP`, and the CRDE fused output remains above the three CRDE
receiver outputs.

## What the results do not support

The intended stable collaboration claim is unsupported. Fold fused gains are
`+0.0952/-0.8311/+0.1605`; aggregate fused gain is `-0.1721`; CNN and
Transformer aggregate gains are negative; and the fused 95% bootstrap lower
bound is `-0.9503`. Five registered scientific conditions fail.

## Supported wording

> V15 CRDE is trainable and protocol-clean in fit-only identity-OOF RGBNT201,
> but its exchange does not reliably improve fused retrieval or all receiver
> branches over the same-fold no-exchange comparator. It is a sealed negative
> mechanism test, not stable collaboration evidence.

## Required action

Seal V15. Do not run D1, dev, official, ablation, multiple seeds, checkpoint
selection or hyperparameter scans. Only read-only analysis of existing
artifacts remains in scope. Any new experiment requires a new preregistered
main hypothesis.
