claim_supported: no

what_results_support: The run supports only engineering viability of CRDE under
the fit-only OOF protocol: three registered folds completed at seed42,
dev0/official0, clean commit/diff, zero overflow, frozen SHA unchanged, and all
110/110 trainable tensors received nonzero gradients in every fold. It also
shows Mamba benefits on aggregate (`+0.2898` mAP points) and fused still beats
the on-exchange receiver branches.

what_results_dont_support: It does not support stable collaboration or Q1
qualification. Fused gains are `+0.0952/-0.8311/+0.1605`, weighted aggregate
is `-0.1721`, aggregate CNN/Transformer gains are negative, and bootstrap 95%
lower bound is `-0.9503`. Runner `status=PASS` is completion only.

missing_evidence: positive aggregate fused gain; nonnegative fused gain in every
fold; at least two positive receivers per fold; positive aggregate gains for
all receivers; positive bootstrap lower bound; Q1 gate pass. D1 evidence is
absent and unauthorized.

suggested_claim_revision: V15 CRDE is trainable and protocol-clean in fit-only
identity-OOF RGBNT201, but does not reliably improve fused retrieval or all
receiver branches. It is a sealed negative Q1 mechanism test.

next_experiments_needed: none within the frozen V15 boundary. Do not run D1,
dev, official, ablations, multi-seed, checkpoint selection or scans. Any new
experiment requires a new preregistered hypothesis.

confidence: high

integrity_status: warn_packaging_only; Q1 internal integrity gates pass

routing_action: FAIL_TO_PROMOTE; seal V15 Q1; D1 unauthorized; pivot to a new
main hypothesis rather than tuning CRDE.
