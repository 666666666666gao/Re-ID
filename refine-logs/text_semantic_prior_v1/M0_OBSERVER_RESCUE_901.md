**RESCUE verdict: REVISE_OBSERVER_ONLY — same-family / provisional.** The R201 pretrained and random endpoints failed the M0 acceptor at `tools/queue_text_semantic_prior.py:100`. The defect is in the numerical observation: its isolated role VJP omits the loss scaling used by the real backward. AMP underflow is strongly supported by the source and receipts; its exact first kernel has not been instrumented. There is no evidence here that the ψ/W task path is disconnected. The proposed repair is prospective and has not run.

This is a fresh rescue review of the first actual implementation attempt after R2 `SOURCE_ONLY_PASS`. Both packages expose the same runtime defect; these are not two failed repair attempts. Requested model/effort: gpt-6-astra/max; actual identity/effort: UNATTESTED. The earlier source review and the failed runtime acceptance remain separate records.

The primary R201 receipts agree on the following:

| Receipt observation | Pretrained | Random |
|---|---:|---:|
| Producer status | M0_PASS | M0_PASS |
| Acceptor | FAIL, line100 | FAIL, line100 |
| Actual optimizer updates | 8 | 8 |
| Trainable tensors with nonzero gradient during M0 | 286/286 | 286/286 |
| Strict reload maximum difference | 0 | 0 |
| Author BN batches tracked | 8 | 8 |
| Frozen text state tensors unchanged | 149 | 149 |
| Actual unscaled ψ/W gradients, steps2–8 | All5 positive on all7 steps | All5 positive on all7 steps |
| Isolated ψ.0/ψ.2 tensor gradients | All0 | All0 |
| Isolated W positive steps | 1, step8 | 0 |

The single nonzero isolated pretrained W value is `5.960464477539063e-08` (2^-24). Actual ψ.2.weight gradient maxima are `1.0781e-9` to `3.6248e-9` for pretrained and `4.1589e-10` to `1.5213e-9` for random. Every first-step ψ/W task gradient is zero as registered, while the context matrix has nonzero gradient and update. First-step ψ/W deltas are already nonzero, so deltas alone cannot establish task activity; the later nonzero `.grad` observations supply that evidence. They do not establish useful retrieval learning.

`training.json` and `text_updates.jsonl` contain exact duplicate actual-gradient values from separate post-step hooks. Initializer objects and their file hashes match their training receipts; both bind the exact published entry SHA `5cd50aefb158fd625a575e2add2930f93a9ca5f77135a7e64026dd753ebf3372`. Both packages have identical saved batch-order bytes. The shared prefix receipt passes all four package/template output and VJP comparisons, but is not a real-data M0 acceptance. Neither R201 receipt has a formal metric row or selected epoch. The loop's literal `epoch=1` is the eight-step sanity loop, not a completed formal epoch.

The relevant source chain is concrete:

- `run_text_semantic_prior.py:231–241` recomputes the role loss on the saved forward outputs and calls unscaled `autograd.grad`. Its later scalar zeros cause the exact `>0` gate to fail.
- `run_foundation_recipe.py:230–241` runs the model and losses under FP16 autocast, then performs `scaler.scale(loss).backward()`, unscales optimizer gradients, checks finiteness, steps, updates the scaler, and asserts the scale did not decrease. Hooks in `run_text_semantic_prior.py:190–196` observe those unscaled gradients.
- `run_global_task_role.py:25–31` returns `global_loss + fused_role_loss`. `global_task_role_heads.py:20–32` routes the global objective through shared-global features and original heads; ψ/W only affect the fused correction. The real ψ/W gradient therefore cannot be explained by a separate global-loss path.
- `text_semantic_prior.py:117–133` keeps ψ/text/W in FP32 and preserves autograd on the augmented context. `role_global_tokens.py:30` then applies `context_queries`, an AMP-eligible linear operation outside that FP32 region. `slot_competition_fp32_roles.py:10–20` starts another FP32 region afterwards. FP32 leaf parameters and the two FP32 regions do not make the intervening backward path FP32. No intermediate dtype/gradient dump was supplied, so the precise first zeroing operation is an inference, not a measured localization.

The actual M0 scale is **256 throughout all eight steps**, derived as follows. The bound foundation source creates a fresh `GradScaler('cuda', init_scale=256.0)` at line198; it neither restores scaler state nor sets a manual scale. The installed library source, supplied locally at `text_semantic_prior900/runtime/grad_scaler.py`, has SHA `eb4bc91fda9815ba40206e2dbb68f9845b85622fee3a92611f227b5845b20780`. Its lines119–147 set growth_interval=2000, growth_factor=2, backoff_factor=0.5, enabled=True, and initial growth tracker=0. Eight iterations cannot trigger growth; the completed finite-gradient/no-decrease checks and eight actual step hooks exclude a skipped/backoff step. The old logs do not directly contain `get_scale()` values; this conclusion is a source-and-receipt derivation.

The smallest correction belongs only in `run_text_semantic_prior.py`'s M0 branch:

```python
diagnostic_scale = 256.0  # Derived actual scale for the fixed fresh8 M0.
gradients = torch.autograd.grad(
    role_loss * diagnostic_scale,
    [output['shared_global']] + parameters,
    retain_graph=True, allow_unused=True,
)
# Keep the existing None/finite checks and shared-global None assertion.
values['text_isolated_task_gradient_max_abs'] = {
    name: float((g.detach().float() / diagnostic_scale).abs().max())
    for name, g in zip(TEXT_DIAGNOSTICS.parameters, gradients[1:])
}
```

Scaling must happen before the isolated backward; multiplying its already-zero returned gradients would not repair the observation. Divide returned gradients in FP32 before reporting them. Return the original training loss unchanged. Preserve the shared-global `None` assertion, all original acceptance checks, the strict positive threshold on steps2–8, first-step zero expectation, actual optimizer observations, exact eight steps, and all training settings. One optional receipt field can name the diagnostic coefficient, explicitly as derived rather than directly observed. The fixed-eight-step proof makes scaler capture, a factory monkeypatch, a proxy, or any historical429 source edit unnecessary. This constant is not a proposal for formal-training diagnostics or a different training scaler.

Do not relabel either old acceptor failure as PASS. Line100 prevents later checks at lines101–109 from executing, so this review does not assert that the full old acceptor would otherwise have passed. Preserve the failed logs, checkpoints and receipts. Do not mutate the active source433 campaign: allow its original fixed six jobs and report to close. If later endpoints pass and train under the original campaign, do not repeat them. Only missing endpoints may receive the reviewed diagnostic revision in new named M0 runs; each must actually pass the original complete gate before its own fresh50. This rescue verdict does not replace that revised-source review or runtime validation.

Validation was local stdlib parsing/arithmetic and source inspection: thirteen extracted files exactly match their intake ZIP, all sixteen actual update observations were checked, and twelve decisive source files match both the existing scope hashes and Git HEAD `7ab14d15e575ce35c5427278f3517970ac38465d`. Of the broader433 scope, this checkout has299 exact files,27 CRLF-only differences and107 absent author/protocol files; no full433 local-byte attestation or runtime recheck is claimed. The installed scaler source was inspected without importing torch. No SSH, GPU/NN execution, installation, source edit or subagent was performed by this reviewer. Only these requested rescue reports were written.
