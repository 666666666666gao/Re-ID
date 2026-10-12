# Closed raw semantic versus text reader weights — 901

Actual CPU inspection: 2026-10-10T04:14:49.463369+08:00. All three checkpoints are their own fused-mAP-best at epoch 8, physically SHA256-matched before CPU loading. The old raw checkpoint is protected by the unchanged input seal. Scalar receipt SHA256: 10d395e8f6440bbbb384bff20f6c5a3d797b45ba2334edac1737db8fadb14212.

The smaller Transformer/Mamba Q-K operators already occur in the no-text raw semantic control. The raw spectral norms are 0.00243987 / 0.00247092, versus 0.0561449 for CNN. The corresponding text-package operators are very close to these raw values. This weakens a text-specific explanation of the small operators; it does not establish their unique training cause.

| Text condition | Role | Q parameter difference / raw Q norm | K parameter difference / raw K norm | V parameter difference / raw V norm | Output parameter difference / raw output norm |
|---|---|---:|---:|---:|---:|
| pretrained | CNN | 19.77706% | 18.67720% | 6.21445% | 6.95971% |
| pretrained | Transformer | 0.08959% | 0.09912% | 6.28682% | 7.75246% |
| pretrained | Mamba | 0.08297% | 0.09149% | 6.77490% | 8.32133% |
| random | CNN | 20.44101% | 17.48789% | 5.76569% | 6.35161% |
| random | Transformer | 0.05913% | 0.06586% | 5.45999% | 6.39076% |
| random | Mamba | 0.05673% | 0.06792% | 4.59190% | 5.81739% |

Relative differences use each raw control matrix Frobenius norm as denominator. They are parameter comparisons, not ranking-effect estimates. Text changes CNN Q/K by roughly 17–20% in this measure; T/M Q/K changes are only about 0.06–0.10%. T/M V/output matrices differ more, about 4.6–8.3%. Thus the observed near-identical Q/K operators must not be extrapolated to equality of every role feature or the whole model.

These runs share a recorded common initialization digest and matching original sampled paths/labels/cameras. Full augmentation-byte equality and complete parameter trajectories were not independently witnessed. Weight equality at initialization, numerical root cause, raw-control per-image attention, and epoch-50 matrices were not newly evaluated here.

The current Transformer/Mamba outputs also receive image-dependent CNN/Transformer bridge messages. Small direct-read operators do not prove those entire roles are disconnected, non-discriminative or unnecessary. A future operator ablation must remove its full action path and retrain; this CPU comparison is not that ablation.

The next method must address how evidence is actually used. This observation gives no justification for blind text stacking, magnitude tuning or changing the completed results. It does not demonstrate usefulness/novelty of the unregistered patch-teacher proposal. No forward, GPU operation, active-progress query, production edit, new experiment or weight deletion occurred.
