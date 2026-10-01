# Source-bound notes for the next decision; no new design launched

Current full12 visual-update/readout panel and original C1/C2 must finish unchanged. These notes do not select a new model from partial official results.

The completed RGBNT201 low_lr roles and independently trained global-only both select epoch1. Their best-to-epoch50 mAP changes are -7.8971537324 and -8.4237464371 while mean training losses fall from4.9991529483 to0.8477877502 and from4.9999343944 to0.8487956198. The new panel has M3 and auxiliary identity supervision disabled. Thus late regression also exists without role computation or prediction losses; this observation does not identify a unique causal module. Fresh frozen controls remain required for the update-boundary comparison.

`CrossLayerAdaptedCLIP.capture` in `modeling/trifusion/correspondence_roles.py` returns `output + sum(deltas)/3` to the shared stream and records three snapshots. The independent global-only control retains the same initialized backbone, neck and classifier but omits the role readout (`tools/run_visual_update_control.py`). Therefore its global path still includes the averaged M1 adapters; it is not a zero-adapter plainCLIP baseline.

The user's proposed shared-global/private-role separation removes this particular mean-writeback channel. It does not mathematically freeze global behavior: the trainable shared backbone can still receive gradients from the fused objective. Keep ordinary shared-adaptation and matched-parameter controls if the proposal is selected after full12 analysis.

For the current bias-free regional linear readout, let c_r denote each role's direct readout contribution and c=sum_r c_r. With the actual learned `readout_gain` gamma, a direct-component removal is `normalize(g + gamma*(c-c_r))`. The expression `normalize(g+c-c_r)` only matches if c and c_r have already absorbed gamma. This is an algebraic source deduction, not an executed ablation.

Removing c_C only from the final readout leaves CNN messages inside Transformer and Mamba because the actual path is CNN -> Transformer -> Mamba. Consequently direct output-component removal must not be claimed as disabling the CNN computation or proving an independent role's structural necessity. Those claims need separate forward or independently trained structural controls.

V17's existing one-sided Signal relation protection and completed target-address/predictor controls remain direct neighbors. Reusing relative margin or teacher-correct protection is not itself a new contribution. Do not repeat old experiments under new module names.

Evidence read locally: `tools/run_visual_update_control.py`, `modeling/trifusion/correspondence_roles.py`, `modeling/trifusion/role_global_tokens.py`, `modeling/trifusion/correspondence_evidence_readout.py`, `modeling/trifusion/correspondence_context_identity.py`; completed text receipts in `logs/visual_update_milestone_702_20261001/raw/`. No model, tensor or distance replay, no optimizer or deployment change.
