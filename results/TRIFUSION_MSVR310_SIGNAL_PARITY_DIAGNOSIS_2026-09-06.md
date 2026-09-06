# MSVR310 Signal feature parity diagnosis

Recorded 2026-09-06T08:08:29.999652+08:00. Full360 four-path probe completed once, exit0,
execution15ffddce0e4db78907b645081577fba0a1ded605, elapsed29.83202269859612 seconds.
Original checkpoint b8a85e167861c51bb7d9a5854d700d11468ee9ca2a6e7ba21b75ce557130003c unchanged.

Original standalone Signal before wrapping exactly reproduces all360 stored B0 features.
After wrapping and loading original final roles, standalone Signal, hierarchical baseline and
full-model baseline agree with each other; all differ from B0 in359604 elements,0/360 exact rows.
The first1536 direct dimensions remain exact. SIM last1536 maxabs1.9073486328125e-6,
meanabs over3072 dimensions1.9935917978376173e-8; maxrelativeL2 9.066239721122114e-8.
All six original batches (five64 plus40) show the same localization. All Signal/model states and
recorded backend flags are unchanged, all Signal modules eval and full fused Signal prefixes exact.
This isolates wrapping/build/loading or their execution effects for further diagnosis; it does not
identify a kernel or cause, and small differences do not waive the registered bitwise gate.

1440 record forwards/360 distinct original gallery records,0 updates/backward/checkpoint/ranking.
Raw probe_arrays.pt remains remote, SHA48f7be2f2140d0bed53e54f60f9f0444ad795b38cbb0cf5f21d71dd12e470c1f.
Full JSON evidence: evidence/trifusion_msvr310_trifusion_v1_baseline_parity_diagnosis_20260906.json.
Original first64 draft was never executed; this was one full360 diagnosis.

Next: fixed first64 cached-input nine-stage operation diagnosis registered before execution.
Plan refine-logs/msvr310_trifusion_v1/SIM_OPERATION_PARITY_DIAGNOSIS_PLAN_20260906.md.
All576 SIM record evaluations,0 optimization/ranking; no repair, tolerance change or fold0 retraining.
Original comparison remains stopped with no AP/Rank result and folds1/2 not started.

## Operation cause measured; repair verification registered

2026-09-06T08:19:47.876038+08:00: fixed nine-stage probe completed once on0be865b, exit0,
72.98739485256374 seconds;576 SIM-record computations,128 inside full Signal,448 cached SIM.
Initial instrumented original Signal exactly matches B0. Repeat SIM exact; freeze SIM only changes64113
SIM elements (max1.9073486328125e-6); restoring flags returns exact equality. Builder/Mamba imports
remain exact. Actual construction then freezing reproduces the same change; final checkpoint load adds
no difference. First mismatch is cross_attn with exact inputs and token selection. No weight value,
stride, storage pointer or dtype changes explain the flag-only pair. Two mm calls become bmm (6 to8).

PyTorch 2.5.1 should_fold explicitly inspects the small operand requires_grad even under no_grad;
noncontiguous projection inputs thus select different mm/bmm implementations after freezing.
Primary source: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/LinearAlgebra.cpp
and https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Linear.cpp ; raw source archived.
MHA native fastpath is separately blocked by non-self-attention. No blame assigned to checkpoint corruption.

Inference helper is registered for full360 verification: functional call with one detached projection
weight view under no_grad restores original dispatch metadata while leaving registered weights frozen.
The repaired whole baseline feature matrix AND210x360 distances must exactly match B0; all role/modal
residuals must remain bitwise unchanged.720 full-role forwards,0 optimization/ranking. Verification NOT_RUN.
Original comparison still stopped; no fold0 training repeated or scientific gate changed.

## Full verification passed; original comparison continuation registered

2026-09-06T08:26:05.110072+08:00: verification9eba027 passed all checks, exit0,30.378821736201644seconds.
All360 baseline3072D features and210x360 distances are bitwise equal to original B0. All role/modal
residuals and direct modalities are bitwise unchanged across original five64+40 batches; model/frozen/
Signal states and all registered parameter flags unchanged. No ranking,720 role forwards,0 updates.
Runtime torch git a8d6afb511a69687bbb2b7e88a3cf67917e1697e, version2.5.1+cu121.

R3 continues the original fixed comparison by reusing original fold0 checkpoint/training and these
verified complete features. Only untouched folds1/2 will train,520 new updates; total study780.
Original training/model/config/gates and old failed files remain unchanged. Full600 queries/1032 gallery,
no scientific early stop, no original fold0 reread/retraining, no M0 optimization repeated.
Plan refine-logs/msvr310_trifusion_v1/COMPARISON_RESUME_R3_PLAN_20260906.md; continuation NOT_RUN.
