# MSVR310 SIM fixed-input operation diagnosis

Registered 2026-09-06T08:08:29.999652+08:00. PREPARED_NOT_RUN.

The completed full360 probe reproduces stored B0 exactly before wrapping; after wrapping,
all three paths agree with each other but differ only in SIM (max1.9073486328125e-6).
The first64 batch is chosen by original order, not retrieval utility; every row was already accessed.
No ranking, AP, Rank, training, optimizer, backward, checkpoint writes, backend scan or gate relaxation.

Exactly nine stages, all inference under no_grad with fixed B64 clean images and cached six CLIP inputs:
0 original Signal including CLIP/SIM; 1 repeat only SIM; 2 freeze SIM requires_grad only;
3 restore original flags; 4 import original V8 builder module; 5 import mamba_ssm;
6 construct original V8 roles, before loading role weights; 7 strictly load saved original fold0 epoch20;
8 original Signal including CLIP/SIM after loading. Record leaf/module inputs, outputs, strides,
storage offsets, parameter flags, backend flags and CPU/CUDA profiler operator counts/traces.
No model forward implementations or parameter values are changed. Reversible flag changes are
only diagnostic; the final original builder and checkpoint remain frozen and byte-state verified.

Budget: 576 SIM-record evaluations total; 128 of those inside two complete Signal forwards,
448 additional cached-input SIM forwards,64 distinct clean image records. No full-role forwards.
All tensor/image/model work and raw arrays/Chrome traces remain remote. Only scalar JSON/log/source
metadata returns locally. Original B0 parity is explicitly measured under instrumentation; if it
is absent, this probe cannot establish an operation cause for the uninstrumented failure.
No result-dependent stage selection, no new retrieval condition. Preserve original 260 training
updates and registered bitwise feature/distance gate. Any repair/resumption needs separate evidence
and registration; no fold0 retraining or blanket relaunch.
