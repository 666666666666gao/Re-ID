# R2 runtime plan byte binding; no algorithm change

Original execution628cce0, wrapper167185/T0PID167187, 2026-09-07 18:15:01-18:15:06+08: stopped at T0 exact Python-dictionary equality. No GPU stage, model forward, training update or heldout image read occurred. Original pipeline/log/config and metadata remain preserved.

Full780batch replay in the actual NumPy1.24.4 environment found31 differing Float64 Beta coefficient scalars in31batches, maximum2.220446049250313e-16. Every actual Float32 coefficient used by mix_patch_statistics is bitwise identical; all donor, activation, fold, step, sampling/exposure fields are unchanged. This is a local-versus-server scalar arithmetic byte binding issue, not different augmentation data or a model result.

R2 config binds the complete server-generated metadata, keeps strict exact plan equality, and retains original failure evidence. No tolerance was loosened; no formula, sampler, source/heldout division, initialization, optimizer, training budget or scientific gate changes. All original T0/M0/CPU/Q1/CPU stages still mandatory. No partial previous training is reused because none occurred.
