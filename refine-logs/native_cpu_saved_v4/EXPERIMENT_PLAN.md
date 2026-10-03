# Original-graph CPU saved-tensor engineering continuation

V1 failed at real full-author B128/K16 forward memory. V2 and V3 preserved
forward/loss but failed the unchanged production AMP gradient gate. One
preregistered semantic RGBNT201 original/original control actually passed
281 gradients (max2.9103830456733704e-11). This rules out its own failure on
that one draw; it does not establish universal determinism or a unique
operator cause for V3. All original failures and source293 remain sealed.

Keep the original CrossLayerAdaptedCLIP instance, bound forward, graph,
camera/modal order, capture hooks and module registration. Wrap only that
forward in torch.autograd.graph.save_on_cpu(pin_memory=False) during
gradient-enabled training. Role operators, Mamba, author heads and loss stay
outside. No activation checkpoint, replay, additional stream, compression,
CPU optimizer, new trainable parameters, fallback or modified precision.
This is a resource intervention, not a scientific contribution.

Use new v4 entry/campaign/report names. From public CLIP plus fresh camera,
heads and modules, run global_only/semantic/native independently on all3
datasets with the already registered author package, seed42 and full50.
Global-only has shared adapters. Preserve native159296/14 parameters,
independent detail QKV, original128-token semantic read, single zero exit,
original author optimizer/loss/BN mode, raw training features and L2_1536
deployment. No SIM/AlignM/ReID weights, local ID, EMA, N2/N3 or external data.

Before M0: all9 real initializers and all3 full-batch eval pairs; then
all3 original-vs-CPU-saved production AMP witnesses for each variant, using
the first32 samples of a real author batch solely for this comparison.
Original heads/loss, float16 autocast, GradScaler256, identical state/config
and CPU/CUDA RNG; no optimizer step. Fixed forward1e-5/gradient1e-4 and exact
gradient-None support, buffers/RNG/BNcount1/state/hook cleanup stay unchanged.
Write all measured deltas and ordered input metadata before fixed assertions.
If any check fails, preserve it and stop; no retry or gate/seed/precision
changes. No parity inferred from PyTorch documentation.

All9 M0 must then run8 actual updates on FULL author batches: 201B64/K8,
100B128/K16, MSVR B64/K4. Require finite effective updates, full intended
optimizer ownership, BNcount8, native all14 tensors changed by step8,
strict full-state reload within1e-5 and the existing production diagnostics.
Only after every M0 passes may all9 formal50 jobs run. No mixed historical
M0 receipts or early official scores authorize continuation.

Only2026 physicalGPU0-3, max4 single-GPU jobs, no preemption;2025 text only.
Existing environment unchanged. Keep the fixed whole-campaign storage
budget10292822016B and2GiB running reserve. Three exact sealed V1 probes
were retired under the persistent goal's prior authorization; original
receipts/hashes remain and their binary replay is explicitly retired.
Actual host availability was116887232kB before launch, not a guarantee.
CPU transfers add time/host-memory use, to be measured by actual M0 and
training; full B128 fit remains unmeasured. Queue polls240s, observer
180-300s during initial engineering, later estimated training milestones.

Scientific primary difference remains native-semantic, alongside semantic-
independent global_only. One mAP-best checkpoint supplies all metrics;
full legal query/gallery and dataset-specific filtering stay fixed. Phase
criterion mAP>=+.5 and R1 nonnegative is not statistical significance.
Report repairs/new errors, identity AP, full50 trajectories and real costs.
Native capacity differs; one seed and consumed official selection cannot
prove source causality, robustness, three-module effectiveness or SOTA.
Goal ACTIVE_UNMET. Obtain fresh code review before actual deployment.

Storage correction before the still-unstarted v4 campaign (2026-10-03):
two private pre-launch attempts failed the unchanged whole-campaign disk
budget BEFORE Popen. Exact8 additional verified closed engineering probes
were retired (2789830448B), with formalbest/distances/text preserved, but
/data free fell again before launch. Actual df at15:56:40 shows /data and
/home/gaob are separate devices; /home/gaob has591852425216B available.
Set this campaign's NEW18 M0/formal output directories explicitly under
/home/gaob/trifusion-native-evidence-v4, which must not already exist.
No old file is moved, no symlink or fallback is added, no source/data path
or scientific setting is changed. Check the same10292822016B budget on
the actual weight-output filesystem. Preserve2GiB running reserves on
both output and /data log/source filesystems. All workers, strict verifiers
and the final report use the same output_dir via configure; manifest
records output_root. Obtain a fresh source review of this correction.
