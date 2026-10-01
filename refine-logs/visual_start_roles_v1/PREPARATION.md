# Executed input and initial-state checks

`prepare_visual_start_inputs.py` executed once and exited0. `INPUTS.json` SHA256 `6fb9d6a1876a67447ad0b126f154397b6af9fd48e1c3b885a2d13de102cd347a` records all three reset weights and the unchanged original pure baselines. The three new weight files stay remote; only metadata is received locally.

`check_visual_start_initialization.py` executed once on the existing GPU0, exited0, and constructed both production architectures for all three datasets. All152 frozen visual tensors differ, every other initial model state is bitwise equal, and the trainable parameter counts match:201=2871242,100=2685386,MSVR=2846666. The ReID-side full initial hashes exactly match the previously completed STATIC controls. See raw `INITIALIZATION_WITNESS.json` and `INITIALIZATION_EXECUTION.log`.

This check performs model construction/state comparisons only. It is not a production input forward, nonzero-gradient check, real Mamba/AMP update, full training or retrieval result. All six real eight-batch M0s still precede their fresh full50 runs. Source syntax was parsed in the existing remote Python; local and remote source hashes match. No dependency/environment change was performed.

Initial free-GPU query for this preparation reported memory0/1/2/3=15/60/15/153MiB, all below500MiB. The queue checks availability again before reserving devices. Earlier hardware failure records remain; this observation does not establish their kernel root cause or administrative repair.
