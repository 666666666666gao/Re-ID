# Visual initialization control: actual launch and M0

At 2026-10-01T11:18:35.612540+08:00 the six-end campaign is RUNNING: four real full50 training processes, two pending controls, no accepted formal endpoint. All four initial production M0 subprocesses exited0. Each had real finite/nonzero gradients on128/128 trainable tensors, unchanged frozen baseline and strict reload max difference0. This supports engineering readiness, not retrieval improvement.

| GPU | Dataset | Visual start | Full training PID |
|---|---|---|---:|
| 0 | RGBNT100 | public_visual | 2077472 |
| 1 | RGBNT201 | public_visual | 2077243 |
| 2 | MSVR310 | public_visual | 2077547 |
| 3 | RGBNT100 | reid_visual | 2077361 |

ReID-visual RGBNT201/MSVR310 remain registered and pending. The controller2074442 and durable wrapper2074440 were launched once at11:10:39; no prior experiment restarted. At the archived snapshot GPU memory was10889/10192/10893/11027MiB and utilization100/85/100/100%; disk free93,491,232,768bytes. This is a current health observation and does not diagnose the prior hardware fault.

The comparison changes only152 frozen visual tensors; all other initial state and trainable initial tensors match. It retains the trained camera/nonvisual state and existing role architecture, so it is not a wholly public-only model or a new algorithm. Preserve original pure-baseline weights, seed42, full50, complete gallery, camera/time filtering and one fused-mAP-best checkpoint per endpoint. No new losses, loader, readout, seed search or interim-score-based cancellation.

All221 runtime source hashes match. The original completed nine-end source closure remains unchanged. Forty exact raw text/JSON/log/source artifacts are preserved with per-file size and SHA; archive45,275bytes, SHA99fb602f544039578a8da223c11e272502da6ff1cf95ffa9897eab626a0382ed. Binary checkpoints, arrays and dataset images stay remote. The raw sync695 proof binds GitHub/remote source publication4ffeb990 and canonical handoff SHA7d5bb15f...; it precedes this launch.

The11:45 observer2089375 (wrapper2089370) is waiting on the estimated milestone, not repeatedly querying interim scores. The completion-only CPU waiter2089376 (wrapper2089371) checks parent status every240seconds. It will execute prepared report source SHAfaa00082ea3a4735181c18a0324e314ec4de5e3a177f84b487ef9e936edd9c13 only once after all six verified endpoints. The new report has passed actual AST parsing; it has NOT been invoked and has no results. Its diagnostics reuse fixed-array ranking, identity AP/negative-flip, complete loss-trajectory and cost analysis. All three pre-registered advancement comparisons are required; official-best/one-seed and retained-upstream-lineage limitations remain attached.

All six training/evaluation endpoints, CPU results and next evidence-based method decision remain outstanding. Overall three-dataset baseline/SOTA goal ACTIVE/UNMET.
