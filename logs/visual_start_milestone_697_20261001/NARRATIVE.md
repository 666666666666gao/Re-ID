# Visual-start control: actual 11:45 milestone and 11:51 intake

The durable 11:45 observer actually completed at 11:45:00.249274+08:00, and its wrapper waited and exited 0 at 11:45:00.263710+08:00. The immutable snapshot SHA256 is `4558fcdaced768adc0ebb624262ec4ddb9a80c04ac32d351eb7f193f164b6b69`.

The new evidence intake actually ran at 11:51:03.164990+08:00: **two complete endpoints, four running full-50 endpoints, no pending endpoints**. All six real-label production M0s passed, each with 128/128 nonzero-gradient tensors, unchanged frozen backbone and strict reload max difference 0. All 221 pinned runtime sources remain unchanged. Four GPUs had 10889/10918/10147/11027 MiB memory use and 100/100/97/100 percent utilization. Free disk was 93,421,002,752 bytes. This does not establish a repair diagnosis for the historical hardware failure.

| Public frozen visual start | Best epoch | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| RGBNT201 | 34 | 70.1618 | 72.3684 | 82.4163 | 86.4833 |
| MSVR310 | 24 | 48.0166 | 66.1591 | 81.2183 | 86.9712 |

These are complete, reloaded single-checkpoint full-50 endpoints, not interim bests. Their fresh matched ReID visual controls were still training. No paired initialization conclusion or advancement decision has been made; neither endpoint is a performance breakthrough.

Current live full-training PIDs: public RGBNT100/GPU0 **2077472**; ReID RGBNT100/GPU3 **2077361**; ReID RGBNT201/GPU2 **2127402**; ReID MSVR310/GPU1 **2135149**. At the intake their last logged epochs were 19/16/17/19 respectively. An estimate from elapsed time and logged epoch count put the slowest training endpoint around 13:15; it is an estimate and excludes later strict evaluation/verification.

The next durable observer was actually launched at 11:50:51.652016+08:00, wrapper **2148218**, for **13:15**. The existing completion-only CPU waiter remains the sole automatic reporter; no duplicate report was run. It checks every 240 seconds and must wait for all six verified endpoints before invoking the prepared report once.

The six M0s show public-start mean Triplet losses **0.4361900/0.4191416/0.4383485** for RGBNT100/RGBNT201/MSVR310 versus ReID-start **0/0.0056321/0.1828338**. Nonzero batches are public **8/8,8/8,8/8** versus ReID **0/8,7/8,8/8**. This measures initial source-task support, not retrieval improvement or optimizer-update share. The earlier four-end derivation is preserved separately and acknowledges correction from a nonexistent `ce` field to the actual `id` field.

Evidence archive: **194416 bytes**, SHA256 `c989c916154f70e888ae7aede1910ffa450abd72c0f513561f1bf227ec12483f`, **55 raw files** checked independently for bytes and SHA. Weights, distance arrays and dataset images remain remote. The archive contains the actual prior publication sync proof; it does not manufacture a current sync proof before synchronization.

Audit provenance supplement: original spawn timestamp/call ID/model/max/fork-none and actual reviewer model/effort were recoverable. The journal's message was an opaque encrypted string, **not a recovered readable prompt**. Only corrected call metadata is published; raw journal/message remain private. Earlier private recovery wording was overstated and is corrected, without changing the raw review or its **WARN**, same-family/provisional status. No audit rerun or verdict change.

The original three-dataset performance/SOTA goal remains **ACTIVE / UNMET**. All registered endpoints continue unchanged; no rescue seed, source mutation, new threshold, cancellation or reboot occurred.
