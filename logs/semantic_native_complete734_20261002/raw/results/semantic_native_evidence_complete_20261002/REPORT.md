# EV1 full-six CPU report

| Dataset | Variant | Best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | semantic | 10 | 69.8596 | 71.6507 | 80.7416 | 85.4067 |
| RGBNT201 | combined | 10 | 69.2029 | 70.5742 | 80.3828 | 85.2871 |
| RGBNT100 | semantic | 12 | 79.1431 | 94.9854 | — | — |
| RGBNT100 | combined | 30 | 78.8446 | 95.1603 | — | — |
| MSVR310 | semantic | 24 | 52.6870 | 71.9120 | — | — |
| MSVR310 | combined | 24 | 51.7132 | 70.8968 | — | — |

| Dataset | Comparison | ΔmAP | ΔR1 | Repairs/new errors | Gate |
|---|---|---:|---:|---|---|
| RGBNT201 | combined minus semantic | -0.6567 | -1.0766 | 16/25 | EV-A: FAIL |
| RGBNT201 | combined minus roles | -0.6735 | -0.8373 | 31/38 | EV-B: FAIL |
| RGBNT201 | combined minus global_only | +1.0912 | +0.8373 | 35/28 | Descriptive only |
| RGBNT201 | combined minus native_high | +0.9014 | +1.3158 | 38/27 | Descriptive only |
| RGBNT100 | combined minus semantic | -0.2985 | +0.1749 | 48/45 | EV-A: FAIL |
| RGBNT100 | combined minus roles | -2.0785 | +0.8746 | 47/32 | EV-B: FAIL |
| RGBNT100 | combined minus global_only | +0.3533 | +1.2828 | 68/46 | Descriptive only |
| RGBNT100 | combined minus native_high | -0.5428 | -0.0583 | 35/36 | Descriptive only |
| MSVR310 | combined minus semantic | -0.9738 | -1.0152 | 17/23 | EV-A: FAIL |
| MSVR310 | combined minus roles | -0.2910 | +2.1997 | 39/26 | EV-B: FAIL |
| MSVR310 | combined minus global_only | -0.0092 | +3.7225 | 41/19 | Descriptive only |
| MSVR310 | combined minus native_high | +0.5631 | +2.8765 | 40/23 | Descriptive only |

Registered gates: {"EV-A": "FAIL", "EV-B": "FAIL"}.

- EV1 only: keep semantic CNN values and add image-native stride8 detail. N2/N3 not tested.
- New semantic/combined arms share all common initial states. Combined adds93248trainable parameters; no fake capacity match.
- Both new arms use512semantic keys and1536Doutput. No new fusion coefficient, loss, sampler or external resource.
- Full50 per endpoint, one official fused-mAP-best with later tied epoch, strict full-state reload, real GT/full gallery/original filtering/no rerank.
- Official benchmark used in development/epoch selection; single seed42 and fixed-model bootstrap do not prove unbiased or multi-seed stability.
- Prior clean controls are fixed matched-recipe records, not newly trained arms; EV-B also changes128to512candidates and capacity.
- Original N1 high/low failure remains sealed; no original training/evaluation/report retry. Missing lowMSVR remains unavailable.
- EV-A/B are intermediate gates, not three-dataset baseline/SOTA achievement, role necessity or novelty.
- Training/epoch-evaluation seconds exclude construction/M0/final reload/prior controls; not total GPU cost.
