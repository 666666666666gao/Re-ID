# Original endpoint CPU diagnosis907

Status: source draft, not run. Original six-end queue906 remains unchanged and live. No replay, training/evaluator edit, new comparator at deployment, seed/gain search or model inference.

Run the one-end reporter only after that original job has four exit0 stages (prepare, real8 M0, fresh50, first strict evaluation) and job status COMPLETE. Other endpoints may still be running; they use different output directories. Missing or failed ends get no substituted score. This CPU consumer does not change the queue or authorize a failed stage to train.

`tools/report_region_use_endpoint.py` verifies original selected-best receipt, checkpoint/distance SHA, full50 trajectory, all recorded step/batch identities and exact old raw-semantic batch order. It reuses the established legal camera/scene AP/CMC scorer and complete-query paired analysis. Controls come from the historical fixed-best seal already pinned in current463/50 scope: raw-semantic824 and independent shared-adapter global794. Every consumed control/candidate distance is checked against its original receipt. No self-generated ground truth, reduced gallery or normalization by model statistics.

Report raw best mAP/R1/R5/R10 from the same weight, full curves/last, auxiliary loss and eligible/total query exposure and legal triplet coverage by epoch, identity AP changes, first repairs/new errors and their contributions to total mAP change. Also keep AP contributions for first-still-correct/first-still-wrong groups and the worst ceiling(5%*queries) AP-damage tail. Group contributions must sum to the original mAP delta, repairs minus new errors must reproduce R1 delta. This fixed-model description is not training-seed uncertainty or a unique failure explanation.

Record producer wall time including construction, training-plus-evaluation/save interval from actual timestamps, and training-loop seconds separately. Loop seconds omit subsequent evaluation/save. Capture measured memory of both partitions. No temperature/power action. GPU visibility is empty for this CPU consumer.

First consumer is201 uniform; later invoke once for each accepted original endpoint. No automatic repeated report or overwrite: each has a distinct output directory. Uniform-versus-ambiguity and same-model global/residual/actual-region decomposition are separate later consumers, not invented in this fused-distance report. A low fused score cannot be rescued into a late-comparison main result. All-three >Signal goal remains unmet until independently verified original results establish it.

Invocation (existing warm environment, after actual first acceptance):

```text
CUDA_VISIBLE_DEVICES='' /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/report_region_use_endpoint.py --campaign /data/gaob/Re-ID/Trifusion/logs/region_use_v1_20261012_906 --dataset RGBNT201 --comparison uniform --output-dir /data/gaob/Re-ID/Trifusion/logs/region_use_endpoint_diagnosis_20261012_907/RGBNT201_uniform
```

Source review and plain CPU fixture arithmetic precede first use. They do not demonstrate neural performance. The running463 source seal is not widened or modified by this separate report source/metadata publication.
