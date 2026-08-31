# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R000 | M0 | RGBNT201 完整性 | audit | all | ids/triplets/JPEG | MUST | DONE | 171/141/30；0 pairing/JPEG errors |
| R001 | M0 | CUDA 环境 | torch smoke | synthetic | CUDA/finite/memory | MUST | DONE | torch 2.5.1+cu121 |
| R002 | M0 | loader | MDReID official loader | RGBNT201 | shapes/labels | MUST | DONE | 三模态真实 batch |
| R003 | M0 | DeMo 图 | forward/backward | synthetic | loss/322 gradients | MUST | DONE | 322/322 finite |
| R004 | M0 | evaluator | fixed worked example | synthetic | exact mAP/CMC | MUST | WAITING | 等精确“接缝同意” |
| R005 | M0 | MSVR310 完整性 | audit | all | ids/pairing/JPEG | MUST | DONE | 310 identities；0 errors |
| R006 | M0 | 查新/协议/SOTA | primary-source audit | 2024–2026 | claims/resources | MUST | DONE | novelty PROCEED WITH CAUTION；implementation-readiness PASS |
| R006A | M0 | 增量 SOTA/代码审计 | 2026-08-31 primary-source sweep | current public | metrics/protocol/artifacts | MUST | DONE | Signal/PEFT-BoA/FUSE/MGRNet/ICPL 均未超过 RoDI-CLIP 84.1/87.2；Hyper-ReID 仅占位 README，转 R071A 复查 |
| R006B | M0 | baseline 许可证审计 | pinned Git trees + GitHub primary sources | current public | license/tree/hash/role | MUST | DONE | Signal=最高指标同输入 MIT baseline；DeMo=MIT 实现底座；MDReID/PEFT/MFRNet/UGG 均无仓库级许可证，不再称开源 |
| R007 | M0 | Mamba CUDA | SM89 build + smoke | synthetic | forward/backward | MUST | DONE | mamba 2.2.6.post3 |
| R008 | M0 | 环境锁 | conda/pip/source pins | WSL2 | rebuild evidence | MUST | DONE | environment.yml + lock |
| R008A | M0 | PEFT 上游环境 | peft_boa torch2.1.1+cu118 | CPU/loader | imports/pip/shape/hash | MUST | DONE | 独立 YAML+全量锁；pip check、CPU math、真实 B32/K4 loader PASS；不代表 GPU/指标通过 |
| R009 | M0 | RGBNT100 完整性 | audit | all | triplets/JPEG | MUST | DONE | 17,250 triplets；70,715 JPEG |
| R010 | M1 | 强 checkpoint parity | MDReID | RGBNT201 test | mAP/R1/R5/R10 | MUST | DONE | 82.0868/85.1675/90.3110/92.5837 |
| R011 | M1 | DeMo tiny overfit | DeMo | tiny train | loss/R1 | MUST | WAITING | 等本地 GPU 低于 500 MiB 空闲门禁 |
| R012 | M1 | DeMo 50 epoch | fixed seed42 B32/K4/TB64 | RGBNT201 | mAP/R1/checkpoint | MUST | INCOMPLETE | epoch31 训练闭合后 ori 评估遭 CUDA unknown error；无 epoch31 评估/DeMo_50；epoch30 是最后固定里程碑 |
| R012R | M1 | DeMo 50 epoch replacement | resumable-v1 seed42 B32/K4/TB64 | RGBNT201 | exact parity/mAP/R1/full state | MUST | WAITING | 必须从 epoch0 重跑；epoch10 与原 R012 417 tensors 逐张量完全一致后才可继续；等 GPU 空闲门禁 |
| R013 | M1 | DeMo 协议偏差 | fixed vs test-selected | RGBNT201 | delta/provenance | MUST | WAITING | 等 R012R 产生固定 DeMo_50.pth；禁止用 epoch17 test-selected 代替 |
| R014 | M1 | 系统门 | B32 + eval latency | RGBNT201 | VRAM/latency/driver | MUST | WARN | 8-step 6,894 MiB；历史 ori >300s；R012 终止附近 7 条 nvlddmkm 153 |
| R015 | M1 | train-only dev | 141-fit/30-dev | train_171 only | overlap/positives | MUST | DONE | test overlap=0；825/825 valid |
| R016 | M1 | 证据包 | 30 versioned JSONs | all | SHA/protocol | MUST | DONE | 加入 R012 incident、baseline/protocol/license audits 与 claim-gates pre-result receipt；SHA256SUMS pass |
| R017 | M1 | MDReID 强化驱动复验 | hardened driver | RGBNT201 | parity/full audit | MUST | WAITING | 等本地 GPU 低于 500 MiB 空闲门禁 |
| R017A | M1 | 最高公开源码训练比较器（无仓库许可证） | PEFT-BoA d2b198b seed1111 B64/K4 | RGBNT201 | fixed-e120 mAP/R1/provenance | MUST | WAITING | exact upstream CPU env/source/loader PASS；公开 82.7/86.1=test-selected e80，fixed e120=82.2/85.8；仅隔离运行、不复制源码；等待接缝同意、加固驱动和 GPU 容量门 |
| R017B | M1 | 高指标 MIT baseline | Signal cd1b0a6 seed1234 B64/K8 fixed-e50 | RGBNT201 | mAP/R1/hash/provenance | MUST | WAITING | MIT、clean source、真实 CPU loader 与公开 fixed-path log已审计；权重字节未取得，需隔离环境、容量门和本机 fixed-e50；不得用 80.3/85.2 冒充本机结果 |
| R017C | M1 | 补充 checkpoint 比较器 | MFRNet ec54a13 official weight | RGBNT201 | mAP/R1/hash | SHOULD | WAITING | checkpoint/env/loader/strict load PASS；Tutel 路由反例证明不可默认 microbatch 等价，parity 必须 B128；等待接缝同意、`<500 MiB` GPU 门，公开 best 标 test-selected |
| R018 | M1 | epoch10 receipt | DeMo_10.pth | RGBNT201 | ordering/hash | MUST | DONE | SHA b2ab79f0…31c0 |
| R018A | M1 | epoch20 receipt | DeMo_20.pth | RGBNT201 | ordering/hash/metrics | MUST | DONE | SHA 5d61a4cf…e7b2；非 test 选点 |
| R018B | M1 | epoch30 receipt | DeMo_30.pth | RGBNT201 | ordering/hash/metrics | MUST | DONE | SHA d5e375fa…4ce5；非 test 选点 |
| R019 | M1 | summarizer fail-closed | live+synthetic | logs | coverage/latency | MUST | DONE | partial/gap/duplicate/overflow 全拒绝 |
| R019A | M1 | quantitative claim gate | claim_gates_v1 | pre-result protocol | hash/statistics/MPRD | MUST | DONE | R020+ 前冻结；identity-cluster bootstrap/sign-flip、Holm、三创新点效应门；SHA 2ca8badf…13e4 |
| R020 | M2 | CNN standalone | TriFusion CNN | 141-fit/dev | mAP/R1 | MUST | WAITING | 等接缝同意；完整专家 |
| R021 | M2 | Transformer standalone | CLIP Transformer | 141-fit/dev | mAP/R1 | MUST | WAITING | 等接缝同意 |
| R022 | M2 | Mamba standalone | 2D bidirectional Mamba | 141-fit/dev | mAP/R1 | MUST | WAITING | 等接缝同意 |
| R023 | M2 | late-fusion controls | mean + concat/MLP | 141-fit/dev | fused/branch/params | MUST | WAITING | 排除 ensemble 容量 |
| R024 | M2 | HFER tracer | HFER-uniform | 141-fit/dev | fused/branch/grad | MUST | WAITING | 至少两个 branch 改善门 |
| R025 | M2 | HFER controls | no-private/no-role/matched MLP | 141-fit/dev | mAP/R1/CKA/FLOPs | MUST | WAITING | capacity floor |
| R025A | M2 | relay depth control | one-stage HFER | 141-fit/dev | fused/branch mAP/R1 | MUST | WAITING | 排除一次交换即可解释增益 |
| R025B | M2 | relay direction control | one-way ring relay | 141-fit/dev | fused/branch mAP/R1 | MUST | WAITING | 对比完整同步双向中继 |
| R025C | M2 | role specificity | source-role mixer permutation | 141-fit/dev | fused/branch/CKA | MUST | WAITING | 角色置换不得复现完整增益 |
| R025D | M2 | heterogeneous necessity | parameter-matched homogeneous triple | 141-fit/dev | mAP/R1/params/FLOPs | MUST | WAITING | 至少一个同构三专家族；不匹配则明确标注 |
| R025E | M2 | ensemble capacity | widened single backbone | 141-fit/dev | mAP/R1/params/FLOPs | MUST | WAITING | 与 U2 params≤2%、activated FLOPs≤5% |
| R026 | M3 | CIRC CLI contract | tiny fold fixture | synthetic IDs | receipt/hash/mode/overlap | MUST | WAITING | dev vs post-freeze-final；xcam；edge hash/cost |
| R027 | M3 | target generator fold0 | HFER-uniform | fit folds1+2 | checkpoint | MUST | TODO | frozen generator |
| R028 | M3 | target generator fold1 | HFER-uniform | fit folds0+2 | checkpoint | MUST | TODO | frozen generator |
| R029 | M3 | target generator fold2 | HFER-uniform | fit folds0+1 | checkpoint | MUST | TODO | frozen generator |
| R030 | M3 | OOF intervention cache | full rerun removals | held-out folds | target/delta/hash | MUST | TODO | all T/D/R；每 stage/row 1 条 hash-edge audit；xcam |
| R031 | M3 | approximation audit | cheap LOO vs full rerun | dev frozen | signed/Spearman | MUST | TODO | cheap 仅通过后启用 |
| R031A | M3 | intervention audit | T/D/R + sampled E + signed strata | dev frozen | effects/interaction/edge coverage | MUST | TODO | r=helpful-vs-not；harmful audit-only |
| R031B | M3 | target validity audit | query-gallery + deployed U2 | dev frozen | symmetry/proxy transfer receipts | MUST | TODO | 不通过则缩窄 causal claim |
| R032 | M3 | generic router | HFER + softmax | 141-fit/dev | mAP/R1/entropy | MUST | TODO | R0 |
| R033 | M3 | observational evidence | joint Beta end-to-end | 141-fit/dev | retrieval/calibration | MUST | TODO | R1 |
| R034 | M3 | cheap target | joint Beta head-only LOO | 141-fit/dev | retrieval/calibration | MUST | TODO | R2 |
| R035 | M3 | CIRC | joint Beta OOF full intervention | 141-fit/dev | mAP/R1/BCE/Brier/ECE/coverage | MUST | TODO | 逐条件；expert/modality/camera/id-frequency；cluster N_eff |
| R036 | M3 | comparability negative | 9 independent heads | 141-fit/dev | shared-vs-independent grouped calibration | MUST | TODO | R4；过度离散与 coverage |
| R037 | M3 | global router control | non-evidential global router | 141-fit/dev | mAP/R1/calibration | MUST | TODO | R5 |
| R037A | M3 | gate granularity controls | expert-only/modality-only/scalar | 141-fit/dev | mAP/R1/params/calibration | MUST | TODO | 排除细粒度或 Beta 形式本身 |
| R037B | M3 | exclusion-granularity control | TIGER-style whole-expert exclusion | 141-fit/dev | mAP/R1/effect agreement | MUST | TODO | 对比 expert×modality T/D/R target |
| R038 | M3 | fusion-only | CIRC posterior at fusion | 141-fit/dev | fused/branch | MUST | TODO | U0 |
| R039 | M3 | relay-only | CIRC posterior at HFER | 141-fit/dev | fused/branch | MUST | TODO | U1 |
| R040 | M3 | promoted core | same CIRC posterior everywhere | 141-fit/dev | all claim metrics | MUST | TODO | U2 默认 r；需过 ΔmAP≥0.30、R1 非劣和两分支门；(1-u) 仅 coverage 过门后可晋级 |
| R041 | M3 | fragmentation control | separate matched routers | 141-fit/dev | mAP/R1/params | MUST | TODO | U3 |
| R042 | M3 | causal negatives | shuffle/permutation/context-free | 141-fit/dev | gain/calibration | MUST | TODO | 增益≤U2 的50%且无 Holm-corrected 正增益 |
| R042A | M3 | calibration-form control | frozen temperature rescaling | dev frozen | ECE/Brier/AUROC/retrieval | MUST | TODO | 排除温度重标定即可解释 CIRC |
| R042B | M3 | leakage probes | identity/camera-only predictors | dev frozen | calibration/gain | MUST | TODO | 泄漏 probe 不得通过负控制门 |
| R043 | M3 | full capacity control | enlarged no-relay MLP | 141-fit/dev | params/FLOPs/mAP | MUST | TODO | params≤2%、activated FLOPs≤5% |
| R044 | M4 | no peer | promoted core | 141-fit/dev | mAP/R1/CKA | MUST | TODO | K0 |
| R045 | M4 | ordinary peer | +symmetric KL | 141-fit/dev | mAP/R1/CKA | MUST | TODO | K1 |
| R046 | M4 | RDPT auxiliary | +direction/reject/role payload | 141-fit/dev | branch/reject/CKA | NICE | TODO | K3；不默认晋升 |
| R047 | M4 | RDPT hard controls | wrong payload/adaptive teacher | 141-fit/dev | branch/reject/CKA | NICE | TODO | K4 |
| R050 | M4 | 单种子晋级 | promoted core | full171→test once | all primary | MUST | TODO | 配置先冻结；原 dev 仅训练回流；无再次选择 |
| R060 | M5 | 正式 seed1 | promoted core | RGBNT201 | mAP/R1/R5/R10 | MUST | TODO | seed42 |
| R061 | M5 | 正式 seed2 | promoted core | RGBNT201 | mAP/R1/R5/R10 | MUST | TODO | seed3407 |
| R062 | M5 | 正式 seed3 | promoted core | RGBNT201 | mAP/R1/R5/R10 | MUST | TODO | seed9199 |
| R063 | M5 | 缺失模态 | seven non-empty masks | RGBNT201 | avg/worst mAP/R1 | MUST | TODO | missing ≠ degraded；平均/最差 mAP 门 +1.00/+0.50 pp |
| R064 | M5 | 质量退化 | registered corruptions | RGBNT201 | AUC/ECE/monotonicity/T-D-R-E | MUST | TODO | fixed strengths/seeds；≥80% cells 单调且 severity→r rho≤−0.50 |
| R065 | M5 | 外部验证 | promoted core | MSVR310/RGBNT100 | mAP/R1 | MUST | TODO | 至少单 seed |
| R070 | M5 | 效率 | baseline/controls/core | real+synthetic | params/FLOPs/fps/VRAM | MUST | TODO | total vs activated |
| R071 | M5 | 统计 | promoted vs controls | identity clusters | 10k bootstrap/100k sign-flip/Holm | MUST | TODO | point delta、95% CI、Holm p 同时过 claim_gates_v1 |
| R071A | M5 | 最终 SOTA 前沿刷新 | official papers/repos/author pages | current public | protocol/resources/threshold | MUST | TODO | 重点复查 Hyper-ReID、RoDI 代码和 2026-08-31 后同赛道发布；冻结更新门槛 |
| R072 | M5 | 结果/claim 审计 | all promoted evidence | all | hashes/protocol/claims | MUST | TODO | SOTA 前置门 |
