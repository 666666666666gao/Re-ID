# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R000 | M0 | 数据完整性 | RGBNT201 audit | all | ids/triplets/pairing/JPEG | MUST | DONE | 171/141/30 identities；0 pairing/JPEG errors |
| R001 | M0 | CUDA 环境 | torch CUDA smoke | synthetic | CUDA/version/memory | MUST | TODO | RTX 4060 8 GB |
| R002 | M0 | 数据接口 | official loader one batch | train_171 | shapes/labels/masks | MUST | TODO | root=`/root/mmreid-trifusion/data` |
| R003 | M0 | 模型接口 | DeMo forward/backward | synthetic | finite loss/grad/memory | MUST | TODO | modern PyTorch compatibility |
| R004 | M0 | 指标正确性 | evaluator worked example | synthetic | exact mAP/CMC | MUST | TODO | 手算固定样本 |
| R005 | M0 | 数据完整性 | MSVR310 audit | all | ids/triplets/pairing/JPEG | MUST | DONE | 310 identities；train/test disjoint；0 pairing/JPEG errors |
| R006 | M0 | 研究口径 | novelty/protocol/SOTA audit | primary sources | protocol/resources/claims | MUST | DONE | `docs/RESEARCH_AUDIT_2026-08-31.md` |
| R010 | M1 | 公开 checkpoint 复现 | MDReID | RGBNT201 | mAP/R1/R5/R10 | MUST | TODO | 官方 checkpoint；目标 82.1/85.2 ±0.3 pp |
| R011 | M1 | 实现底座 quick overfit | DeMo | tiny train subset | loss/Rank-1 | MUST | TODO | 先证明现代环境可学习 |
| R012 | M1 | 实现底座完整复现 | DeMo | RGBNT201 | mAP/R1/R5/R10 | MUST | TODO | seed 42；报告值 79.0/82.3 |
| R013 | M1 | 基线偏差诊断 | MDReID/DeMo protocol audit | RGBNT201 | delta vs paper | MUST | BLOCKED | 等 R010/R012 |
| R020 | M2 | 单分支 | CNN-only | RGBNT201 | mAP/R1 | MUST | TODO | matched width |
| R021 | M2 | 单分支 | Transformer-only | RGBNT201 | mAP/R1 | MUST | TODO | DeMo-compatible tokens |
| R022 | M2 | 单分支 | Mamba-only | RGBNT201 | mAP/R1 | MUST | TODO | bidirectional multi-axis scan |
| R023 | M2 | 普通融合控制 | mean / concat+MLP | RGBNT201 | mAP/R1/params | MUST | TODO | 排除集成收益 |
| R024 | M2 | 互补容量 | branch oracle | RGBNT201 train-safe eval | oracle/error overlap | MUST | TODO | 不用于测试调参 |
| R030 | M3 | 创新 1 | +HFER | RGBNT201 | fused + per-branch metrics | MUST | TODO | full-expert staged relay |
| R031 | M3 | 普通路由控制 | +soft gate | RGBNT201 | mAP/R1/router entropy | MUST | TODO | 与 URGC 对照 |
| R032 | M3 | 创新 2 | +URGC | RGBNT201 | mAP/R1/corr/ECE/Brier | MUST | TODO | unified calibrated posterior |
| R033 | M3 | 普通互教控制 | +symmetric KL | RGBNT201 | mAP/R1/CKA | MUST | TODO | 与 RDPT 对照 |
| R034 | M3 | 创新 3 | +RDPT | RGBNT201 | mAP/R1/CKA/reject-rate | MUST | TODO | directional, rejectable teaching |
| R035 | M3 | 容量匹配 | larger concat control | RGBNT201 | mAP/R1/FLOPs | MUST | TODO | 参数不少于完整模型 |
| R036 | M3 | 主方法 | full TriFusion | RGBNT201 | all primary metrics | MUST | TODO | 单种子晋级门 |
| R040 | M4 | 正式种子 1 | full TriFusion | RGBNT201 | mAP/R1/R5/R10 | MUST | TODO | seed 42 |
| R041 | M4 | 正式种子 2 | full TriFusion | RGBNT201 | mAP/R1/R5/R10 | MUST | TODO | seed 3407 |
| R042 | M4 | 正式种子 3 | full TriFusion | RGBNT201 | mAP/R1/R5/R10 | MUST | TODO | seed 9199 |
| R043 | M4 | 缺失模态 | full TriFusion | RGBNT201 six masks | avg/worst mAP/R1 | MUST | TODO | 固定组合 |
| R044 | M4 | 质量退化 | full TriFusion | RGBNT201 corruptions | robustness AUC/ECE | MUST | TODO | 强度预注册 |
| R045 | M4 | 跨数据集 | full TriFusion | MSVR310 | mAP/R1 | MUST | TODO | 官方数据已下载并审计 |
| R046 | M4 | 跨数据集 | full TriFusion | RGBNT100 | mAP/R1 | MUST | BLOCKED | 百度网盘数据 |
| R050 | M5 | 效率 | baseline/controls/full | synthetic+RGBNT201 | params/FLOPs/fps/VRAM | MUST | TODO | 区分总与激活 FLOPs |
| R051 | M5 | 行为解释 | router/relay visualization | RGBNT201 | weights/margins/maps | MUST | TODO | 不使用测试标签调参 |
| R052 | M5 | 失败分析 | full TriFusion | RGBNT201 | failure taxonomy | MUST | TODO | 与 DeMo 配对比较 |
| R053 | M5 | 结果审计 | all promoted runs | all | hashes/protocol/claims | MUST | TODO | SOTA 声明前置门 |
