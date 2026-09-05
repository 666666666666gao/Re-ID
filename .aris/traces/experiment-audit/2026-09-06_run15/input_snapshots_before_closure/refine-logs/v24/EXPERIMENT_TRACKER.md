# V24 experiment tracker

更新时间：2026-09-06T03:21:48.280439+08:00。状态 Q1_FAIL_SEALED；原六端已终止、exit0。
执行源码6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33，原PID52030，全程未重启或修改配置。
完整Q1独立审计 PENDING；已完成的独立M0审计不覆盖本Q1结论。

| 项目 | 状态 | 证据 |
|---|---|---|
| 固定合同 / CUDA T0 / 完整 M0 | DONE_PASS_ENGINEERING | EXPERIMENT_PLAN.md; results/TRIFUSION_RGBNT201_V24_COMPLETE_M0_2026-09-06.md |
| 独立 M0 审计 | DONE_WARN_ENGINEERING_PASS | EXPERIMENT_AUDIT_V24_M0.md/json |
| 三折六端 Q1 | DONE_Q1_FAIL_SEALED | evidence/trifusion_v24_q1_seed42_6a4ac2c.json |
| 终态文件 / 全数组 / 日志核验 | DONE_PASS | evidence/trifusion_v24_q1_terminal_file_verification_20260906.json; evidence/trifusion_v24_q1_array_verification_20260906.json |
| 完整 Q1 独立审计 | PENDING | 不以执行者数值核验代替独立审计 |
| D1 / dev / official / 消融 | NOT_QUALIFIED_NOT_RUN | 原五项科学门未全部通过 |

三折×两端×20epoch=120epoch，3360优化更新/6720视图前后传；共3126完整gallery、571query、21query身份。
本结果属于反复用于开发的 train-internal complete-path OOF，非固定30-ID dev，非官方测试。
全程elapsed 8523.534031秒（包括M0、初始化、六端训练与评价）；不称为纯训练耗时。

| 输出 | 对照 mAP / Rank1 | 候选 mAP / Rank1 | mAP增益pp |
|---|---:|---:|---:|
| baseline_only | 77.487603116 / 79.334500876 | 77.487603116 / 79.334500876 | +0.000000000 |
| fused | 79.534977571 / 82.136602452 | 80.026284762 / 82.311733800 | +0.491307191 |
| cnn | 78.288484437 / 81.786339755 | 77.543382052 / 80.560420315 | -0.745102386 |
| transformer | 76.857705097 / 80.385288967 | 78.208970723 / 81.085814361 | +1.351265626 |
| mamba | 78.454236596 / 81.085814361 | 77.875033978 / 79.859894921 | -0.579202618 |

各折fused增益：-0.125144076, +0.166400744, +1.359049739 pp。
身份聚类bootstrap95%下界-0.694806868 pp，21身份、10000次、seed42。
原五项门：{"aggregate_fused_gain_at_least_1pp": false, "all_fold_fused_nonnegative": false, "all_expert_aggregate_nonnegative": false, "fused_bootstrap_lower_positive": false, "fused_beats_baseline_and_experts": true}。

两端原203可训练张量/7,841,292参数、98,800,141总参数相同，新增推理参数0。
相同初始化、双视图、采样顺序和原七组ID/Triplet，原型计算/更新均执行，只有原型损失系数不同。
所有正负身份真实且source-only；每折每端独立108个身份/相机原型。训练新增原型不接入推理。
该比较检验匹配双视图配置下原型系数的效果，不代表整个双视图方案相对旧单视图运行的净贡献。

终态summary SHA 7e43c38529598de5ca04cbe67502c60f60d26f8c6d8e20232d11254a0e2aa86a；log SHA e5999d17900312d81a4a1e78e371c3eed5cd4719983fcd74e0bab04c26b31730。
远端核对63份文件和24份原型二进制。
本地重算全部AP/Rank、原标签query过滤、120epoch损失和采样/原型年龄轨迹；不重建特征或距离。
数组最大差1.11022302463e-16 pp。保存全部五输出、全身份变化与负结果。

完整结果：results/TRIFUSION_RGBNT201_V24_COMPLETE_Q1_2026-09-06.md。
V24封存：不扫描原型温度/动量/权重、增强、采样、epoch/LR/seed，不重训或改变原门。
已登记单次只读source诊断，见SOURCE_PROTOTYPE_DIAGNOSTIC_PLAN_20260906.md/json；其运行状态单独记账。
独立终态审计和只读诊断均不构成晋级。当前可部署dev最好仍为V8 Phase-B58.4050/59.3939。
