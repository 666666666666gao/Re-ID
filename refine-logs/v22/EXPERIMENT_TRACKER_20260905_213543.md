# V22 Experiment Tracker

更新时间：2026-09-05T21:35:43.112525+08:00；状态Q1_FAIL_SEALED_INDEPENDENT_AUDIT_COMPLETE。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V22-SOURCE | 全1680batch相机监督元数据及整数重放 | DONE_METADATA_ONLY | evidence/trifusion_source_camera_metadata_20260905.json |
| V22-T0 | 三项CUDA数学/行域/梯度契约 | DONE_PASS | evidence/trifusion_v22_t0_20260905.json |
| V22-M0 | 全六模型/两端容量/固定100步 | DONE_PASS | evidence/trifusion_v22_m0_seed42_5ae096b.json |
| V22-M0-AUDIT | 独立M0审计 | DONE_FIXED_M0_PASS_INTEGRITY_WARN | EXPERIMENT_AUDIT_V22_M0.md/json |
| V22-Q1 | 三fold两端各20epoch完整五路检索 | DONE_ALL_FIVE_SCIENTIFIC_GATES_FAIL | evidence/trifusion_v22_q1_seed42_5ae096b.json |
| V22-Q1-FILES | 36项完整文件SHA/六receipt一致 | DONE_PASS | evidence/trifusion_v22_terminal_file_verification_20260905.json |
| V22-Q1-ARRAYS | 全mask/五路/21身份/10000次Bootstrap | DONE_PASS | evidence/trifusion_v22_terminal_array_verification_20260905.json |
| V22-Q1-LOG | 全120epoch逐行一致/3360步/全部loss诊断 | DONE_PASS | evidence/trifusion_v22_terminal_log_and_loss_verification_20260905.json |
| V22-Q1-AUDIT | 独立终态审计 | DONE_ENGINEERING_PASS_INTEGRITY_WARN_SCIENTIFIC_FAIL | EXPERIMENT_AUDIT_V22_Q1.md/json |
| V22-D1 | 141-fit refit和30-dev | NOT_QUALIFIED_NOT_RUN | 固定五科学门均FAIL |

V22原始单进程完整结束，退出码0；20:43:56实际观测PID34656已退出、GPU1MiB/0%。
原始运行耗时4165.238463401794秒（69.420641分钟，含M0）；三fold两端各20epoch、
共120条完整epoch记录/3360优化步全部完成，未因第一折负收益提前停止。
执行commit5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36及固定配置/方案不变。

全部141 heldout身份、3126gallery、571合法query/21跨camera身份完整保留，
2555条只从query分母排除；六个最终checkpoint均从原source重建后strict reload，
模型state SHA与该端训练final一致，冻结Signal与共享尾部不变。
baseline/fused/CNN/Transformer/Mamba五输出与三fold/all21身份完整表均已生成。

MCNL/control aggregate fused为78.984454/80.640677 mAP，差-1.656222pp；
R1为82.311734/83.712785。三fold fused差-0.676068/-2.149943/-2.140645。
CNN/Transformer/Mamba aggregate mAP差-0.897312/-4.265440/+0.317625；
candidate CNN79.152126高于candidate fused78.984454，因此五项固定科学门全部FAIL。
全部21身份、10000次seed42 cluster bootstrap的95%下界为-3.8769957222550886。

原始汇总2207490字节 SHAb8cd7db81efc3827a91d165d47e001073785420baf8ddfc8507a2eead9c3d6a3；
完整日志255971字节 SHAa091390a25ced0cfd336fce3c5bd6c51565fcad6becfbf3419778bcb9b7a2f1a。
远端36项source/config/plan/tests/metadata/CLIP/V12/source及新final权重全文件SHA通过，
六独立receipt对象与summary内嵌对象相同。核验没有加载额外权重tensor/图像或重跑检索。
本地NumPy2.5.2完整mask、AP/Rank聚合、全21身份Bootstrap数值最大差0；
120行日志逐对象等于六端history，步数和不可变M0都精确一致。

每臂1680batch/107520样本暴露：98796行具两类负例、7852缺同相机负例、
872缺其他相机负例、17600具跨camera正例；六端全部训练批次支持数与冻结metadata相同。
三候选末epoch MCNL项均下降，但未选用的普通残差Triplet项均较首epoch上升；
这是训练目标改变后的实测诊断，不能把MCNL下降写成检索泛化改善或唯一因果证明。

固定V22封存Q1_FAIL，不作margin/系数/sampler/epoch/LR/seed变体或重训，
D1/dev/official均未执行。M0工程/固定门独立审计PASS、完整性WARN已归档；
它不覆盖完整Q1；独立终态审计现已完成，结论见下。当前best dev仍V8的58.4050/59.3939，
65mAP开发门和官方目标均未达到，整体goal继续active。

完整比较：results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_2026-09-05.md，
及evidence/trifusion_v22_complete_comparison_20260905.json。
M0审计原始scope/四项LF-only依赖及remote-ledger持有限制保留；
终态执行器核验不替代独立审计。
历史每版tracker和第一折部分结果原件不改写。

## 独立终态审计

独立终态审计已完成：GPT-5.5 xhigh原始结论为工程PASS、固定M0 PASS、
完整性WARN、科学资格FAIL，实际evaluation_type为
real_gt_train_internal_complete_path_oof_reused_development_qualification。
全部五项固定科学门均FAIL；没有D1/dev/official/test结果，也没有SOTA晋级证据。

审计员以本地Python3.12.14/NumPy2.3.5从原始AP/Rank和真实ID/camera数组重算
全部fold/output聚合、21身份/10000次seed42 Bootstrap、全部120epoch日志及loss、
1680批相机元数据整数计数、M0固定首/100步门；指标/增益/Bootstrap最大差0，
loss分量舍入最大差7.105646293581458e-09，support舍入最大差7.105427357601002e-15。
全部六receipt与summary对象一致，query mask无不一致，3360步训练记录完整。

WARN原样保留：criterion、mamba、semantic_residual及protocol四份本地文本仅LF归一化后
等于运行字节；远端CLIP/V12/final权重等14项只有清单持有，审计员未独立取得权重字节，
没有加载tensor、图像或重算模型embedding/distance。运行commit5ae096b与审计时
current HEAD ad7841d不同，绑定依赖执行源码SHA，不把文档commit当作训练commit。
这仍是反复使用的train-internal完整路径OOF开发资格，不能写成独立dev或官方泛化。
报告EXPERIMENT_AUDIT_V22_Q1.md/json和trace run10保留原始字节与完整限制；
审计时result/tracker快照及全部46输入SHA已归档。固定Q1_FAIL及禁扫描/重训规则不变。
