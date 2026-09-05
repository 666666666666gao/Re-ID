# V22 Experiment Tracker

更新时间：2026-09-05T20:51:57.952144+08:00；状态Q1_FAIL_SEALED_TERMINAL_AUDIT_PENDING。

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
| V22-Q1-AUDIT | 独立终态审计 | PENDING | 将交原始文件和固定计划 |
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
它不覆盖完整Q1，本终态独立审计待完成。当前best dev仍V8的58.4050/59.3939，
65mAP开发门和官方目标均未达到，整体goal继续active。

完整比较：results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_2026-09-05.md，
及evidence/trifusion_v22_complete_comparison_20260905.json。
M0审计原始scope/四项LF-only依赖及remote-ledger持有限制保留；
终态执行器核验不替代独立审计。
历史每版tracker和第一折部分结果原件不改写。
