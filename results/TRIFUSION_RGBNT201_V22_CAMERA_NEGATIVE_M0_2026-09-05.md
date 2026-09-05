# V22 Camera-Negative M0 已完成，完整 Q1 运行中

记录时间：2026-09-05T18:01:37.883354+08:00

M0于原进程约225.699910秒时完成，全部五项工程检查通过。17:55:06实际观测
原PID34656存活、三fold配对完整、GPU6546MiB/93%；没有完整Q1 paired fold。
原进程自动继续固定完整Q1，未重启、未改变执行source5ae096b或方案/配置。

六个真实模型各98800141总参数、7841292训练参数、203训练tensor，94-source/
47-heldout严格隔离；48个forward-only batch的初始化、五路输出、增强/路径/索引
及camera支持全部配对。两端容量各8步，实验端从fresh source固定batch100步，
共116项目优化步和116对前向/反传，另有48forward-only；M0不计入Q1预算。
容量峰值6054/6200MiB；两端容量及100步训练均203/203非零梯度、overflow0、
完整冻结state不变。T0三CUDA测试已通过，T0项目/toy optimizer更新均0。

固定100步初/末loss0.7189050912857056/0.5803177952766418，
解析底0.57838292104621，超额loss比0.013769174124866987<=0.1。
选定总loss的共同ID及branch项0.6106956005→0.5803177953，
加权MCNL项0.1082095206→0；固定batch56/64行具备两类负例，8行缺少同相机负例。
三专家初始camera hinge active rows为47/53/40，末步均0。
未进入实验端优化目标的原残差triplet诊断0.0003517768→0.0321446024，
不能把“MCNL降至0”扩写成所有度量都改善；本次M0没有检索性能结论。
全部100步分量、两段hinge及active rows都保留，未挑选中间checkpoint。

原始M0 snapshot374648字节 SHAad0a27abbba79f1c039f68ebcfcc64eba731916581a8dec67a5c64c19d212427。
17:57:12 snapshot日志104518字节 SHA47dc1a77075d588551f2fe73933da03a739815d40c09d3960ea9b9ef30491369。
当时Q1控制端已记录3个完整epoch/87优化步，尚无新checkpoint；当前未完成epoch
的步数不包含在87中。因此summary的folds=[]代表零完整检索fold，不能说Q1优化步0。
M0 snapshot只是工程阶段的不可变证据，原始run_summary继续随完整Q1更新。

远端30个source/config/plan/tests/metadata/CLIP/V12/六source权重完整文件SHA全通过，
Signal commit/diff与执行绑定相同；该检查没有加载权重tensor或额外优化。
本地JSON/整数/解析loss复核全部48batch、116更新及100步分量通过：
总分量最大差5.960464477539063e-08，MCNL分量最大差7.450580596923828e-09，
解析底和超额比数值差0。执行器复核不替代独立M0审计，独立审计待完成。

完整Q1始终为三fold两端各20epoch/3360优化步、120epoch记录，最终checkpoint唯一；
所有5输出/3126gallery/571query/21身份与五项科学门保持。当前没有Q1检索结果、
D1/dev/official或SOTA新指标。首个完整paired fold下一次观测约18:18–18:20；
完整Q1暂估19:10–19:30，收到完整epoch/端点用时后修正，不提前反复查询GPU。

## 原始证据

- evidence/trifusion_v22_m0_seed42_5ae096b.json
- evidence/trifusion_v22_m0_run_snapshot_20260905.log
- evidence/trifusion_v22_m0_file_verification_20260905.json
- evidence/trifusion_v22_m0_array_verification_20260905.json
- evidence/trifusion_v22_progress_20260905_175506.json
- evidence/trifusion_v22_t0_20260905.json
- evidence/trifusion_v22_launch_20260905.json
- evidence/trifusion_source_camera_metadata_20260905.json
- evidence/trifusion_source_camera_metadata_integer_verification_20260905.json

## 全100步分量范围（固定batch；末步不是中间最佳）

| 分量 | 初值 | 第100步更新前 | 最小 | 最大 | 末20步均值 |
|---|---:|---:|---:|---:|---:|
| total | 0.718905091286 | 0.580317795277 | 0.580317795277 | 0.718905091286 | 0.580341613293 |
| common_identity_and_branch_triplet | 0.61069560051 | 0.580317795277 | 0.580317795277 | 0.643919408321 | 0.580341613293 |
| ordinary_residual_triplet | 0.000351776834577 | 0.0321446023881 | 0.000351776834577 | 0.0554895401001 | 0.0329893773422 |
| camera_residual_metric | 0.108209520578 | 0 | 0 | 0.108209520578 | 0 |
| camera_valid_rows | 56 | 56 | 56 | 56 | 56 |
| camera_same_negative_missing_rows | 8 | 8 | 8 | 8 | 8 |
| camera_other_negative_missing_rows | 0 | 0 | 0 | 0 | 0 |
| camera_cross_camera_positive_rows | 8 | 8 | 8 | 8 | 8 |
| mcnl_cnn_positive_term | 0 | 0 | 0 | 3.46696833731e-05 | 0 |
| mcnl_cnn_camera_term | 0.161605209112 | 0 | 0 | 0.161605209112 | 0 |
| mcnl_cnn_positive_active_rows | 0 | 0 | 0 | 1 | 0 |
| mcnl_cnn_camera_active_rows | 47 | 0 | 0 | 47 | 0 |
| mcnl_transformer_positive_term | 0 | 0 | 0 | 0 | 0 |
| mcnl_transformer_camera_term | 0.123479753733 | 0 | 0 | 0.123479753733 | 0 |
| mcnl_transformer_positive_active_rows | 0 | 0 | 0 | 0 | 0 |
| mcnl_transformer_camera_active_rows | 53 | 0 | 0 | 53 | 0 |
| mcnl_mamba_positive_term | 0 | 0 | 0 | 0.00011707629892 | 0 |
| mcnl_mamba_camera_term | 0.147753119469 | 0 | 0 | 0.147753119469 | 0 |
| mcnl_mamba_positive_active_rows | 0 | 0 | 0 | 1 | 0 |
| mcnl_mamba_camera_active_rows | 40 | 0 | 0 | 40 | 0 |


## 独立M0审计完成（2026-09-05T18:27:50.881702+08:00）

独立M0审计原始结论为overall/integrity WARN/warn、engineering_integrity=pass、
fixed_m0_qualification=pass、scientific_qualification=fail。审计者用Python3.12.14/
NumPy2.3.5独立复算元数据和全部100步分量，sidecar最大差0，固定超额loss比
0.013769174124866987；总loss及MCNL分量重构差分别5.960464477539063e-08、
7.450580596923828e-09。未选中的原残差triplet上升这一负诊断原样保留。

这里scientific_qualification=fail表示本次审计所持M0文件尚无Q1终态检索科学证据，
不表示固定M0门失败，也不冒充Q1终态判定。审计范围为17:57的M0/早期训练日志：
3个完整Q1 epoch、87步、当时零checkpoint；不覆盖随后18:20捕获的第一折检索。
实际分类为source_only_engineering_m0_real_train_source_batches_and_metadata_replay_plus_synthetic_cuda_t0_with_nonterminal_q1_training_log_no_heldout_retrieval。

保留来源限制：审计时HEADf63889f不同于执行5ae096b、远端M0验证观察2fd6506；
17个依赖中13个本地原字节匹配，criterion.py、experts/mamba.py、
experts/semantic_residual.py及protocols/rgbnt201_dev_v1.json四个只在LF标准化后匹配。
远端30项全文件SHA账本存在，但审计者未独立持有/加载CLIP、V12、权重tensor或图像。
审计没有远端命令、网络、下载、模型、训练或额外检索；WARN不能由文字整理抹去。

原始审计MD30347字节 SHA6c8420dfb7275df657c53b387eb02a8913077fc5d3bd11d6f30881d39d280e9c；
JSON50920字节 SHA371f54725d8c601559a323f3beb31c2e00ced50b68b978d49cd2af81834de91d。
完整请求、逐字回复、元数据、报告和审计前M0说明/跟踪表已在本地trace run09归档。
本段为执行器在审计完成后的归档说明，不声称审计者复核了本段及后续Q1结果。
