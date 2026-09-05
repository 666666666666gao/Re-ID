# V22 Experiment Tracker

更新时间：2026-09-05T18:27:50.881702+08:00；状态M0_AUDITED_PASS_Q1_RUNNING_FIRST_PAIR_NEGATIVE。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V22-SOURCE | 全1680batch相机监督元数据与整数复核 | DONE_METADATA_ONLY | evidence/trifusion_source_camera_metadata_20260905.json |
| V22-T0 | 三项CUDA数学/行域/梯度契约 | DONE_PASS | evidence/trifusion_v22_t0_20260905.json |
| V22-M0 | 六模型配对、两端容量、100步过拟合 | DONE_PASS | evidence/trifusion_v22_m0_seed42_5ae096b.json |
| V22-FILES | 30完整source/config/plan/weights字节SHA | DONE_PASS | evidence/trifusion_v22_m0_file_verification_20260905.json |
| V22-ARRAYS | 全48batch/116更新/100步实际整数及loss复核 | DONE_PASS | evidence/trifusion_v22_m0_array_verification_20260905.json |
| V22-M0-AUDIT | 独立M0审计 | DONE_ENGINEERING_PASS_FIXED_M0_PASS_INTEGRITY_WARN | EXPERIMENT_AUDIT_V22_M0.md/json |
| V22-Q1 | 三fold两端各20epoch完整检索 | RUNNING_FIRST_PAIR_NEGATIVE | 原PID34656；18:20首fold完整，44完整epoch日志/1272步；继续全部端点 |
| V22-D1 | 141-fit refit与30-dev | NOT_QUALIFIED_NOT_RUN | 五项Q1门全通过后登记执行细节 |

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
解析底和超额比数值差0。独立M0审计现已完成，固定M0资格PASS、完整性WARN；详见下方归档。

18:20:12实际观测原PID34656仍运行，GPU6546MiB/100%；第一折两端最终20epoch
checkpoint均已strict reload并完成全部五输出检索，第二折control已记录第4epoch。
共44个完整Q1 epoch日志，已知完成1272步；其中两端完整receipt共40epoch/1160步。
这个计数不包含正在进行的epoch，不能把summary仅有1fold当作总训练只有1160步。

第一折1000gallery、47heldout身份、190合法query、810条仅从query分母排除。
fused对照71.201727/71.578947 mAP/R1，MCNL70.525659/69.473684，
fused差-0.676068pp；CNN/Transformer/Mamba的mAP差为-1.072661/-1.791450/+0.848838。
预注册“每折fused非负”条件已不满足，但全部后续端点照原计划继续，
不择端/择折，不重新训练或修改margin/权重/epoch/LR，终态仍未完成。

本地JSON/NumPy复算第一折两端五输出mAP/R1/R5/R10全部差0；
两端各580步的相机支持累计数与冻结metadata精确一致，loss加和最大误差
6.583487088818174e-09。这是执行器的第一折部分结果核验，不替代独立终态审计。
完整Q1仍为六端120epoch/3360步，下一完整paired-fold观测窗口约18:39–18:43，
按实际第一折速度修正整轮ETA为19:00–19:10（估计），原进程不重启。

## 独立M0审计归档

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
