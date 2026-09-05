# V22 Experiment Tracker

更新时间：2026-09-05T18:01:37.883354+08:00；状态M0_PASS_Q1_RUNNING。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V22-SOURCE | 全1680batch相机监督元数据与整数复核 | DONE_METADATA_ONLY | evidence/trifusion_source_camera_metadata_20260905.json |
| V22-T0 | 三项CUDA数学/行域/梯度契约 | DONE_PASS | evidence/trifusion_v22_t0_20260905.json |
| V22-M0 | 六模型配对、两端容量、100步过拟合 | DONE_PASS | evidence/trifusion_v22_m0_seed42_5ae096b.json |
| V22-FILES | 30完整source/config/plan/weights字节SHA | DONE_PASS | evidence/trifusion_v22_m0_file_verification_20260905.json |
| V22-ARRAYS | 全48batch/116更新/100步实际整数及loss复核 | DONE_PASS | evidence/trifusion_v22_m0_array_verification_20260905.json |
| V22-M0-AUDIT | 独立M0审计 | PENDING | 原始文件交独立审计 |
| V22-Q1 | 三fold两端各20epoch完整检索 | RUNNING | 原PID34656；17:57已有3完整epoch/87步，零完整paired fold |
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
解析底和超额比数值差0。执行器复核不替代独立M0审计，独立审计待完成。

完整Q1始终为三fold两端各20epoch/3360优化步、120epoch记录，最终checkpoint唯一；
所有5输出/3126gallery/571query/21身份与五项科学门保持。当前没有Q1检索结果、
D1/dev/official或SOTA新指标。首个完整paired fold下一次观测约18:18–18:20；
完整Q1暂估19:10–19:30，收到完整epoch/端点用时后修正，不提前反复查询GPU。
