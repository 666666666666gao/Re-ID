# V21 Experiment Tracker

更新时间：2026-09-05T17:24:33.877849+08:00；状态M0_FAIL_AUDITED_Q1_NOT_RUN。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V21-T0 | 三项CUDA数学/AMP/BN单测 | DONE_PASS | evidence/trifusion_v21_t0_20260905.json |
| V21-M0 | 六模型配对、两端容量、100步过拟合 | DONE_FAIL_FIXED_OVERFIT_GATE | evidence/trifusion_v21_m0_seed42_3c39351.json |
| V21-FILES | 远端30个完整源文件SHA | DONE_PASS | evidence/trifusion_v21_m0_terminal_file_verification_20260905.json |
| V21-ARRAYS | 全配对/预算/100步/解析底实际复核 | DONE_PASS | evidence/trifusion_v21_m0_array_verification_20260905.json |
| V21-M0-AUDIT | 独立M0终态审计 | DONE_ENGINEERING_PASS_INTEGRITY_WARN_FIXED_M0_FAIL | EXPERIMENT_AUDIT_V21_M0.md/json |
| V21-Q1 | 完整三fold ordinary40/SAM20 | NOT_QUALIFIED_NOT_RUN | M0固定门失败 |
| V21-D1 | 141-fit refit与30-dev | NOT_QUALIFIED_NOT_RUN | 无Q1 |

V21原进程执行342.520254秒后按固定M0门自动停止，状态M0_FAIL；
16:49:44实际观测原PID32331已退出、GPU1MiB/0%。Q1 fold/endpoint/epoch
均为0，未生成新checkpoint，没有D1/dev/official检索或模型改进指标。

三fold六模型共48个preflight batch完整配对，初始化、增强和五路输出SHA
均相同；各模型98,800,141总参数、7,841,292训练参数、203训练tensor。
普通/SAM各8容量优化步，前向反传分别8/16对；SAM固定batch100优化步、
200对前向反传。M0项目训练总116步、224对前向反传，另有48个forward-only
preflight batch；不能将这些计成Q1主训练预算。

容量峰值reserved分别6126/6284MiB。两端容量和SAM100步的第一遍与实际
更新梯度都覆盖203/203训练tensor，无缺失、overflow0、冻结state不变。
七BN每实际step计数只+1，SAM参数及第一遍统计按源码逐步精确恢复。
实际扰动范数范围0.04999999329447746–0.050000011920928955，符合rho0.05。

唯一未通过的固定M0条件是100步过拟合超额损失比：
原参数点loss第1/100步更新前为0.6110473871231079/0.5914160013198853，
解析底0.75H=0.57838292104621，比例0.39899872365870204>0.1。
用于更新梯度的扰动点loss为0.627093493938446/0.6106454133987427。
完整100步原始轨迹都保存；中途最小原参数loss0.5813781023025513，
末20步均值0.5843785017728805，都只是轨迹描述，不替代预先固定的第100步。
不能择最小值、均值、更换batch/rho/步数或放宽门槛使本次通过。

该结果只否定本次M0准备资格，尚无SAM的heldout检索结果，不足以断言SAM
检索泛化一定有害。此固定V21运行封存，不进行rho/LR/epoch/种子扫描或重训；
后续优化需要新的证据和新主假设，不能把改名重跑当新实验。

原始M0 summary297310字节，SHA
2ecc322270e4e1b82a77cf76e22ab76e359179fda9abc5ef7f2036db064d3c5d；
完整日志47488字节，SHA
0be3a21d007f1ff125779c13231c672ffaab67e884f02478b4be68e620f85194。
远端30个源/配置/方案/CLIP/V12/六source权重全字节SHA通过，目录新权重0；
本地JSON/math实际复核全6配对、116/224预算、100步全部分量、范数和状态
回执以及熵下界/比值，后两者数值差0。没有额外模型或权重张量加载。
工程数值可运行与M0固定准备门通过是两回事。独立M0终态审计已完成，
工程PASS、完整性WARN、固定M0资格FAIL，保留原始结论及全部限制。

V21固定运行封存；无V22方案、代码或训练。用户整体目标未完成。

## 已完成独立审计

独立V21 M0审计已完成，原始EXPERIMENT_AUDIT_V21_M0.md/json逐字节保留。
overall/integrity为WARN/warn；engineering_integrity=pass，
fixed_m0_qualification=fail，scientific_qualification=fail。
审计实际使用Python3.12.14/NumPy2.3.5重算全部M0记录，数值最大差0；
五项M0布尔门、116优化步/224前反传对和6项T0 toy更新均与原始证据一致。

实际分类为
source_only_engineering_m0_real_train_source_batches_plus_synthetic_t0_no_heldout_dev_official_or_test_retrieval。
原summary中的real_gt_train_internal_complete_path_oof是宽于本次执行范围的
runner/Q1标签，不能据此声称执行了OOF检索。原始summary不改写；
解释以M0_FAIL、folds=[]和零Q1/D1/dev/official访问共同约束。
scientific_qualification=fail在此表示没有科学推进资格及检索证据，
不等于观测到SAM heldout检索下降。

保留四项来源限制：执行commit3c393510与审计时HEAD4f31651不同，按文件SHA核对；
19个依赖中15个本地原字节一致，criterion.py、experts/mamba.py、
experts/semantic_residual.py、protocols/rgbnt201_dev_v1.json四个仅LF标准化一致；
远端数据/CLIP/V12权重只由完整文件SHA账本绑定，审计者没有直接持有这些字节；
参数和BN逐步精确恢复由运行assert及计数支持，原始文件没有逐步完整tensor dump。
这些限制不改变固定M0负结果，亦不能通过格式整理抹去WARN。

审计MD22047字节 SHA00244a74f9f1732aed811d3af9953dd5ea7245981b8c4ddc028b2f0593221244；
JSON24466字节 SHA3ec24312ed4f2c4ea466247ed93be3fc0d44bcbead3f22669965cfadc346ce95。
完整请求、逐字回复、元数据及报告原件已在本地trace run08归档；
trace不公开提交，审计报告本身随仓库发布。本段是审计完成后的归档说明，
不冒充审计者已复核本段新增文字。V21封存，当前没有新的训练或检索结果。
