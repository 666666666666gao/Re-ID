from pathlib import Path
import json,hashlib,datetime
OUT=Path(__file__).parent
def load(n):return json.loads((OUT/n).read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with (OUT/n).open('x',encoding='utf8',newline='\n') as f:
  if isinstance(v,str):f.write(v)
  else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
text=load('independent_text_results.json');arrays=load('independent_remote_arrays_results.json');extra=load('additional_text_results.json');stats=load('independent_remote_statistics_results.json');probe=load('remote_probe.result.json')
O=['baseline_only','fused','cnn','transformer','mamba'];labels=['Signal','fused','CNN','Transformer','Mamba']
table='| 输出 | Control mAP | Smooth-AP mAP | 配对增益 pp | Control R1 | Smooth-AP R1 |\n|---|---:|---:|---:|---:|---:|\n'
for name,label in zip(O,labels):
 a=text['metrics']['control'][name];b=text['metrics']['smooth_ap'][name];table+=f"| {label} | {a['mAP']:.9f} | {b['mAP']:.9f} | {text['paired']['gains_mAP'][name]:+.9f} | {a['Rank-1']:.6f} | {b['Rank-1']:.6f} |\n"
foldtable='| Fold | Source 身份/记录 | Heldout 身份/gallery记录 | Query 身份/记录 | Gallery-only 身份/记录 |\n|---|---:|---:|---:|---:|\n'
for r in text['folds']:foldtable+=f"| {r['fold']} | {r['source_identities']}/{r['source_records']} | {r['heldout_identities']}/{r['gallery_records']} | {r['query_identities']}/{r['queries']} | {r['gallery_only_identities']}/{r['gallery_only_records']} |\n"
cktable='| 端点 | 最终 checkpoint SHA-256 |\n|---|---|\n'
for p,v in arrays['files'].items():
 if p.endswith('roles_epoch20.pth'):cktable+=f"| {Path(p).parent.name} | `{v['sha256']}` |\n"
checkdefs={
 'A':dict(name='Ground truth provenance / full-path isolation',status='PASS',details='Dataset filename labels and train-only stratified identity folds are independently consistent for all 1032 records/3096 modal paths; all 3096 paths exist remotely; all source checkpoint source/heldout bindings and 1950 source-training index rows stay in source folds. Full gallery retains all 95 single-scene distractor identities/432 records. Image contents/archive authenticity were not re-read.',evidence=['tools/build_msvr310_train_oof_protocol.py:12','tools/build_msvr310_train_oof_protocol.py:35','tools/build_msvr310_train_oof_protocol.py:69','tools/train_msvr310_trifusion_oof.py:27','tools/train_msvr310_signal_oof.py:70','independent_text_results.json:1','independent_remote_arrays_results.json:1']),
 'B':dict(name='Raw metric definition / score normalization',status='PASS',details='L2 feature normalization precedes squared Euclidean ranking; AP divides by real positive count and averages all valid queries. No performance denominator uses model maxima/minima/means. Same-ID AND same-scene filtering is used; same-scene different-ID negatives remain. All 6000 AP/CMC outputs, 2069520 rank positions and fixed identity-cluster bootstrap/gates are independently recomputed.',evidence=['tools/train_msvr310_trifusion_oof.py:196','tools/train_msvr310_trifusion_oof.py:244','tools/train_msvr310_signal_oof.py:223','tools/msvr_smooth_ap.py:30','independent_text_checks.py:54']),
 'C':dict(name='Artifacts / terminal claims / hashes',status='PASS',details='57 raw intake files/83471329 bytes match inventory. Six final checkpoints and complete retrieval arrays remain remote and were independently hashed/read; 472 state tensors per checkpoint reconstructed, frozen/source state hashes match. Core code/config identical at launch, Q1-start and terminal commits. Three mutable current-status documents changed during audit and were re-read/snapshotted; no training/raw evidence mutation found. Requested EXPERIMENT_PLAN.md is absent; actual pinned contract is TRAINING_PLAN.md.',evidence=['pipeline.json:66','tools/run_msvr_smooth_ap.py:31','tools/train_msvr_smooth_ap.py:318','tools/train_msvr_smooth_ap.py:337','additional_text_results.json:1','input_drift_during_audit.json:1','remote_probe.result.json:1']),
 'D':dict(name='Executed objective / derivative and reconstruction scope',status='WARN',details='Active code changes only fused ranking term from step66; history begins step67. All training queues/masks, 99840 per-anchor APs, 116501504 stored distance entries and 29125376 fused distance derivative positions per objective/dtype checked. Analytic vs deployment gradient errors float64<=3.535e-17 and float32<=6.887e-8; hard derivatives exact. Q1 has zero independent direct full-model-gradient reconstruction checks; cached fields/RNG/feature leaves/parameter gradients/optimizer states were not saved. Runtime VJP invariants and role norm arithmetic are witnesses, not independent full-gradient replay.',evidence=['tools/train_msvr_smooth_ap.py:143','tools/train_msvr_smooth_ap.py:147','tools/train_msvr_smooth_ap.py:187','tools/train_msvr_smooth_ap.py:195','tools/train_msvr_smooth_ap.py:239','tools/msvr_smooth_ap.py:49','remote_array_checks.py:58','independent_remote_arrays_results.json:1']),
 'E':dict(name='Actual scope / resource and numerical limits',status='WARN',details='Complete fixed single-seed 3-fold/2-endpoint/5-output internal Q1, not official or multi-dataset evidence. Repeated internal protocol and no independent run-to-run variance bound. Pixel/index/init bindings match, but both endpoint trajectories differ from step2 during shared-objective warmup. Float64 algebra changes some near-tie retrieval ranks, while the registered FP32 replay is exact. Tiny positive derivative sign counts depend on dtype. These are explicitly bounded; scientific gate remains FAIL.',evidence=['refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md:19','tools/train_msvr_smooth_ap.py:246','additional_text_results.json:1','independent_derivative_census_comparison.json:1','verification_compact_summary.json:1']),
 'F':dict(name='Evaluation type',status='PASS',details='Q1 retrieval is real_gt, train_internal_identity_oof_not_official_test. Training losses use real source identity labels but approximate AP on batch/history is an optimization diagnostic. Synthetic T0 math is synthetic_proxy. Saved-distance derivatives/numerical identity checks are implementation-consistency diagnostics, not external-GT task performance or self-supervised learning.',evidence=['protocols/msvr310_train_oof_v1.json:4','tools/msvr_smooth_ap.py:33','tools/check_msvr_smooth_ap_math.py:1','tools/train_msvr_smooth_ap.py:310'])}
now=datetime.datetime.now().astimezone().isoformat()
report={
 'audit_skill':'experiment-audit','verdict':'WARN','reason_code':'complete_saved_evidence_passes_with_gradient_reconstruction_and_scientific_scope_limits',
 'summary':'Complete independent saved-evidence verification passes; fixed scientific Q1_FAIL is confirmed. No evidence of fabricated GT, score self-normalization or phantom terminal results within audited scope.',
 'date':'2026-09-09','generated_at':now,'auditor':'gpt-6-astra-max','agent_id':'/root/audit_msvr_smooth_ap_q1','verdict_id':'smooth_ap_q1_independent_audit_20260909',
 'executor_model':'gpt-6-astra','executor_family':'openai','reviewer_model':'gpt-6-astra','reviewer_family':'openai','reviewer_reasoning':'max','review_independence':'same-family','acceptance_status':'provisional',
 'overall_verdict':'WARN','integrity_status':'warn','deterministic_checks_status':'pass','scientific_qualification':'Q1_FAIL','fixed_q1_qualification':{'status':'FAIL','paired_gates_passed':1,'paired_gates_total':5,'candidate_vs_signal_gates_passed':0,'candidate_vs_signal_gates_total':5,'next_phase_qualified':False},
 'checks':checkdefs,'evaluation_type':'real_gt','evaluation_scope':'MSVR310 official-training records partitioned into 3 identity folds; seed42; 2 endpoints; fixed20epochs/260steps each; 600 eligible queries and1032 full gallery records; no official test or new model/image/optimizer execution in audit',
 'deterministic_scope':{'intake_identity_files':57,'intake_identity_bytes':83471329,'q1_training_updates':1560,'epoch_rows':120,'source_model_training_index_rows':1950,'q1_anchor_ap_values':99840,'saved_distance_elements':116501504,'fused_distance_derivative_positions_per_objective_per_dtype':29125376,'dtypes':['float64','float32'],'objectives':['hard','smooth_ap'],'query_output_endpoint_metrics':6000,'ranking_positions':2069520,'final_checkpoints':6,'checkpoint_state_tensors_each':472,'bootstrap_resamples_each':10000,'real_model_forwards':0,'optimizer_updates':0,'image_reads':0},
 'metrics':text['metrics'],'paired':text['paired'],'endpoint_gates':text['endpoint_gates'],
 'limitations':['No image-content/archive revalidation or syscall-level historical access audit.','No saved historical frozen fields, original RNG vectors, per-step feature leaves, full parameter gradients or optimizer states; full model-gradient replay unavailable.','No independent seeded role-initialization reconstruction; source/final state and all initialization bindings checked.','Q1 retrieval arrays originate from post-reload execution; saved checkpoint/features are not a fresh image-to-feature model replay.','203/203 is accumulated nonzero-gradient tensor coverage, not all203 at every step.','One training seed and reused internal folds; shared warmup trajectories already differ from step2.','Tiny Float32 derivative sign counts differ from Float64, despite small absolute derivative error; full rank permutation depends on FP32 near-tie convention.','A fresh same-family review is provisional, not cross-family acceptance.'],
 'claims':[{'id':'fixed_terminal_q1','impact':'supported','claim':'This fixed Q1 comparison completed and failed both required qualification groups.'},{'id':'positive_fused_mean','impact':'supported_with_scope','claim':'Paired mean fused +0.3434546569pp in this single seed internal run; no stable superiority claim.'},{'id':'complete_saved_matrix_math','impact':'supported','claim':'Full saved distance objective/masks/arithmetic and terminal ranking replay checked at stated scope.'},{'id':'full_parameter_gradient_correctness','impact':'unsupported','claim':'No independent full Q1 parameter-gradient reconstruction.'},{'id':'all_positive_gradient_sign_interpretation','impact':'needs_qualifier','claim':'Float32 sign census describes that arithmetic; tiny signs are precision-sensitive. Last65 inverted-nonmaximum candidate 973 favorable/1 adverse is independently reproduced in Float64.'},{'id':'official_or_general_gain','impact':'unsupported','claim':'No official-test, multi-seed, cross-dataset or SOTA qualification.'}],
 'audited_input_hashes':{p:'sha256:'+v['sha256'] for p,v in load('input_hashes_final.json').items()},'remote_audited_input_hashes':{p:'sha256:'+v['sha256'] for p,v in arrays['files'].items()},
 'trace_path':str(OUT/'.aris/traces/experiment-audit/2026-09-09_run01'),'independent_check_artifacts':['independent_text_results.json','independent_remote_arrays_results.json','independent_remote_statistics_results.json','independent_derivative_census_comparison.json','additional_text_results.json','input_hashes_initial.json','input_hashes_final.json','input_drift_during_audit.json','failed_attempts_final.json']}
write('EXPERIMENT_AUDIT.json',report)
md=f'''# MSVR310 Smooth-AP Q1 独立实验完整性审计

日期：2026-09-09。审计者：gpt-6-astra / max，独立上下文只读审查；`review_independence=same-family`、`acceptance_status=provisional`。审计任务标识 `/root/audit_msvr_smooth_ap_q1`。

**总体 WARN；确定性核验 PASS；科学资格 Q1_FAIL。** 已保存的完整 Q1 证据与运行实现相符，原门不通过。总体 WARN 来自未保存的真实模型梯度/历史字段重建缺口、单 seed 与重复内部开发协议，以及数值精度和轨迹限制；不是将科学负结果判成实验造假。

主输入：`C:/Users/gb/.codex_tmp/smooth_ap_q1_complete_20260909`。仓库：`C:/Users/gb/.trifusion_github_publish_22c3bee`。原远端 run：`/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4`。本报告相对代码路径以仓库为根；独立结果文件以本审计目录为根；`Q/` 代表主输入的 `q1/`。`snapshots/` 保存本次最终读取的全部本地文本副本。未改变训练、仓库、旧证据或门槛，未访问官方测试目录，未重做已关闭 M0 审计。

## A. Ground Truth Provenance：PASS

GT 是数据集文件名中的 identity/camera/scene。协议构建器只选原训练清单 `bounding_box_train`，按是否具有多个 scene 分层，再对各层排序身份按模3分折。该规则与全部1032条记录/155身份重建一致，不是把模型预测转换为真值。所有3096条模态路径的身份、scene、camera都与文件名一致，远端训练目录完整枚举恰有这3096个非空文件；没有枚举官方 query/gallery 目录。证据：`tools/build_msvr310_train_oof_protocol.py:12`、`:21`、`:35`、`:44`；`independent_text_checks.py:27`；`independent_remote_arrays_results.json:1`。

{foldtable}
每折的 source 与 heldout 身份、记录、完整路径均不相交。三份来源 Signal checkpoint 的折、source_ids、heldout_ids、SHA及完整模型状态均核实；它们全部1950条源训练日志的 sampled_record_indices 都在本折source集合中。Q1 role模型加载本折Signal后才以seed42初始化角色，不加载旧角色权重；冻结trunk与尾部块来自同一source teacher。证据：`tools/train_msvr310_trifusion_oof.py:27`；`tools/train_msvr310_signal_oof.py:70`；`tools/build_v12_complete_path_oof_targets.py:384`；`modeling/trifusion/signal_preserving_v8.py:33`。

评价过滤为 **same identity AND same scene**；所有异身份负例均保留，包括同scene负例。600个query来自60个有合法跨scene正例的身份；其余95身份/432记录虽无合法query，仍全部保留在gallery作为干扰。每折距离单独排序，再汇总query AP，不跨折比较特征。Signal源码 `utils/metrics.py:67` 的实际分支正是这个scene规则；函数旧docstring中的camera/Market文字不代表实际分支。实际调用及上游差异比较见 `tools/train_msvr310_trifusion_oof.py:196`。

限制：没有重新打开图像内容、重算原压缩包或外部标注真实性；训练期间“official_image_reads=0”与已审查路径一致，但该字段不是系统级访问追踪。故此PASS限于数据标签、文件身份、隔离及调用链的直接证据。

## B. Score Normalization：PASS

检索先对每条特征作标准L2单位化，然后使用平方欧氏距离。AP由每个真实正例所在排名的precision平均，分母为合法真实正例个数；mAP为所有合法query的AP均值，R1/R5/R10由first-match rank确定。没有以模型自身max/min/mean作为性能分母。训练Smooth-AP中的rank ratio是标准目标定义，且不冒充最终检索AP。代码：`tools/train_msvr310_trifusion_oof.py:212`；`tools/train_msvr310_signal_oof.py:223`；`tools/msvr_smooth_ap.py:30`。

本审计独立重算全部 **6000个query×output×endpoint结果、2,069,520个完整排名位置**，各输出全部AP、R1/R5/R10、60身份统计、修复/新增错误、折增益和三组固定identity bootstrap均匹配。Bootstrap独立用seed42/10000次完整身份簇重采样，保留query权重并取2.5%线性分位数。没有挑选样本、折、checkpoint或种子。实现和结果：`independent_text_checks.py:54`、`independent_text_results.json:1`、`independent_per_query_metrics.json:1`。

{table}
配对fused三折增益为 **-0.2001142871 / +0.2457622051 / +1.0777268742 pp**；身份bootstrap下界 **-0.1011244375 pp**。配对五门仅“各角色平均增益非负”通过，**1/5**。候选相对Signal五门 **0/5**，候选fused较Signal低 **0.3417902914 pp**，相对Signal bootstrap下界 **-2.2400609419 pp**。fused确实高于三个角色，严格最高门失败是因为低于Signal。原科学终态Q1_FAIL完全复现。

配对fused query AP改善/下降/不变为273/245/82，R1修复/新增错误13/6；身份均值改善/下降/不变29/26/5。没有把正均值隐藏，也没有以正均值替代完整门。

## C. Artifact Existence / Terminal Claims：PASS

全部57份原始接收文本、83,471,329字节均逐文件hash匹配inventory；其中M0/T0只做接收身份核对，未再次计算其工程/科学审计。Q1五阶段退出0、pipeline=`COMPLETE_VERIFIED_Q1_FAIL`，Q1 summary=`Q1_FAIL`；独立03:33:59远端进程检查确认wrapper48170、Q1 49157、Q1_CPU58836均已不存在。证据：`additional_text_results.json:1`；`remote_probe.result.json:1`；`tools/run_msvr_smooth_ap.py:31`。

核心哈希：

- 配置：`974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302`
- Q1 summary：`6502d1bfa1108bffb91dea1a663cdbdfb60975091dd35dfd4223ee45c0d7d32f`
- Q1 CPU：`2c797e9f50671b761e6d43d73da9306812bd9e37fa9e977a999906a0e051c9cd`
- 实际Smooth-AP目标：`9d7c01830493233183b2cc9366ec8742caab1f782208870badc8dc93a94b3b68`
- 实际Q1 runner：`67e25e4e5a6ccc97f1973778141baabc53cc9a1f4ccf7e5b0491d3977178c5b4`

{cktable}
六个最终checkpoint均远端重新读取，按241个baseline alias加231个role-state tensor重建每端472个完整state张量；final/strict-reload/frozen/Signal state哈希全部一致。全部retrieval feature/距离数组也远端读取并与自身SHA、完整rankings关联。保持注册FP32/56线程算法时，所有30组feature→distance逐位一致，distance→argsort也逐位一致。证据：`independent_remote_arrays_results.json:1`；`remote_array_checks.py:85`；保存/重载路径 `tools/train_msvr310_source_style.py:164`、`tools/train_msvr_instance_memory.py:94`。

**固定源码身份没有被Git HEAD混淆。** 启动提交`2e947a4325144e37fed638105ac954e7e54b5fe5`、Q1 summary记录的`3fe8e9c1942717188c677771b691717af19cc8b0`及远端审查时`9494df045de81a0becf40dc08f7eda4a4eedc1f0`，五个核心文件Git blob SHA完全相同。93项当前递归配置绑定、另17个Signal源码/7个base项目源码及CLIP权重直接核对。143份旧远端源码/协议快照与当前远端字节全部相同，因此可复用文本，但没有复用旧M0结论。

本地工作树与旧远端快照比较有83份字节一致、61份仅CRLF/LF不同、1份tracker内容变化。不能称“整个Windows工作树与远端字节相同”。初次广泛递归扫描还将历史配置/Signal相对路径错误地按本仓库根解析，所生hash mismatch被保留为无效探索结果；正式判定采用实际context链各自正确根路径与远端hash。

初始提供的`EXPERIMENT_PLAN.md`不存在；实际配置固定的是`refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md`，它存在且hash匹配。初审tracker停留5/6，审计期间执行器更新了AGENTS、handoff、tracker至完整终态。我重新读取最终§41.201和结果报告，完整保留三个文档前后hash，且快照保存最终所审文字。此次文档更新不算运行源码/原始证据变更。相关证据：`input_drift_during_audit.json:1`、`snapshot_difference_classification.json:1`、`failed_attempts_final.json:1`。

限制：Q1 checkpoint状态可独立重建，但未构建模型重新从图片产生features；原Q1只在reload后提取gallery，未保存“reload前全输出”。最初随机角色状态没有独立保存/重新构建，本审计核实的是init绑定、固定seed代码、三份来源checkpoint和末态；不将初始化日志hash当作独立随机初始化重演。

## D. Active Objective / Derivatives：WARN，已保存距离级核验PASS

实际调用是wrapper→`train_msvr_smooth_ap.run`→`fit`→`paired_objectives`，终点调用旧完整evaluator和旧两组五门汇总。新目标不是死代码。`fit`第66步起将`components.triplet_fused`替换为选中目标；候选记录`active_fused_metric=smooth_ap`，其余13项及权重未改变。第66步启用/入队但没有历史；第67步首次使用历史。全部六端均20epoch/260步，65步原目标+195步替换，194步实际有历史。证据：`tools/train_msvr_smooth_ap.py:130`、`:143`、`:155`、`:246`、`:262`；六份`memory_steps.jsonl:66`、`:67`。

Smooth-AP用单位表示的`s=1-d²/2`，tau=.01；全部同身份候选位置为正例，但排除当前anchor自身位置，每个正例自己的self-comparison也排除。当前重复抽样产生的不同view位置保留；历史对记录去重并排除当前记录，位置映射使用该历史batch中最后出现的位置。其他身份全部为负，包括同scene。历史候选不作anchor。该式与Brown等人标准Smooth-AP的rank approximation吻合，不是本项目原创loss；原论文公式和温度设置已核对：[Smooth-AP原论文](https://arxiv.org/html/2007.12163v2)。代码：`tools/msvr_smooth_ap.py:44`、`:49`、`:55`；`tools/msvr_freshness_probe.py:49`；`tools/msvr_instance_memory.py:1`。

历史坐标按原冻结field与角色入口RNG，用当前encoder参数重新编码；原当前图包含当前anchor和当前candidate的梯度。历史叶子偏导另取同一定义目标，按原64-record历史group重放encoder并作VJP，在一次optimizer更新前加到189个encoder参数；融合层没有可训练参数，其他14个classifier张量不应收到历史fused VJP。相关实现：`tools/probe_msvr_role_set_gradients.py:30`、`:40`；`tools/train_msvr_smooth_ap.py:147`、`:195`、`:231`；`tools/probe_msvr_history_candidate_gradients.py:42`。这些源码与实际memory日志/非零角色贡献/应用后norm恒等式相互支持。

独立确定性核验具体覆盖：

- 全1560更新真实队列、年龄、current排除、64 anchor身份/scene、正例计数、采样序列、像素SHA配对、AMP有限记录与120条epoch日志。
- 全 **116,501,504** 个已保存四空间距离元素有界读完，无尾部遗漏；全部 **99,840** 个anchor Smooth-AP从Float64 NumPy rank sums重算，最大AP误差 **2.550403574996807e-7**，最大hard/AP/batch标量误差 **1.2518931047367232e-7**，均在原2e-6内。四空间均核对有限/非负；标量/导数重建针对fused空间。
- 每个hard/Smooth-AP目标、每个Float64/Float32 dtype均比较全部 **29,125,376** 个fused current/history距离位置。独立解析Smooth-AP导数与部署自动微分最大差 **3.5344990823027445e-17 / 6.886213898757432e-8**；hard导数完全一致，当前self距离导数精确0。没有用合成小矩阵替代全Q1覆盖。
- 全14项加权账本最大误差 **6.283322973033023e-7**；完整保存统计的正负/跨scene/更难候选/违反关系计数均重新计算一致；学习率20个固定值各端一致。

独立脚本：`independent_text_checks.py:103`、`remote_array_checks.py:58`、`remote_statistics_checks.py:17`。结果：`independent_remote_arrays_results.json:1`、`independent_remote_statistics_results.json:1`、`additional_text_results.json:1`。

**没有独立重建的内容必须保留：** Q1的`direct_single_group_check`全为{{}}，本审计未重做M0的单group检查；历史fields/RNG向量、逐步feature leaves、参数梯度和optimizer状态未保存。CPU距离导数无法恢复encoder Jacobian，也无法恢复每步模型总梯度。role norm/cosine、RNG/buffer保留、逐位reencoding及梯度加法是运行见证；其摘要算术全部通过，但不等于独立GPU重演。203/203是全程累计至少一次非零覆盖，不是每步全部非零。未保存logits与残差距离，其他13项监督只做代码定义和全账本核对，未从模型输出独立复算全部CE/Triplet。

## E. Scope / Cost / Numerical Boundaries：WARN

证据范围是固定seed42、MSVR310训练身份内部三折、两端、五个输出。没有官方测试、其他数据集成绩、新随机种子或训练重复。旧版本已多次使用该内部协议，因而600query/60身份不是全新、无适应的最终测试；10000 bootstrap重采样也不是10000训练seed。完整覆盖此固定Q1可以接受，跨数据集/稳定泛化/SOTA主张不能接受。

像素、记录与初始化绑定完全配对，但在两端目标仍相同的steps1–65里，三折从step2开始有数值差异。最大total loss差 **0.0018461943 / 0.0014842749 / 0.0021481514**；最大单component差 **0.0021758080 / 0.0016922951 / 0.0034337044**。这在合同明确不保证CUDA轨迹逐位相同的边界内，却表示不能把单seed的+0.3435pp全部识别为无噪声因果效应。不能由日志确定具体底层原因，也不能用同图重复反传代替独立训练方差。

另做Float64向量代数时，检索距离最大偏差 **1.2398810658176274e-6**，某些接近相同距离的rank位置随精度改变；注册FP32路径全部逐位重现，因此原门与指标不作替换。每输出完整差异计数列于`verification_compact_summary.json`，不存在“任意dtype下完整排序都相同”的结论。

Smooth-AP正例导数计数也必须标明dtype。所有120个phase/category的**关系数量**都与执行器一致，所有严格反序及严格非最大反序类别的**符号计数**亦一致；但all-positive/cross-positive类别的48组符号计数在Float64与Float32间不同，形成96个正/负字段差异。candidate末65步全部正例，Float32为86990正/7882负；Float64为94866正/6负。此处极小导数受相减抵消和精度影响，而全距离位置导数最大绝对差仍仅6.89e-8。不得将每个极小负号都解释为实质错误监督。记录的Float32统计本身并非伪造；其解释必须受精度和量级限制。证据：`independent_derivative_census_comparison.json:1`。

对当前具体来源主张，末65步非最远反序正例曝光control1299→candidate974，跨scene1061→781；candidate974中973个距离下降方向/1个相反方向，跨scene780/1，均由独立Float64解析census确认。hard对严格非最大正例位置的直接导数为0是hard选择定义的结果。它支持“给多数此前未直接选中的反序正例提供直接目标导数”，不支持完整来源图库校准或真实参数更新改善。全部数量是重复关系曝光，非独立图片，也非不同目标在同一参数状态上的对照。

末65步，batch hard **0.05734946→0.06248964**、expanded hard **0.14114936→0.15871105**、Smooth-AP loss **0.00817222→0.00667341**；active total **0.91693943→0.78666309**的定义不同，不能直接据后者降低认定训练更优。最终报告已披露此区别。

资源：每端16640个当前record曝光，六端99840；两侧每侧额外fresh角色record前向292800，其中192为首步零更新核验，历史fresh为292608。历史VJP前向control265344、candidate292608。epoch训练耗时 **6201.269009→6557.227357秒，+5.7401%**；Q1整阶段12894.751186秒，CPU验证子进程21.276343秒。峰值reserved7034–7064MiB，无新增推理参数。相同更新数不等于相同算力，重编码/VJP不是免费训练。计数和耗时均由完整日志求和，见`additional_text_results.json:1`。

## F. Evaluation Classification：PASS

| 内容 | 分类 | 可支持的结论 |
|---|---|---|
| 600query内部OOF检索 | real_gt | 本次固定Q1的真实标签检索性能/原门结果 |
| batch/history近似AP、hard及其他身份监督 | real_gt来源的训练诊断 | 当前优化行为，不是完整gallery mAP |
| 距离解析导数、checkpoint/state、数值恒等式 | numerical consistency diagnostic | 指定保存数组和实现一致性；不是独立模型梯度真值 |
| T0合成矩阵测试 | synthetic_proxy | 公式/实现测试；本次只追踪调用与类别，未重做M0 |
| 官方、多seed、跨数据集性能 | 未执行 | 无对应新性能结论 |

不应将本次有监督目标标成self_supervised_proxy，也不应把梯度一致性测试包装成独立real-GT任务性能。

## Claim Impact and Closure

支持：固定Q1完整结束；+0.3434546569pp配对均值；完整两组原门FAIL；所有可保存距离/排名/状态证据的上述确定性重建通过；非最大反序正例直接导数覆盖增加。

需要限定：真实模型历史参数贡献是执行时见证；完整正例tiny-gradient符号限Float32语义；相同像素/seed绑定不等于位级轨迹一致；内部训练协议不能替代新未知分布评价。

不支持：全程参数梯度独立重构、所有13个非fused项完整模型重算、跨scene关系主导真实参数更新、全部正例都被正确有力地优化、已稳定胜过Signal、官方或SOTA晋级。

没有为提高分数要求重训，也不调整门、温度、尺度、epoch或seed。本审计不授权任何后继GPU实验。最小收束动作是发布本审计并保留上述范围、数值与证据边界；已有终态tracker修订已核读通过。

## Evidence and Failed Attempts

完整输入hash为`input_hashes_initial.json` / `input_hashes_final.json`（最终295个本地文本输入），额外远端76个直接访问文件hash位于`independent_remote_arrays_results.json`；143份旧快照身份与93个当前配置链pin位于`remote_probe.result.json`。`snapshots/`只含文本，数组/图片/checkpoint留在远端。

独立数组检查首尝试因原transport的60秒静默timeout没有取得终态，原CPU进程59782确认结束后才重试，绝未当成功证据。重试新增定期进度输出、用等价Float64矩阵代数避免临时巨型广播数组，128.101393秒完成，0模型/0图像/0optimizer/0CUDA初始化；独立统计census25.302808秒完成。全部失败脚本、stderr、进程观察与重试原始stdout/JSONL保留，不覆盖旧输出。

其他审计器失误也保留：初次假定未分层round-robin导致本地checker失败，读取真实固定split_rule后修正为分层规则并全量通过；一次辅助PowerShell引号导致Python SyntaxError；文件路径读取失败与输出截断。它们未改变科学数据，详见`failed_attempts_final.json`。

独立结论由本审计者直接阅读源码/文件并判断；执行器补充的来源/排名/导数分析仅作为可验证主张，不替代审计。原样最终回复保存`final_response.md`，调用/模型元数据保存`.aris/traces/experiment-audit/2026-09-09_run01/`。
'''
write('EXPERIMENT_AUDIT.md',md)
attempts=load('failed_attempts.json')
attempts.extend([
 {'action':'Generic recursive local pin discovery','result':'Exploratory checker treated comparator-relative and historical reference hashes as project-root paths; resulting mismatch entries are not active-run failures. Correct remote context-root checks passed.','evidence':['inspect_inputs.stdout.txt','recursive_pin_checks.json','remote_probe.result.json']},
 {'action':'Independent text checker attempt1','result':'Auditor assumed one unstratified sorted-ID round robin; failed at fold identity assertion. Real fixed protocol stratifies multi-scene/single-scene identities. Corrected auditor rule, full rerun PASS.','evidence':['independent_text_checks_attempt1.py','independent_text_checks.stderr.txt','independent_text_checks_attempt2.stderr.txt']},
 {'action':'Search guessed protocol builder filename','result':'tools/build_msvr310_internal_protocol.py absent; rg --files located tools/build_msvr310_train_oof_protocol.py, which was read directly.'},
 {'action':'Remote arrays attempt1','result':'Paramiko channel timeout after60s without stdout; no terminal result accepted. Original read-only CPU process59782 observed at last fold, later confirmed absent before retry. No training process stopped or restarted.','evidence':['remote_array_checks_attempt1.py','remote_array_checks.stderr.txt','remote_timeout_observation.json','remote_audit_process_detail.json','remote_timeout_observation2.json']},
 {'action':'Auxiliary inline local snapshot classifier','result':'PowerShell nested quote handling caused SyntaxError: unterminated string literal. Replaced by persisted finalize_checks.py; classification completed. No evidence edits.'}
])
write('failed_attempts_final.json',attempts)
print(json.dumps({'report':str(OUT/'EXPERIMENT_AUDIT.md'),'verdict':'WARN','deterministic_checks_status':'pass','scientific_qualification':'Q1_FAIL'},ensure_ascii=False))
