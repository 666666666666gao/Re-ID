"""Materialize the independent review and complete forensic trace locally."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
REQUEST=Path('C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_request_20260908.md')
def read(name):return json.loads((OUT/name).read_bytes())
def line(name,needle):
    for i,s in enumerate((OUT/name).read_text(encoding='utf-8').splitlines(),1):
        if needle in s:return f'{name}:{i}'
    raise ValueError((name,needle))
ar=read('independent_arithmetic.json');ck=read('independent_checkpoints.json');lr=read('local_report_checks.json');ri=read('remote_inventory.json');li=read('local_inventory.json');sc=read('supplementary_checks.json');bl=read('binding_and_launch_checks.json')
now=datetime.now(timezone.utc).isoformat()
refs={
 'ar':line('independent_arithmetic.json','"status"'),
 'folds':line('independent_arithmetic.json','"fold_provenance"'),
 'overfit':line('independent_arithmetic.json','"weighted_CE_floor"'),
 'direct':line('independent_arithmetic.json','"direct_recorded_norm_checks"'),
 'ck':line('independent_checkpoints.json','"m0_checkpoint_checks"'),
 'source':line('independent_checkpoints.json','"source_checkpoint_checks"'),
 'hashes':line('independent_checkpoints.json','"cpu_receipt_file_hash_checks"'),
 'label':line('supplementary_checks.json','"all_1032_records_match_original_label_evidence"'),
 'tie':line('supplementary_checks.json','"independent_exact_tie_derivative"'),
 'scalar':line('supplementary_checks.json','"independent_random_distance_checks"'),
 'local':line('local_report_checks.json','"status"'),
 'launch':line('binding_and_launch_checks.json','"primary_launch_files"'),
 'stage':line('binding_and_launch_checks.json','"closed_stage_receipts"'),
 'files':line('remote_inventory.json','"files"'),
 'binding':line('remote_inventory.json','"bindings"'),
 'endings':line('remote_inventory.json','"local_remote_differences"'),
 'process':line('remote_inventory.json','"original_processes"'),
}
checks={
 'A':dict(status='PASS',name='ground_truth_provenance_and_complete_path_isolation',evidence=['tools/build_msvr310_train_oof_protocol.py:12-55','tools/train_msvr310_signal_oof.py:68-109','tools/train_msvr310_trifusion_oof.py:27-55','tools/train_msvr_role_set.py:85-115',refs['label'],refs['folds'],refs['source']],details='1032 dataset filename records and 3096 training-modality paths match the saved label evidence; scene-stratified source/heldout partitions and source-label maps reconstruct exactly. All 780 registered source batches independently regenerated. Three B0 checkpoints and their 1950 saved source-training batch memberships match source-only fold bindings. No model-produced identity targets are used.'),
 'B':dict(status='PASS',name='loss_definition_normalization_and_warmup',evidence=['tools/msvr_role_set_relations.py:20-56','tools/train_msvr_role_set.py:125-160','tools/run_signal_preserving_v5.py:99-142','tools/run_signal_preserving_v5.py:1580-1617',refs['tie'],refs['scalar'],refs['overfit'],refs['local']],details='Fixed mean of selected unsquared-Euclidean margin-0.3 hinges, with detached fused/role argmins, unique positions, original hard-term tie derivative, and mean over 64 anchors. No prediction-max score normalization. All saved M0 fused objectives and all 14-term weighted scalar ledgers reconstruct. Warmup is two steps for M0 and 65 for registered Q1. Analytic weighted label-smoothed CE floor is 0.5857136327437849.'),
 'C':dict(status='PASS',name='real_execution_files_hashes_full_endpoint_coverage',evidence=['tools/run_msvr_role_set.py:31-65','tools/train_msvr_role_set.py:319-378',refs['binding'],refs['launch'],refs['stage'],refs['ck'],refs['hashes'],refs['local']],details='137 recursive hash-binding entries, explicit nested bindings, primary launch-commit bytes, 29 intake files, six final checkpoint files, reconstructed final/frozen states, 37 original CPU receipt files, eight training log events and all 248 saved updates match. T0/M0/M0_CPU exited zero. No number mismatch or missing claimed M0 artifact found. Model output equality remains a runtime witness as qualified under D.'),
 'D':dict(status='WARN',name='runtime_paths_gradients_and_reconstruction_limits',evidence=['tools/msvr_freshness_probe.py:48-60','tools/probe_msvr_role_set_gradients.py:29-50','tools/probe_msvr_history_candidate_gradients.py:43-50','tools/train_msvr_role_set.py:146-262','tools/train_msvr_role_set.py:338-347','tools/verify_msvr_role_set.py:124-137','refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:29-33',refs['direct'],refs['ck']],details='Both endpoints call the complete current loss/backward and historical candidate VJPs before one optimizer step. Direct total-loss/full-graph comparisons cover only the first historical group at step 4 of each capacity endpoint, in the 189-encoder-tensor domain; recorded maximum relative error is 1.6810499980311218e-05. Norm identities and grouping arithmetic are independently checked. Original gradient vectors, RNG/cached fields, augmented tensors, initial role-state snapshots and before/after reload output arrays are not retained, so their model-level equality and per-step nonzero coverage cannot be independently regenerated from saved artifacts.'),
 'E':dict(status='PASS',name='registered_scope_pairing_fairness_and_cost',evidence=['refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:7-19','refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:25-49','tools/train_msvr_role_set.py:71-105','tools/train_msvr_role_set.py:356-378',refs['folds'],refs['ar'],'local_report_checks.json:1-20'],details='The full registered M0 is completed: six eight-update capacity endpoints and two 100-update fixed-batch endpoints, one seed. Paired sampled records and all three modal pixel SHA values match at every step. M0 used 738 unique source records in total, with repeated views retained; overfit uses 53 unique records and zero historical candidates. Capacity observed history at most 221 candidates and age 5; age 8 is exercised in 780-step metadata replay. Same architecture/optimizer/other losses; no additional trainable module. The candidate changes relation coverage and hard-hinge weighting together. M0 engineering success gives no Q1 or scientific qualification.'),
 'F':dict(status='PASS',name='evidence_type_and_claim_classification',evidence=['results/MSVR310_ROLE_SET_V1_M0_2026-09-08.md:9-21','tools/check_msvr_role_set_relations.py:62-68','tools/verify_msvr_role_set.py:300-307',refs['ar'],refs['label']],details='Dataset labels and supervised source losses are real_gt evidence; baseline equality and graph/VJP comparisons are engineering synthetic_proxy comparisons, not retrieval accuracy; synthetic CPU tensor checks are simulation_only. Deterministic replay and original runtime assertions are explicitly distinguished. No official-test or partial Q1 scores were consumed.')
}
claims=[
 dict(id='complete_registered_m0_248_updates',impact='supported',basis='Eight complete saved step sequences, arrays, epoch events and exit receipts.'),
 dict(id='source_only_complete_path_and_paired_exposures',impact='supported_with_evidence_boundary',basis='Dataset metadata, source checkpoint bindings, source-only code paths and all sampled indices verified; pixels compared by recorded SHA, not regenerated.'),
 dict(id='fixed_role_set_objective_and_loss_floor_ratios',impact='supported_deterministically',basis='Independent raw-array arithmetic, analytic loss floor and isolated CPU distance-gradient tests.'),
 dict(id='all_203_trainable_tensors_cumulative_nonzero',impact='supported_runtime_witness_only',basis='203-name parameter domain and final trainable tensor values are verified; per-step gradients/nonzero-name sets are not saved.'),
 dict(id='six_final_and_frozen_checkpoint_states',impact='supported_deterministically',basis='Independent alias reconstruction from three source checkpoints matches all six full/frozen final hashes.'),
 dict(id='strict_reload_all_outputs_bitwise_equal',impact='needs_runtime_witness_qualifier',basis='The GPU runner executes strict reload and asserts all five outputs on eight source records per endpoint; neither before nor after output arrays are retained.'),
 dict(id='direct_total_gradient_matches_current_plus_history_vjp',impact='supported_runtime_witness_with_narrow_scope',basis='Six first-single-history-group comparisons only, each at capacity step 4 and 189 encoder tensors; recorded scalar/norm arithmetic is checkable.'),
 dict(id='full_248_step_model_gradient_reconstruction',impact='unsupported_and_not_claimed',basis='Saved distance arrays do not encode parameter Jacobians, original fields/RNG, or per-step model states.'),
 dict(id='role_diversity_alone_causes_gain',impact='unsupported',basis='Negative coverage and mean-hinge softening change together; the registered plan explicitly disclaims sole attribution.'),
 dict(id='q1_retrieval_or_method_or_official_qualification',impact='not_evaluated',basis='This is M0 engineering evidence; Q1 artifacts and official images were outside the audit.')
]
hashes={path:'sha256:'+v['sha256'] for path,v in li['hashes'].items()}
hashes.update({path:'sha256:'+v['sha256'] for path,v in ri['files'].items()})
for path in ['evidence/vehicle_query_protocol_labels_20260905.json','evidence/msvr310_dataset_install_20260905.json','tools/build_msvr310_train_oof_protocol.py','tools/audit_vehicle_query_protocol_labels.py']:
    snap=OUT/'snapshots/remote/root/autodl-tmp/trifusion-v2/TriFusion-ReID'/path
    hashes['/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'+path]='sha256:'+hashlib.sha256(snap.read_bytes()).hexdigest()
for path in [REQUEST,Path('C:/Users/gb/.codex/skills/experiment-audit/SKILL.md'),Path('C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md'),Path('C:/Users/gb/.codex/skills/shared-references/reviewer-independence.md'),Path('C:/Users/gb/.codex/skills/shared-references/experiment-integrity.md'),Path('C:/Users/gb/.codex/skills/shared-references/review-tracing.md')]:
    hashes[str(path)]='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()
limits=[
 'Same-family fresh-context review; semantic acceptance is provisional and backend model identity is not independently attested.',
 'No training or model forward/backward was run by the reviewer. CPU tensor arithmetic and checkpoint deserialization used at most two compute threads.',
 'No model gradients, original role-entry RNG states, frozen-field cache tensors or intermediate model states are saved for all-step independent replay.',
 'Pixel equality, cumulative nonzero gradients, precise optimizer-step execution and strict model-reload output equality rely on original runtime assertions corroborated by source code, arrays, file hashes and terminal states; not an external execution attestation.',
 'M0 has shorter warmup and history than Q1. Both overfit endpoints have zero historical candidates; the 512-entry/age-8 contract is not a claim that M0 exercised a full 512-candidate GPU history.',
 'Raw images, image-content hashes, upstream archive authenticity, actual GPU instruction execution and performance were not re-measured. Dataset filenames, pairing, labels, file presence and source checkpoint metadata were checked.',
 'No Q1 scientific outcome, official-test qualification, novelty attribution, cross-dataset result or SOTA claim is evaluated.'
]
audit=dict(audit_skill='experiment-audit',verdict='WARN',overall_verdict='warn',integrity_status='warn',engineering_status='PASS',fixed_m0_status='PASS_WITH_RUNTIME_WITNESS_LIMITS',deterministic_checks_status='pass',deterministic_verification={'status':'PASS','acceptance_status':'accepted','scope':'137 recursive bindings, 29 intake texts, 780 reconstructed source batches, all 248 saved M0 steps, 4945920 distance elements, scalar ledgers, six reconstructed checkpoint states, 37 receipt hashes, metadata and synthetic mathematics','excludes':'Independent real-model forward/backward, original per-step gradient vectors, augmented pixel regeneration and before/after reload features'},scientific_qualification='not_evaluated',reason_code='bounded_model_runtime_witness_not_independently_reconstructable',summary='The completed registered M0 passes independent saved-artifact engineering checks. Preserve a WARN on model-level runtime assertions that cannot be rebuilt from retained artifacts; no M0 numerical or scientific-source defect found.',date='2026-09-08',generated_at=now,auditor='gpt-6-astra max fresh native Codex reviewer',executor_model='codex-gpt-6-astra',executor_family='openai',reviewer_model='gpt-6-astra',reviewer_family='openai',reviewer_reasoning='max',model_identity_basis='Requested native Codex reviewer route and installed policy; backend identity not independently attested',review_independence='same-family',acceptance_status='provisional',agent_id='/root/audit_msvr_role_set_m0',reviewer_task='/root/audit_msvr_role_set_m0',trace_path=str(OUT),run_commit='26c97390704c629237687d263b4381f5584cbe97',checks=checks,evaluation_types=['real_gt','synthetic_proxy','simulation_only'],claims=claims,claim_limits=limits,audited_input_hashes=hashes,verification_results=['local_inventory.json','remote_inventory.json','independent_arithmetic.json','independent_checkpoints.json','supplementary_checks.json','binding_and_launch_checks.json','local_report_checks.json','snapshot_manifest.json','raw_text_snapshot_recovery.json'],attempted_check_failures=['attempt_01_transport_failure.txt','attempt_02_remote_inventory.command.json','attempt_02_remote_inventory.stderr.txt','attempt_03_snapshot_failure.txt','attempt_04_report_packaging_failure.txt','attempt_04_reference_validation.json'],boundaries={'remote_scientific_file_writes':0,'training_runs':0,'model_forward_jobs':0,'model_backward_jobs':0,'gpu_initializations':0,'max_cpu_compute_threads':2,'process_mutations':0,'package_installs':0,'official_image_reads':0,'q1_artifact_reads':0,'agents_spawned':0},actions=['Retain the engineering-only qualification and the runtime-witness qualifiers in downstream summaries.','Use the original full Q1 terminal artifacts for any later scientific audit; this M0 review does not authorize or infer a scientific result.','No patch, restart, retraining or extra experiment is required by this M0 audit.'])
table='\n'.join('| {endpoint} | {steps} | {unique_records} | {matrix_elements:,} | {history_candidates} | {max_history} / {max_age} | {extra_history_vjp_record_forwards} |'.format(**x) for x in ar['endpoint_checks'])
report=f'''# MSVR310 role-set v1 完整 M0 独立工程完整性审计

日期：2026-09-08。审计者：gpt-6-astra / max，新上下文原生 Codex reviewer；`review_independence: same-family`，`acceptance_status: provisional`。模型归属来自调用规范，未取得独立后端身份认证。

**总体判定：WARN。工程判定：PASS。确定性核验：PASS。Q1 科学资格：未评价。**

完整登记 M0 的文件、来源记录、距离、关系、损失账本、终点权重和阶段退出均能对应。未发现本轮 M0 的数字错报、虚构文件、身份目标来自模型输出、归一化作弊或需要修改科学代码的问题。WARN 来自可复现边界：部分关键模型级结论只有原运行断言和范数汇总，现有保存产物不能独立重建原模型梯度、缓存/RNG 或重载前后输出。

审计对象为执行提交 `26c97390704c629237687d263b4381f5584cbe97`，远端运行 `/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739` 的 T0/M0/M0_CPU。源码引用相对于 `C:/Users/gb/.trifusion_github_publish_22c3bee`；审计 JSON/脚本引用相对于本报告目录。远端精确文本快照保存在 `snapshots/remote/`，本地原始副本在 `snapshots/project/` 与 `snapshots/intake/`。完整输入 SHA 在 `EXPERIMENT_AUDIT.json`，快照 SHA 在 `snapshot_manifest.json`。

## 分项结论

| 检查 | 状态 | 结论 |
|---|---|---|
| A. 标签与完整路径身份隔离 | PASS | 真实数据文件名标签、三折 source/heldout 与 Signal 来源绑定相符；无伪 GT |
| B. 目标、归一化与 loss floor | PASS | 固定均值 hinge、原 hard tie、预热和其余 13 项账本均可重算 |
| C. 执行、文件、更新与终点 | PASS | 全 248 步和六权重状态匹配，阶段退出与原文本一致 |
| D. 模型路径与梯度 | WARN | 活跃调用链明确；原参数梯度、缓存/RNG 与输出逐位证明只能作为有界运行见证 |
| E. 范围、公平性与成本 | PASS | 完整执行所登记的 M0，配对资源一致；没有科学晋级结论 |
| F. 证据分类与声明 | PASS | 区分真实标签、工程自比较、合成数学和独立确定性重算 |

## A. 标签与完整路径隔离 — PASS

原始标签证据 `evidence/vehicle_query_protocol_labels_20260905.json` 的 SHA 与协议登记 `c835d20478b817a54b7710463269186af2619cab3e38850534b01f3aaee6e3c8` 一致；1032 条训练记录的 identity/camera/scene 均与文件名和原始标签逐项一致，vis/ni/th 三模态共 3096 个路径均存在且完整配对。仅枚举训练文件名和元数据，没有打开图片字节。训练身份与官方 query/gallery 的标签身份集合不相交。依据：`tools/build_msvr310_train_oof_protocol.py:12-55`，`tools/train_msvr310_signal_oof.py:68-76`，`{refs['label']}`。

三折来源身份为 103/103/104，来源记录 672/683/709；对应留出身份 52/52/51，完整图库记录 360/349/323，合法 query 210/207/183。独立从 scene eligibility 分层轮转规则重建划分，并逐一重算所有图库/合法正例元数据。这些是既定协议的标签检查，不是本次已运行的留出评价。依据：`tools/build_msvr310_train_oof_protocol.py:24-123`，`{refs['folds']}`。

来源 Signal 是各折固定 50 epoch 的 B0 checkpoint；独立加载三个权重，核对 source/heldout/fold 元数据、文件 SHA 和完整 Signal 状态 SHA，并检查其保存的每折 650 个来源训练 batch，共 1950 步均只引用该折来源记录。M0 `build_model` 每端重新构建，载入该折 Signal，角色 seed42 随机初始化且 `role_weights_loaded=False`；不读取旧角色终点或全训练身份 teacher。`HierarchicalFrozenSignalBackbone` 冻结同一 Signal，三个 frozen tail 引用也来自它，因此被测基座/角色整条路径均继承该折来源隔离。依据：`tools/train_msvr310_trifusion_oof.py:27-55`，`modeling/trifusion/signal_preserving_v8.py:33-77`，`tools/build_v12_complete_path_oof_targets.py:384-411`，`{refs['source']}`。

训练分类标签是 source label map 的连续编号，关系掩码用真实 identity；二者同属同一等价关系，类别 0 合法。scene 在训练关系日志中用于描述，未误用于排除所有同 scene 负例。M0 的加载器、预检和重载抽样均走 source records；留出读取只在 `m0=False` 分支。依据：`tools/train_msvr_role_set.py:85-115`、`:319-355`，`tools/train_msvr_instance_memory.py:53-91`。

## B. 目标、归一化与账本 — PASS

每个 anchor 保留同一最难正例；负例集合取 fused 与三个完整角色（Signal+角色）各自最近合法负例的位置去重并集。使用单位特征的未平方欧氏距离、margin 0.3。记集合大小为 n，原 hard hinge 为 h，去掉 fused 首选位置后的额外位置集合为 E，则逐 anchor 目标为 `(h + sum(relu(d_pos - d_neg[j] + 0.3), j in E)) / n`，再对 64 anchors 求均值。索引和角色距离只负责提议，不反传；最终 loss/两侧导数在 fused 空间计算。依据：`tools/msvr_role_set_relations.py:20-64`。

该分母是所选关系数，不是模型输出最大值等分数重标定。标准特征 L2 归一化和既有等能量拼接出现在表示定义中，也不是把检索成绩缩放到高值。额外关系与最难关系降权是同时发生的干预，不能把将来的差异独归因于异构角色。依据：`modeling/trifusion/signal_preserving_v8.py:494-516`，`refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:11-15`。

原 hard 项分别做 current/history 极值，再用 `torch.maximum/minimum` 合并，保留跨 current/history 相等时的导数分配。独立 CPU toy 指定了应为 ±1/8 的精确导数，实际代码完全相符；另 12 组独立构造距离的集合/均值标量和距离导数均与显式逐项参考式相符，并重做合成候选 VJP 链式法则检查。它们仅是合成数学检查。依据：`{refs['tie']}`，`{refs['scalar']}`，`supplementary_checks.py:1-104`。

M0 前两次更新用原批内 triplet，第三次起替换 fused 项；第一次历史候选出现在第四次更新。登记 Q1 的对应预热为 65 步，不能从 M0 的短预热推断长训练行为。其余 13 项仍从同一个 `ExpertFormationV8Criterion` 计算，权重沿用原合同。独立重算所有 248 步保存的 fused 距离目标，最大误差 **{lr['maximum_independent_loss_error']:.12g}**；独立从已保存 14 个分量重建加权总损失，最大误差 **{lr['maximum_total_loss_ledger_error']:.12g}**。未重新生成其余 13 项的模型前向。依据：`tools/train_msvr_role_set.py:125-160`，`modeling/trifusion/signal_preserving_v8.py:690-742`，`tools/run_signal_preserving_v5.py:99-142`，`{refs['local']}`。

103 类、label smoothing 0.1 的加权 CE 熵下界为 **0.5857136327437849**，ID 总权重为 0.75。两端首 loss 均为 4.122129917144775；末 loss 为 control 0.5881941318511963、role_set 0.5881949663162231。由 `(last-floor)/(first-floor)` 得 **0.0007014160404002304 / 0.0007016520038614711**，均低于登记 0.1。没有使用中途最低 loss 或模型自身最大值。依据：`tools/run_signal_preserving_v5.py:1580-1617`，`{refs['overfit']}`。

## C. 原始文件、执行与六终点 — PASS

递归配置展开包含八份配置，其中 R1/原 source 配置作为来源链保留；137 条 project/fixed/Signal 源码 hash 绑定全部在远端匹配，另核对 CLIP、baseline summary、protocol、metadata 等嵌套 SHA。新合同与 14 个直接 project bindings 共 15 个文件和原执行提交逐字节相同；核心被调用依赖与执行提交语法树一致。Signal commit 和 diff hash 也匹配。依据：`{refs['binding']}`，`{refs['launch']}`，`binding_and_launch_checks.json:1-29`。

本地五个旧源码依赖的工作副本为 CRLF，而固定远端为 LF，导致本地直接字节 SHA 不同；独立换行归一和 AST 比较相同，远端原字节符合合同。不能用本地 checkout 字节冒充该五项运行字节。依据：`{refs['endings']}`。

29 份 M0 原始文本共 **5,717,997 字节**，本地接收副本、仓库证据副本与远端都与 intake manifest 匹配。summary SHA 为 `203f2e572d40e00d0d41842289746ad2c077d8d7e97ada5fe237214f1b161a38`。原 CPU 收据列出的 **37 文件**被再次重新散列；六个角色 checkpoint 共 **196,617,690 字节**，所有原始距离数组共 **19,783,680 字节**。依据：`local_inventory.json` 中 `intake_checks`，`{refs['hashes']}`，`local_report_checks.json:1-20`。

每个 checkpoint 的 231 个角色状态项和 241 个 frozen baseline 别名项被从真实来源 Signal 重建；六个完整状态 SHA、冻结状态 SHA、Signal 状态 SHA 均与训练终点/严格重载收据一致。训练参数域是 203 个张量，其中 encoder 189，另 14 为七组可训练 BN weight/分类器；冻结 BN bias 七项和 buffer 单独核对。每折两端 trainable parameter count 相同，fold0/1 为 8,076,300，fold2 为 8,102,412，差异来自来源分类数。依据：`tools/train_msvr310_source_style.py:164-174`，`tools/train_msvr_instance_memory.py:94-110`，`{refs['ck']}`。

原阶段收据：T0 用时 4.8872329947 秒、M0 **614.7843318302 秒**、M0_CPU 8.3967608288 秒，均 exit 0，结束后原 wrapper 才启动 Q1。八个 epoch 日志事件和保存的 training history 全等。只读进程观察时间为 {ri['generated_at']}：原 wrapper 35302 和 Q1 36320 存活，已完成阶段的原 PID 不存在。没有打开 Q1 目录、Q1 log 或分数。依据：`{refs['stage']}`，`{refs['process']}`，`{refs['local']}`。

## D. 实际梯度路径与有界见证 — WARN

`ViewFields.capture` 保存当前真实 source batch 的 frozen anchor/reference/baseline 字段，以及角色入口的 CPU/CUDA RNG；同 record 重复采样时 queue 留最后一次视图。历史重编码读取原完整 64 样本组，在当前 encoder 参数下重跑，保留其原 batch/RNG 语义；不是将旧特征作为当前坐标。`refresh_all` 两端都计算 fused 与三个角色，保存距离也确有四矩阵。依据：`tools/msvr_freshness_probe.py:48-60`，`tools/msvr_instance_memory.py:15-37`，`tools/probe_msvr_role_set_gradients.py:29-50`。

当前完整目标先 backward；候选侧把新鲜历史向量作为 leaf 求上游导数，再在原历史完整 batch 上用 `encode_graph` 计算 encoder VJP。循环只跳过上游为零的历史组；随后把历史导数准确加到现有当前参数梯度，unscale 后执行一次 AdamW step。原 current graph 包括当前样本既作 anchor 又作 current candidate 的导数，历史路径只有 candidate 导数，没有历史 anchor。分类器/neck 不决定 fused 检索嵌入，所以历史 VJP 限于 encoder 189 张量符合当前模型定义。依据：`tools/train_msvr_role_set.py:146-247`，`tools/probe_msvr_history_candidate_gradients.py:43-50`，`modeling/trifusion/signal_preserving_v8.py:614-653`。

六个容量端均在第 4 步首个单历史组执行直接完整图检查：重新计算 fused 与三个完整角色，四输出逐位一致断言；原总 loss 的直接 encoder 导数与当前导数+历史 VJP 相比较。最大记录相对 L2 误差为 **1.6810499980311218e-05**，小于 0.005。它是原完整 14 项目标在 **189 个 encoder 参数张量**上的比较，并不是在 203 个全部训练张量上的直接图比较。依据：`tools/train_msvr_role_set.py:166-183`、`:227-241`，`{refs['direct']}`；例如 `snapshots/project/evidence/msvr310_role_set_m0_complete_20260908/m0/fold_0_control/memory_steps.jsonl:4`，其余五端见 direct_recorded_norm_checks 列表。

独立算术核对全部 248 步的梯度范数关系、余弦恒等式、所选历史组、上游范数长度、应用前后统计和直接比较比率。可核验的记录计数是：fresh role 重编码 **6272**，历史 VJP **5760**，直接检查 **384** 个 record-forwards。它们是按原组记录重算的计数，不是本审计重新执行的模型前向。依据：`independent_arithmetic.py:1-254`，`local_report_checks.json:10-18`。

限制必须保留：

- 没有保存真实参数梯度向量、每步模型状态、原 RNG 或 frozen-field 缓存，所以范数/计数一致不能替代真实模型 Jacobian 的独立复现。
- 203/203 指原运行的累计非零梯度覆盖。名字域/终点权重可核对，逐步全部非零既没有被证明，也不是报告中的主张。
- 首历史组之外，原代码检查 fused VJP 重编码相等、RNG/buffer 不变和梯度累加；没有全历史组直接完整图或三角色图输出的独立证明。
- 六次严格模型重载和五种检索输出逐位相同是原 runner 在每端八条来源记录上的运行断言。原始 before/after 输出数组没有保存；本审计重建的是状态映射及 SHA，没有重跑模型输出。
- 初始角色权重与两个 overfit 最终权重没有独立保存文件；相应初始/最终 hash 和冻结状态仍为运行见证。

这些边界在当前登记计划/结果中已有明确限定，本审计没有发现需要补写成失败的已登记全程直接图要求。旧梯度探针没有历史全角色图比较的历史限制也不能被本轮首组检查追溯改写。依据：`refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:29-33`，`results/MSVR310_ROLE_SET_V1_M0_2026-09-08.md:9-12`。

## E. 完整长度、配对、公平性和成本 — PASS

| 端点 | 更新 | 唯一来源记录 | 距离元素 | 历史候选曝光 | 最大历史数 / 年龄 | VJP record-forwards |
|---|---:|---:|---:|---:|---:|---:|
{table}

总计 **248 更新、4,945,920 距离元素、738 条跨端去重来源记录**；每步仍为 64 个当前样本位、8 身份×8 视图。全部四对序列（含 overfit）逐步记录索引和三模态增强像素 SHA 相等。重复曝光不作为独立样本。像素 tensor 没有保存，因此只确认日志 SHA 配对，未重新生成增强图像。依据：`{refs['ar']}`，`tools/train_msvr_role_set.py:109-115`、`:356-375`。

容量端的实际最大历史候选数为 221、最大年龄 5；登记容量为 512、最大年龄 8。另 780 个完整 source batch 的独立无模型 replay 验证年龄 8、顺序、当前 record 排除与过期规则；这不等于 GPU 容量端运行过年龄 8 或 512 个历史候选。两个 100-step overfit 固定 batch 只有 53 个唯一 record，因当前 record 的历史副本被排除，历史候选和历史 VJP 均为 0。其通过仅证明固定来源 batch 的优化能力，不证明长历史反传过拟合。依据：`{refs['folds']}`，`{refs['overfit']}`，`tools/msvr_instance_memory.py:15-37`。

两端共用新鲜坐标、完整历史反传、V8 架构、AdamW、LR0.00035、weight decay0.0001、AMP 初始 scale256、既有其他损失和配对来源；没有新训练模块。角色提议代码即使在 control 端也实际执行；改变的是所选 fused 关系目标及其候选上游。运行的累计非零梯度/AMP/frozen assertions 均完成；容量端峰值 allocated **11266.8193359375 MiB**、峰值 reserved **12116 MiB**，未越登记 24 GiB。依据：`tools/train_msvr_role_set.py:71-105`、`:125-160`、`:242-283`，`independent_arithmetic.json` 的 `endpoint_checks`。

8 个训练循环记录耗时合计 **459.3965334482 秒**；整个 M0 阶段用时 **614.7843318302 秒**，包含构建/预检/序列化等开销，不能把两者混写。封闭的 M0/T0/CPU 共 **42 文件、222,116,266 字节**（不含正在变化的 pipeline/Q1）。启动盘空闲 **11,129,188,352 字节**，启动 GPU 占用收据为 **1 MiB**。本轮未测量新的 GPU 性能、包裹整个 Q1 的盘占用或未来运行时长。依据：`binding_and_launch_checks.json`，`local_report_checks.json:14-20`。

科学门仍是完整三折两端、固定 20 epoch、1560 更新后两组既有五项全部成立；单种子身份 bootstrap 不是多训练种子。M0 没有新的 600-query 检索数组或科学成绩，原 CPU 收据的 retrieval element count=0。本报告不推断 Q1 结局。依据：`refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:19-41`，`tools/verify_msvr_role_set.py:250-307`。

## F. 证据类型与声明影响 — PASS

| 证据 | 类型 | 能支持的声明 |
|---|---|---|
| 数据集文件名 identity/scene、来源训练监督 | `real_gt` | 标签/隔离/监督来源正确；不自动成为检索成绩 |
| Signal 相等、重载相等、直接图/VJP 相比较 | `synthetic_proxy` 工程自比较 | 有界实现一致性，不能称作任务 GT 或方法优越性 |
| 随机矩阵、类 0、tie、合成链式法则 | `simulation_only` | 数学实现检查，没有真实模型或一般化结论 |
| 保存距离与标量重算、hash、标签队列重建 | 确定性审计证据 | 精确说明重算域；不替代模型级原运行见证 |

声明影响逐条见 `EXPERIMENT_AUDIT.json:claims`。完整 M0、标量/距离/终点状态均支持；203 非零覆盖、初始权重一致、原模型逐位输出和直接图数值关系必须保留“原运行见证”限定。全 248 步真实模型梯度独立复现、角色多样性单因果归因、Q1 资格与官方/SOTA 等声明均未建立。

## 审计过程、失败尝试与复现边界

所有独立脚本、输出、快照和本报告只写入本审计目录。未训练、未构造模型、未做模型/GPU forward/backward、未安装包、未改进程或科学文件、未读官方图片、未读 Q1 结果。远端 CPU 设置最多 2 个 compute threads；checkpoint 只在 CPU 用 `weights_only=True` 加载。CPU 合成 tensor autograd 不含模型对象。

保留了三项审计自身的问题及修复记录：私有 SSH helper 的 stdout.reconfigure 与 StringIO 不相容；初版 inventory 在无 signal binding 的配置上误取 `signal_source`；文本快照的 CRLF 归一导致本地快照字节不符。前两项分别在连接准备/清单收集阶段停止，第三项是审计快照序列化问题。对应 `attempt_01_transport_failure.txt`、`attempt_02_remote_inventory.*`、`attempt_03_snapshot_failure.txt`。22 个受换行影响的文本随后经只读二进制 SFTP 收齐，原规范化副本保留在 `attempt_03_normalized_snapshots/`；最后快照字节差异为 0。没有将不可用检查记为 PASS。

`README.md` 给出脚本/命令顺序，所有成功远端执行命令与脚本 SHA 在 `*.command.json`。凭据及 SSH helper 原文没有复制到审计目录。结论无需修改在运行代码、重启或追加实验；下游保持本报告的证据范围，并在原完整 Q1 终态另行审计。
'''
report += '\n最后的交付核验另发现报告生成器先对 LF 字符串计算 verdict_id，而 Windows 实际写入 CRLF，以及四处报告行范围超出文件末尾/一处通配引用不能直接定位。均已在审计报告层修正，原失败保存在 attempt_04_report_packaging_failure.txt 和 attempt_04_reference_validation.json；最终使用实际文件字节 SHA，并核验所有明确引用存在、范围有效。科学输入和核验结论未变化。\n'
(OUT/'EXPERIMENT_AUDIT.md').write_text(report,encoding='utf-8')
audit['verdict_id']='sha256:'+hashlib.sha256((OUT/'EXPERIMENT_AUDIT.md').read_bytes()).hexdigest()
(OUT/'EXPERIMENT_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'evidence_references.json').write_text(json.dumps(refs,indent=2),encoding='utf-8')

response=f'''独立只读审计已完成。**总体 WARN；工程 PASS；确定性核验 PASS；Q1 科学资格未评价。** 审计采用新上下文原生 Codex reviewer，`review_independence: same-family`、`acceptance_status: provisional`。未发现 M0 数字错报、虚构文件、目标来自模型输出或需要修改科学代码的问题。

- **来源与完整执行匹配：** 137 条递归 hash 绑定、29 份原始文本（5,717,997 字节）、三个来源 Signal checkpoint 的身份绑定、780 个独立重建来源 batch、完整 248 步均通过。M0 覆盖 738 条跨端去重来源记录；重复视图仍按原采样定义保留。
- **原始数组和终点重建通过：** 独立复算 4,945,920 个距离元素、全部关系提议/去重/mask/queue/损失账本；最大目标误差 3.2410e-08、最大总损失账本误差 5.7121e-07。六个终点完整/冻结/Signal 状态 SHA 重建匹配；原 CPU 收据 37 文件再次散列匹配。
- **过拟合算术通过：** 加权 CE floor 为 0.5857136327437849，control/role_set 校正末首比为 0.0007014160404002304 / 0.0007016520038614711，均低于 0.1。两个固定 batch 各只有 53 条唯一记录，历史候选为 0，因此这部分不能证明历史 VJP 的过拟合能力。
- **WARN 的具体范围：** 203/203 是原运行累计非零梯度见证；直接完整图比较仅覆盖六个容量端各第 4 步首个历史组、189 个 encoder 张量，记录最大相对 L2 误差 1.6810499980311218e-05。独立核对了范数/组别/比率算术，但原梯度向量、缓存/RNG、初始角色状态和重载前后输出数组没有保存，不能声称全程模型级独立重建。M0 实际历史最大 221 个候选、年龄 5；年龄 8 来自无模型的完整 T0 元数据重放。
- **范围和成本明确：** M0 阶段 614.7843 秒；峰值 allocated 11266.8193 MiB、reserved 12116 MiB；封闭 M0/T0/CPU 产物 222,116,266 字节。角色负关系覆盖与最难 hinge 降权同时改变，不能单因果归于角色多样性。

所有审计产物只写入 [审计目录](/C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908)。主要文件：

- [完整审计报告](/C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/EXPERIMENT_AUDIT.md)
- [机器可读审计结论及完整输入 SHA](/C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/EXPERIMENT_AUDIT.json)
- [248 步独立数组核验](/C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/independent_arithmetic.json)
- [独立 checkpoint 重建核验](/C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/independent_checkpoints.json)
- [命令、脚本与失败尝试说明](/C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/README.md)
- [本回复原文](/C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/reviewer_full_response.md)

本审计未训练、未运行模型/GPU前后向、未改科学文件或进程、未安装包、未读官方图片或部分 Q1 分数；CPU 算术限制为 2 线程。原 wrapper/Q1 在只读观察时存活，未作任何干预。本轮审计不要求补跑、重启或改代码，后续结论继续等待原完整 Q1 终态。'''
response=response.replace('](/C:/','](C:/')
(OUT/'reviewer_full_response.md').write_text(response,encoding='utf-8')
trace=OUT/'trace';trace.mkdir(exist_ok=True)
meta=dict(skill='experiment-audit',run_id='role_set_m0_independent_audit_20260908',generated_at=now,executor='codex',reviewer_model='gpt-6-astra',reviewer_reasoning='max',reviewer_family='openai',review_independence='same-family',acceptance_status='provisional',project_dir=str(ROOT),agent_id='/root/audit_msvr_role_set_m0',status='complete')
(trace/'run.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
(trace/'001-independent-audit.request.json').write_text(json.dumps(dict(call_number=1,purpose='fresh-independent-M0-review',tool='spawn_agent',model='gpt-6-astra',reasoning_effort='max',prompt=REQUEST.read_text(encoding='utf-8'),request_sha256=hashlib.sha256(REQUEST.read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
(trace/'001-independent-audit.response.md').write_text(response,encoding='utf-8')
(trace/'001-independent-audit.meta.json').write_text(json.dumps(meta|{'verdict':'WARN','engineering_status':'PASS','deterministic_verification':'PASS','verdict_id':audit['verdict_id']},indent=2),encoding='utf-8')
readme='''# Independent M0 audit reproduction and artifacts

The scientific repository and remote run are read-only inputs. All local output paths are restricted to this directory. Credentials are obtained privately from the pre-existing helper; its contents are not included.

## Main outputs

- EXPERIMENT_AUDIT.md: complete independent reviewer analysis with exact source/receipt line evidence.
- EXPERIMENT_AUDIT.json: verdict, per-check statuses, claim impacts, boundaries and audited-input hashes.
- reviewer_full_response.md and trace/001-independent-audit.response.md: identical verbatim reviewer response.
- independent_arithmetic.json: all 248 saved-array/label/queue/loss step recomputations and 780 independently generated source batches.
- independent_checkpoints.json: three source checkpoint bindings, six terminal state reconstructions and 37 CPU receipt file hashes.
- supplementary_checks.json: label origin, exact tie derivatives, 12 synthetic mean-hinge/gradient cases and synthetic VJP.
- remote_inventory.json / local_inventory.json / snapshot_manifest.json: inventory and hashes.
- binding_and_launch_checks.json / local_report_checks.json: launch bytes, closed exits, logs, reports and resource accounting.

## Exact execution sequence

Local Python:
`C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe -X utf8`

Remote transport launcher:
`C:/Users/gb/AppData/Local/Programs/ClawX/resources/bin/uv.exe run --offline --with paramiko python -X utf8 remote_transport.py <script> <label>`

Run local inventory.py; run remote_transport.py with remote_inventory.py / remote_inventory; run local unpack_remote.py. Then execute independent_arithmetic.py / independent_arithmetic, independent_checkpoints.py / independent_checkpoints, supplementary_checks.py / supplementary_checks and binding_and_launch_checks.py / binding_and_launch_checks via remote_transport.py. Remote commands and stdin script SHA are preserved separately in each *.command.json. These scripts perform only CPU metadata, array arithmetic and safe checkpoint tensor deserialization, with CUDA hidden and two threads.

Run local finalize_checks.py. If reproducing the original newline issue, run fetch_raw_text_bytes.py through the offline Paramiko launcher; it compares the existing snapshot to the raw hash manifest and retrieves only mismatching source text bytes. Then run finalize_checks.py again. Finally run write_reports.py. See the original failed attempts below; the final audit did not accept mismatching snapshots.

## Preserved attempted-check failures

1. attempt_01_transport_failure.txt: the private helper called stdout.reconfigure, while the initial audit transport redirected to StringIO. Replaced the private output sink with a real null TextIOWrapper; no credentials emitted.
2. attempt_02_remote_inventory.command.json / stderr.txt / stdout.txt: initial optional mapping loop accessed signal_source on a configuration without that binding. Corrected the inventory loop; no scientific source changed.
3. attempt_03_snapshot_failure.txt and attempt_03_normalized_snapshots/: pathlib read_text normalized 22 remote CRLF text files while serializing snapshots. Their original remote byte hashes were already correct. raw_text_snapshot_recovery.json records binary SFTP receipt, full SHA/length matches and exact byte snapshots. Final remote_snapshot_byte_mismatches is empty.

Some shell rg searches returned exit 1 for no matches; these were discovery no-hit results, not experiment failures. No blocked check was converted into PASS.

## Scope limits

The scripts do not reconstruct any real-model forward/backward; no module instance is created. GPU gradients, original RNG/cache/augmented images, strict reload output arrays and initial role state are not saved. Existing assertions remain runtime witnesses. Official image content, partial Q1 output and ongoing training are deliberately outside this audit. Current package versions are observations, not retroactive proof of the original backend.
'''
readme += '\nFinal packaging attempt: attempt_04_report_packaging_failure.txt records the LF-string versus written-CRLF verdict hash mismatch. attempt_04_reference_validation.json preserves draft line-range problems. The writer now hashes actual written report bytes, and the final artifact_validation.json checks all source reference ranges, script ASTs and verbatim response/trace identity. These are audit packaging corrections only.\n'
(OUT/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps(dict(verdict='WARN',engineering_status='PASS',deterministic_verification='PASS',audited_input_hashes=len(hashes),report_bytes=(OUT/'EXPERIMENT_AUDIT.md').stat().st_size,response_bytes=(OUT/'reviewer_full_response.md').stat().st_size,verdict_id=audit['verdict_id']),indent=2))
