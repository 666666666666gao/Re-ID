"""Write source-only independent review artifacts; never import experiment code."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
OUT = Path(__file__).resolve().parent
ENTRIES = [
    'tools/run_native_research.py',
    'tools/queue_native_research.py',
    'tools/check_native_research_pair.py',
    'tools/report_native_research.py',
    'refine-logs/native_research_v6/EXPERIMENT_PLAN.md',
]
INHERITED = [
    'tools/run_native_partitioned.py',
    'tools/run_independent_native_evidence.py',
    'tools/run_foundation_recipe.py',
    'tools/queue_foundation_recipe.py',
    'tools/queue_native_partitioned.py',
    'tools/check_partitioned_native_pair.py',
    'tools/report_native_partitioned.py',
    'tools/run_clean_clip_joint.py',
    'tools/run_correspondence_roles.py',
    'tools/run_training_feature_scale.py',
    'tools/analyze_correspondence_distances.py',
    'modeling/trifusion/evidence_author_heads.py',
    'modeling/trifusion/partitioned_evidence_clip.py',
    'modeling/trifusion/independent_native_roles.py',
]
for path in ENTRIES + INHERITED:
    if path.endswith('.py'):
        ast.parse((ROOT / path).read_text(encoding='utf-8'), filename=path)
hashes = {path:hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in ENTRIES}
mock = json.loads((OUT / 'MOCK_RESULT.json').read_text())
assert mock['status'] == 'PASS' and mock['model_runtime'] is False
checks = [
    {
        'id':'cli_namespace_and_receipt', 'status':'PASS',
        'evidence':['tools/run_native_research.py:22','tools/run_native_partitioned.py:54',
                    'tools/run_independent_native_evidence.py:167','tools/run_independent_native_evidence.py:187'],
        'finding':'V6 configure replaces the partitioned schema/build_core; partitioned configure binds foundation.train to the original partitioned wrapper. The independent entry main then rebinds build/loss/optimizer globals and invokes its own train wrapper. Thus CLI train is independent.train -> partitioned.train -> original foundation.train, then partition memory and production_m0_diagnostics are written once. V5 and final V6 each produced exactly one diagnostics result call in the exact-function AST mock; there is no recursion or omitted receipt.'
    },
    {
        'id':'fresh_initialization_and_author_batch', 'status':'PASS_SOURCE',
        'evidence':['tools/run_clean_clip_joint.py:37','tools/run_independent_native_evidence.py:36',
                    'tools/run_foundation_recipe.py:103','tools/run_foundation_recipe.py:189',
                    'tools/check_partitioned_native_pair.py:26'],
        'finding':'Every prepare/M0/train/evaluate is a separate process and uses the fresh public-CLIP/fresh-camera/head constructor with seed42 and strict initialization binding. M0 weights are never loaded into formal training. Author B/K configuration and augmentation are inherited unchanged; the first actual full batch, config, common state, shared output, and semantic/native zero-exit outputs are checked. Formal loops assert each batch size; all variants retain the same two-device placement, full BN and full triplet batch.'
    },
    {
        'id':'updates_optimizer_bn_and_full_state', 'status':'PASS_SOURCE',
        'evidence':['modeling/trifusion/evidence_author_heads.py:42',
                    'modeling/trifusion/evidence_author_heads.py:63',
                    'tools/run_independent_native_evidence.py:100','tools/run_independent_native_evidence.py:126',
                    'tools/run_foundation_recipe.py:229','tools/run_foundation_recipe.py:263'],
        'finding':'Each run constructs a fresh author optimizer, scheduler and AMP scaler at 256. Author optimizer membership is exactly the trainable parameter set. Loss and unscaled gradients must be finite; every M0 trainable tensor must have cumulative nonzero gradients; visual and camera state must change. The real optimizer post-step hook requires eight effective updates, native14-tensor activity/change and author BN num_batches_tracked=8. Full model state_dict includes BN buffers and is strictly loaded into an independently constructed model before the original 1e-5 output check. The checkpoint intentionally contains model state, not optimizer/scaler resume state; this is fresh training and inference reload, not a training-resume test.'
    },
    {
        'id':'scoring_and_best_reload', 'status':'PASS_SOURCE',
        'evidence':['tools/run_foundation_recipe.py:248','tools/run_foundation_recipe.py:287',
                    'tools/run_correspondence_roles.py:79','tools/queue_foundation_recipe.py:84'],
        'finding':'The loop trains epochs1..50, overwrites one best_map.pth only on nondecreasing mAP, and selects the latest epoch for exact mAP ties. All reported CMC values follow that same model. A fresh evaluate process verifies the selected epoch/metrics, strict full-model load and complete query/gallery author-versus-independent GT scoring with original camera/scene filtering and 1e-5 tolerance. These are source invariants, not achieved runtime results.'
    },
    {
        'id':'per_endpoint_queue_resource_and_failure', 'status':'PASS_SOURCE_AND_STDLIB_MOCK',
        'evidence':['tools/queue_native_research.py:49','tools/queue_native_research.py:62',
                    'tools/queue_native_research.py:97','MOCK_RESULT.json'],
        'finding':'Exact-function AST mock executed the intended39-command sequence: for201,MSVR,100, prepare all3 variants and paired forward, then global_only M0/train/evaluate, semantic M0/train/evaluate, native M0/train/evaluate. No backward-repeatability command or global nine-M0 barrier remains. Each child launch explicitly exposes only physical0,1; only one child is active and the existing resource wait is240s. Simulated201/native M0 nonzero exit stopped after command11, preserved FAILED status/exit code and invoked no report. Successful completion invokes one CPU report. Actual run_logged zero/nonzero branches, active-command cleanup and waiting GPU readings were exercised with stubs.'
    },
    {
        'id':'terminal_report', 'status':'PASS_SOURCE_AND_STDLIB_MOCK',
        'evidence':['tools/report_native_partitioned.py:23','tools/report_native_partitioned.py:45',
                    'tools/report_native_research.py:15','MOCK_RESULT.json'],
        'finding':'The report still requires all18 M0/full jobs and all9 verified endpoints, checks identical full batch order and reuses the original diagnosis/scoring formulas. The V6 wrapper now appends every query identity/AP/first-match rank using the same camera/scene scorer; mock data confirm both routing branches and control/candidate row alignment. Existing paired harm/repair counts, identity distributions, curves, best-to-last change, parameter placement and runtime receipts are retained. No new scientific module or claimed result was added.'
    },
]
review = {
    'schema':'native_research_v6_independent_code_review',
    'reviewed_at':datetime.now().astimezone().isoformat(),
    'verdict':'PASS_SOURCE_AND_STDLIB_MOCK',
    'blocking_issues':[],
    'nonblocking_issues':[],
    'reviewer_model':'gpt-6-astra',
    'reasoning_effort':'max',
    'review_independence':'same-family',
    'acceptance_status':'provisional',
    'fresh_context':True,
    'repository':str(ROOT),
    'source_sha256':hashes,
    'reviewed_inherited_sources':INHERITED,
    'authorization':'User selected option1 (1先修复吧): per-endpoint research training with finite actual updates/full model save-reload/GT scoring retained; extra backward-repeatability repair is not a prerequisite. Historical V1-V5 FAIL/STOP results remain unchanged.',
    'checks':checks,
    'resolved_during_review':[
        'Original draft incorrectly said the V5 CLI did not write production_m0_diagnostics and added a redundant train wrapper. Exact namespace inspection showed the original independent entry already writes it. Executor removed the V6 train wrapper and corrected plan line12; final mock confirms one diagnostics write for both V5 and V6.',
        'Inherited terminal compare discarded per-query AP/rank arrays despite the plan promise. Executor added report-only query_changes using the existing verified scorer; final source/mocks pass without changing scoring or training.'
    ],
    'validation':{
        'ast_parse':'PASS for all listed current Python entry/inherited files',
        'stdlib_mock':'PASS',
        'mock_result':'MOCK_RESULT.json',
        'mock_source':'namespace_queue_mock.py',
        'harness_notes':'HARNESS_EXECUTION_NOTES.md',
        'success_command_count':39,
        'failure_stop_command_count':11,
        'actual_model_forwards':0,
        'actual_optimizer_updates':0,
        'remote_commands':0,
        'package_installs':0,
        'source_files_modified_by_reviewer':0,
        'harness_failures':'PATH Python E:/Scripts/python.exe failed before execution because pyvenv.cfg was missing. Existing offline-discovered Python ran the harness; its initial subprocess stub lacked STDOUT. Only the mock constant was fixed and a fresh fixture used. Both are review-harness failures, not model evidence; first fixture and notes retained.'
    },
    'limitations':[
        'This is local source review and stdlib control-flow mocking only. It is not a neural M0, GPU memory test, numerical repeatability pass, full50 result, efficacy result, or independent replay of remote weights.',
        'The existing309-file remote dependency precheck and final V6 SOURCE_SCOPE are executor preparation outside this reviewer run. This verdict does not claim that deployment bytes, live GPU availability, or storage capacity were remotely verified by the reviewer.',
        'Actual paired initialization and each endpoint M0 must pass before its own fresh50. RGBNT100 B128 capacity remains unproven until that real check; any real update/reload/scoring/OOM failure must stop and preserve its original trace.',
        'Full state here means full model parameters/buffers for inference reload; optimizer/scheduler/scaler continuation is neither implemented nor claimed.',
        'All results remain single-seed, officially selected research results; no SOTA, capacity-isolated mechanism, multi-seed stability or overall Goal completion follows from this review.'
    ],
    'deployment_scope':'Source gate supports the registered new V6 campaign on server2026 physicalGPU0/1 only, through its own paired-forward/per-endpoint-M0 gates. No authorization to restart or relabel historical V1-V5 failures.'
}
(OUT / 'REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines = [
    '# V6 逐端研究执行：独立部署前审查',
    '',
    '**结论：PASS_SOURCE_AND_STDLIB_MOCK；BLOCKING = 0。** 当前四个入口和继承调用链正确实现已选的逐端执行口径。此结论仅关闭源码审查门槛，不是实际 M0、容量、50轮训练或科学收益通过。',
    '',
    f"审查时间：{review['reviewed_at']}。实际模型：gpt-6-astra；reasoning_effort：max。fresh-context；review_independence：same-family；acceptance_status：provisional。",
    '',
    '## 关键核验',
    '',
    '1. **实际 CLI 调用链正确。** `run_native_research.configure` → partitioned.configure → independent.configure；`independent.main` 再配置后执行 `independent.train` → partitioned.train → 原 foundation.train。返回后写两卡内存和一次 production_m0_diagnostics。精确函数 AST mock 验证 V5/V6 各一次诊断写入，无递归、无漏写。',
    '2. **M0 与正式训练从同一登记初始化分别新建。** 独立进程、seed42、公开CLIP、新camera/head；fresh50不读M0权重。完整作者batch/增强/BN/Triplet及两卡放置继承不变。首次实际整batch输入、共有state、shared输出及semantic/native zero-exit前向仍须通过。',
    '3. **真实更新与状态检查保留。** AMP初始scale256、unscaled梯度/loss有限、8次有效optimizer step、全部可训练张量累计活动、视觉/camera变化、作者BN计数8和native14张量活动/变化均有检查。保存完整model state（含BN buffers），独立构造strict load及原1e-5输出比较。optimizer/scheduler/scaler各端新建；本任务不声称optimizer续训恢复。',
    '4. **完整50轮与统一best语义正确。** 同mAP取较晚epoch；只覆盖一份best_map.pth，所有CMC跟随同一权重。新evaluate进程严格重载，完整query/gallery按真实GT和原camera/scene过滤评分，与独立实现核对。',
    '5. **逐端顺序、资源与失败停止正确。** mock成功路径39条命令，201→MSVR→100，各集三初始化+paired forward，再global_only/semantic/native各M0→train→evaluate。无额外backward探针、无全九M0屏障；每次Popen固定CUDA_VISIBLE_DEVICES=0,1、单个双卡子任务，资源等待240秒。模拟201/native M0失败后第11条命令即停止，FAILED及退出码保留、报告0次。',
    '6. **终态报告交付补齐。** 原18-job/9端验证、batch顺序一致性、GT评分和配对诊断公式不变；新wrapper落盘完整query identity/AP/首位rank，camera及scene分支的mock输出正确。曲线、身份分布、修复/新增错误、best到末轮变化、参数/内存/耗时原记录保留。',
    '',
    '## 审查期间已修正',
    '',
    '- 初稿关于“V5未写M0诊断”的解释不符合实际CLI调用链。执行者已删去V6重复train包装并改正计划，最终版本仅复用原回执链。',
    '- 原compare虽计算逐query AP/rank，却未返回落盘。执行者已在V6报告层按原scorer补齐query_changes，未改模型、loss、优化器或评分公式。',
    '',
    '## 验证与边界',
    '',
    '`MOCK_RESULT.json` 为PASS；精确AST函数执行只用stdlib和假计算/假进程，实际模型forward=0、optimizer update=0、远端命令=0、包安装=0。审查者未修改项目源码。',
    '',
    '首次PATH Python因缺pyvenv.cfg未运行；首次有效Python harness因mock遗漏STDOUT失败。仅修正mock常量，使用新fixture后通过，原fixture及`HARNESS_EXECUTION_NOTES.md`保留。这两项不属于模型失败或模型证据。',
    '',
    '远端309文件预检及最终SOURCE_SCOPE由执行者准备，本审查未独立远端核验。实际GPU/磁盘、初始配对和每端M0仍须运行通过，尤其B128容量尚未证实。旧V1–V5 FAIL/STOP不改判；用户选择只取消额外反向重复性前置目标。',
    '',
    '## 最终审查输入 SHA256',
    '',
    '| 文件 | SHA256 |',
    '|---|---|',
]
lines.extend(f'| {path} | `{digest}` |' for path,digest in hashes.items())
(OUT / 'REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'verdict':review['verdict'],'blocking_issues':[],
    'review_json':str(OUT/'REVIEW.json'),'review_md':str(OUT/'REVIEW.md'),
    'source_sha256':hashes},indent=2))
