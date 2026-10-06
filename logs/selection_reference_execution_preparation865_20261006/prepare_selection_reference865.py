from pathlib import Path
from datetime import datetime
import ast,hashlib,json,shutil,subprocess

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp')
packet=base/'independent_evidence_draft/signal_selection_reference_v1'
proof=base/'foundation_recipe_v1_20261002'
assert not (proof/'publication865_local.json').exists()
previous=json.loads((proof/'four_copy864_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
retirement=json.loads((base/'independent_evidence_draft/selection_storage865_retirement_r2/RETIREMENT.json').read_bytes())
assert retirement['removed_count']==6
at=datetime.now().astimezone().isoformat()
archive=repo/'logs/selection_reference_execution_preparation865_20261006'
assert not archive.exists();archive.mkdir(parents=True)
prior=archive/'previous_draft864';prior.mkdir()
for name in ('run_signal_selection_reference.py','LOCAL_SOURCE_REVIEW.md','EXPERIMENT_PLAN.md','MANIFEST.json'):
 path=('tools/' if name.endswith('.py') else 'refine-logs/signal_selection_reference_v1/')+name
 (prior/name).write_bytes(subprocess.check_output(['git','show',previous['head']+':'+path],cwd=repo))
for name in ('selection_storage865','selection_storage865_v2','selection_storage865_v3','selection_storage865_retirement','selection_storage865_retirement_r2'):
 shutil.copytree(base/'independent_evidence_draft'/name,archive/name)
shutil.copytree(base/'selection_reference_publication864_precompile_failure',archive/'publication864_precompile_failure')
for name in ('qualify_selection_storage865.py','qualify_selection_storage865_v2.py','qualify_selection_storage865_v3.py','retire_selection_storage865.py','retire_selection_storage865_r2.py','prepare_selection_retirement865_r2.py','prepare_selection_reference865.py'):
 (archive/name).write_bytes((base/name).read_bytes())
plan=packet/'EXPERIMENT_PLAN.md'
plan.write_text(plan.read_text(encoding='utf-8')+f'''\n\n## §41.865 execution revision — {at}\n\nQueue and the sole all-query CPU report now implemented. Each dataset prepares three fresh-process initializers, verifies both SIM initial states/capacity and the common plain foundation state, then runs each actual8M0 → fresh50 → first strict evaluation. Retire only that M0 probe after accepted full receipt; all nine formal bests stay until consumers close. Six explicitly qualified historical own weights retired, freeing {retirement['removed_bytes']}B; actual free {retirement['free_after']}B at retirement. Startup budget unchanged5,922,357,248B and each stage requires2GiB.\n\nThe entry now logs M0 per-tensor actual parameter changes, and requires all12 SIM interaction tensors plus var classifier/BN weight to change, in addition to original finite/live/reload gates. Original CPU component8-step evidence remains component-only. Common initializer and BN/optimizer/runtime gates have not run. No source modification after launch. Report requires9/450epochs and all9 comparisons, exact labels/camera/view/basename batch metadata equality; full official arrays, same-best CMC, query repairs/new errors and identity changes, fixed-model bootstrap only.\n\nOriginal storage qualificationV2 stopped before mutation because its generic loop incorrectly forbade a protected retained winner as well as targets. V3 narrows the protection exclusion to target weights; all winner SHA preservation remains required. No old producer or experiment was rerun. This source revision is root self-checked, not independently reviewed or runtime accepted.\n''',encoding='utf-8')
(packet/'EXPERIMENT_TRACKER.md').write_text(f'''# Signal source-reference execution tracker\n\n{at}: SOURCE_READY_ONLY. Queue/report implemented; source self-check only. New9 prepare/M0/full NOTstarted. Globalonly1536 vs SIM3072 explicit; primary masked−all_patch matchedcapacity/heads. HistoricalF1 on25 remains historicalonly, no25probe. Warmconda unchanged.\n\nStartup disk qualification is live at launch, not inferred from cleanup. Only26physical0/1; no power/temp queries/actions; one NN. ETA9.5–12h estimate pending actual production step timing. All old failures preserved; Goal ACTIVE/UNMET.\n''',encoding='utf-8')
(packet/'LOCAL_SOURCE_REVIEW.md').write_text(f'''# Root source self-check — {at}\n\nSOURCE_READY_ONLY, not independent review or production acceptance.\n\nSource trace: pinned Signal forward returns global and SIM raw heads; training var1536 Triplet counted once, inference3072 concatenation. All_patch bypasses only TokenSelection forward; both arms retain the same frozen6 selector tensors and12 trainable interaction tensors plus var heads. Globalonly has no SIM/roles/adapters. CPU-private SIM initialization restores CPU RNG; full baseline/public/camera/common state equality checked per dataset before training. Original author losses/AMP/batch/schedule/evaluator are delegated unchanged.\n\nFresh processes avoid YACS cross-dataset carryover. Optimizer covers every trainable parameter exactly once. Both BN/global and SIM var counts must be8 in actual M0; all SIM12 plus var classifier/BN weight must actually change. Original every-trainable finite/nonzero-gradient, unchanged frozen state and strict-reload gates remain. M0 and fresh50 separate processes. Each full first strict success retires only its own M0 probe. Full nine formal bests retained until all consumers close.\n\nQueue fresh endpoints18jobs and9initializers; status/actual command/pid/startticks/time/exit saved. No replay/retry/rescue path. Sole CPU report9rows/450epochs/9pairs rechecks SHA/full50/best/distance/metadata and all-query scorer parity. It reports pair query repairs and identity distribution; no training seed inference from bootstrap. No power/temp or2025 calls.\n\nStartup pre-registered9best+oneprobe at360MiB plus2GiB budget5,922,357,248B; per-stage2GiB. Six target binaries qualified separately by exact complete50/strict/SHA/all4dominance/no dependency and retired; source366 and protected187 unchanged. Source scope adds only new3tools and this plan/tracker/review.\n\nRemaining: actual prepare/commonstate, M0/AMP/memory/activity/BN/heads/reload and fresh50/first strict/full report. CPU component and AST do not prove these. Zero original novelty or current SOTA claim.\n''',encoding='utf-8')
target=repo/'refine-logs/signal_selection_reference_v1'
newtools=['run_signal_selection_reference.py','queue_signal_selection_reference.py','report_signal_selection_reference.py']
for name in newtools:
 source=(packet/name).read_text(encoding='utf-8');ast.parse(source)
 (repo/'tools'/name).write_bytes((packet/name).read_bytes())
for name in ('EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','LOCAL_SOURCE_REVIEW.md'):
 (target/name).write_bytes((packet/name).read_bytes())
frozen=json.loads((repo/'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json').read_bytes())['source_sha256']
sources=dict(frozen)
for name in [*['tools/'+n for n in newtools],*['refine-logs/signal_selection_reference_v1/'+n for n in ('EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','LOCAL_SOURCE_REVIEW.md')]]:
 sources[name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
(target/'SOURCE_SCOPE.json').write_text(json.dumps(dict(status='FROZEN_EXECUTION_SOURCES',original_source_count=366,new_source_count=len(sources)-366,source_sha256=sources),indent=2)+'\n')
manifest_files={n:dict(bytes=(repo/n).stat().st_size,sha256=hashlib.sha256((repo/n).read_bytes()).hexdigest()) for n in sources if n not in frozen}
(target/'MANIFEST.json').write_text(json.dumps(dict(at=at,status='SOURCE_READY_ONLY',files=manifest_files,source_scope_sha256=hashlib.sha256((target/'SOURCE_SCOPE.json').read_bytes()).hexdigest(),prepare=False,m0=False,formal=False),indent=2)+'\n')
(packet/'SOURCE_CHECK865.json').write_text(json.dumps(dict(at=at,status='ROOT_SOURCE_AND_AST_ONLY',source_count=len(sources),production_prepare=False,m0=False,formal=False,queue_report_implemented=True),indent=2)+'\n')
(archive/'SOURCE_CHECK865.json').write_bytes((packet/'SOURCE_CHECK865.json').read_bytes())
doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
path=repo/doc
text=path.read_text(encoding='utf-8')
text+=f'''\n\n## §41.865 — 来源选择参考九端执行入口完成；先退役已闭合冗余权重\n\n记录时间：{at}。上一Goal轮实际完成§864源码草稿与CPU组件检查；本轮属于PROGRESS，不是已有九端结果。当前无活动旧NN/CPU报告；仍仅26物理GPU0/1，warm conda不重建，不查询或设置功率/温度，不访问2025或其他项目。统一Goal ACTIVE/UNMET。\n\n独立研究边界维持：global_only/masked/all_patch×RGBNT201、MSVR310、RGBNT100，共九个fresh50。SIM两臂同初始化、同活动容量、同var头、同3072维，只改变hard union mask；无SIM globalonly1536比较还含容量/额外head/联合视觉目标/维度差异，不能归为选择专属增益，更不计作我们的创新。public CLIP、新camera/head、作者RAW分头CE/softmargin、Adam/原AMP、B/K、增强和50轮日程不改。旧F1实存路径在25，仅历史参照，不重定位距离。\n\n源码队列与唯一CPU全query报告已补齐并根代理自查/AST通过，**尚未执行真实prepare/M0/full**。每数据集三份fresh initializer先核SIM两臂state和common plain state，逐端实际8M0→fresh50→首次strict，接受后退役该M0 probe；新九份best保留到消费者全部结束。M0保留原有限/全trainable非零梯度/冻结状态/严格重载门，另记录全部活动参数实变并要求SIM12 interaction及var classifier/BN权重改变，BN计数8。新3工具与plan/tracker/review构成扩展source{len(sources)}；旧366不改。根代理自查不是独立审计，也不等于真实模型通过。\n\n终态CPU报告只调用一次，要求9端/450epoch/全部9配对；主要masked−all_patch，另报告两臂−新匹配global。完整距离/过滤/同best CMC、batch labels/camera/view/RGB basename、query修复/新增错误、身份分布/固定模型bootstrap与实测训练成本均保留。metadata相等不证明增强图像字节相等；bootstrap不替代训练多种子。不得按首个分数换topk/LR/gain/margin/seed或叠N2/N3。\n\n存储启动预算仍5,922,357,248B，各阶段2GiB，须启动实时再核。旧cleanup后共享磁盘实际free曾降至4,625,711,104B。六份已完成50轮和首次strict的本项目clean_joint车辆/EV1semantic权重，经实际binary/distance/receipt SHA核验，四项指标均被保留winner覆盖，且不在RAW187或当前任务依赖中，单独退役{retirement['removed_bytes']}B，retirement时free{retirement['free_after']}B。完整日志/距离/回执保留，三个winner及旧source366/RAW187不变。qualificationV2错误把保留winner也要求不在protected中，断言后未删除；V3仅限制targets，原失败保留。新权重预算未降低。证据在`logs/selection_reference_execution_preparation865_20261006/`。\n\n此时仍无新检索分数，ETA9.5–12小时只是执行估计；下一动作是同步已登记源码后一次启动队列、真实初始化/M0验证，按180–300秒或预计里程碑观察。2025镜像I/O仍pending，不假称两服当前字节一致。\n'''
start=text.index('**当前进度：');end=text.index('\n\n',start)
text=text[:start]+'**当前进度：§41.865（2026-10-06更新）。** 原row六端与fixed-best全部闭合且无稳定进步，旧失败不改判。作者SIM masked/all_patch加26新global-only九端对照的执行队列与唯一CPU全query报告已实现、自查并登记；尚未真实prepare/M0/full。六份无当前依赖且被保留winner四指标覆盖的旧best已核SHA退役，完整日志/距离/回执和来源依赖保留。启动预算5,922,357,248B不变，须实时核查。当前无活动NN/observer；仅26 GPU0/1、无功率温度动作，25原I/O pending不探测。Goal ACTIVE/UNMET。以下旧首页细节为历史摘要，现态以本段及正文后段为准。'+text[end:]
path.write_text(text,encoding='utf-8')
desktop=Path('C:/Users/gb/Desktop/document')/path.name;desktop.write_bytes(path.read_bytes())
current=repo/'refine-logs/CURRENT_GOAL.md'
current.write_text(current.read_text(encoding='utf-8')+f'''\n\n§41.865：九端SOURCE_READY_ONLY，queue/report/扩展source{len(sources)}已实现，尚未真实prepare/M0/full。六份明确dominated已闭合own权重退役{retirement['removed_bytes']}B，旧366/RAW187/winners不改。启动预算仍5,922,357,248B，须实时核查。真实M0通过后各端fresh50，不再额外parity修复/尺度搜索。根自查非独立复现。Goal ACTIVE/UNMET。\n''',encoding='utf-8')
changed=set([doc,'refine-logs/CURRENT_GOAL.md',*['tools/'+n for n in newtools],*['refine-logs/signal_selection_reference_v1/'+n for n in ('EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','LOCAL_SOURCE_REVIEW.md','SOURCE_SCOPE.json','MANIFEST.json')]])
changed.update(str(p.relative_to(repo)).replace('\\','/') for p in archive.rglob('*') if p.is_file())
protected=json.loads((proof/'publication864_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo/n).read_bytes()).hexdigest()==d for n,d in protected.items())
record=dict(previous_head=previous['head'],files=sorted(changed),protected_files=protected,immutable_archive_roots=[str(archive.relative_to(repo)).replace('\\','/')],remote_sparse_roots=['refine-logs/signal_selection_reference_v1',str(archive.relative_to(repo)).replace('\\','/')])
(proof/'publication865_local.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status='LOCAL_REGISTERED_NOT_PUSHED',files=len(changed),source_count=len(sources),doc_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
