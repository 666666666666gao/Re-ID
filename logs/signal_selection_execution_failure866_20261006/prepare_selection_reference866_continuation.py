from pathlib import Path
import hashlib,json,subprocess
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');base=Path('C:/Users/gb/.codex_tmp')
proof=base/'foundation_recipe_v1_20261002';packet=base/'independent_evidence_draft/signal_selection_reference_v1'
previous=json.loads((proof/'four_copy865_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
assert not (proof/'publication866_local.json').exists()
target=repo/'refine-logs/signal_selection_reference_v1'
archive=repo/'logs/signal_selection_execution_failure866_20261006'
at=json.loads((archive/'MINIMAL_CHANGE_CHECK.json').read_bytes())['at']
sources=json.loads((target/'SOURCE_SCOPE.json').read_bytes())['source_sha256']
frozen=json.loads((repo/'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json').read_bytes())['source_sha256']
canonical=base/'rt_complete862/received'
assert len(frozen)==366 and all(hashlib.sha256((canonical/n).read_bytes()).hexdigest()==d for n,d in frozen.items())
for name in set(sources)-set(frozen):sources[name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
(target/'SOURCE_SCOPE.json').write_text(json.dumps(dict(status='FROZEN_EXECUTION_SOURCES',original_source_count=366,new_source_count=6,source_sha256=sources),indent=2)+'\n')
(target/'MANIFEST.json').write_text(json.dumps(dict(at=at,status='STATS_ONLY_REVISION_SOURCE_READY',files={n:dict(bytes=(repo/n).stat().st_size,sha256=sources[n]) for n in sources if n not in frozen},source_scope_sha256=hashlib.sha256((target/'SOURCE_SCOPE.json').read_bytes()).hexdigest(),original865_terminal=1,original_prepare=3,m0_updates=0,formal=False),indent=2)+'\n')
doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';path=repo/doc;text=path.read_text(encoding='utf-8')
text+=f'''\n\n## §41.866 — 首次九端执行失败定位；仅移动显存统计重置\n\n{at}：原865 supervisor3862123/queue3862125已实际EXIT1于11:07:40.242523；首180秒观察确认两者不存在。不是初始化失败：RGBNT201 global_only/masked/all_patch三份真实prepare全部EXIT0，同SIM状态与common plain状态配对核验通过。首global M0进程3865331在调用`torch.cuda.reset_peak_memory_stats(1)`时，设备CUDA allocator尚未初始化，明确抛`Invalid device argument: did you call init?`，未进入模型构造/optimizer/训练目录，M0更新0、formal0。原19份文本和旧入口接收SHA核对并保留在`logs/signal_selection_execution_failure866_20261006/`。首次日志collector误读failed_preparation键，在本地KeyError、SSH未连；正确读取实际failed_job后只接收原日志，没有重跑。\n\n唯一源修订：GPU1 peak reset从train入口移到optimization，在模型已分段放置GPU1之后执行。删除memory-reset语句后的新旧AST完全一致，模型/头/损失/topk/原AMP/采样/日程和阈值不变，无fallback或额外try。峰值范围由初始化之后开始，与原GPU0循环一致，不能称包括构造或独立推理测速。根自查仍不是独立审计；真实M0/正式性能尚无。旧865失败不改判。\n\n另立866执行，fresh prepares→逐端8M0→fresh50→首次strict→唯一九端CPU报告，entrySHA已变所以不复用旧3初始化binding。旧3PASS仍保留。预算5,922,357,248B、source372中旧366/RAW187不变、仅26GPU0/1、一NN/no功率温度/25范围不变。已退役六份权重不再重放或重复删除。此时无活动NN/observer，下一步同步新源后一次启动修订，Goal ACTIVE/UNMET。\n'''
start=text.index('**当前进度：');end=text.index('\n\n',start)
text=text[:start]+'**当前进度：§41.866（2026-10-06更新）。** 原九端首次执行在三份RGBNT201真实初始化及配对状态通过后，首M0入口因CUDA尚未初始化就重置显存统计而EXIT1；更新0、正式成绩0。原失败保留。只将统计调用移到模型分段后，新旧去统计语句AST一致，研究模型/目标/阈值不变。修订执行源已登记，尚未真实M0/full。六份dominated旧best已退役且日志/距离/依赖保留。只26 GPU0/1、无功率温度动作，25原I/O pending不探测。Goal ACTIVE/UNMET。以下旧首页细节为历史摘要。'+text[end:]
path.write_text(text,encoding='utf-8');(Path('C:/Users/gb/Desktop/document')/path.name).write_bytes(path.read_bytes())
current=repo/'refine-logs/CURRENT_GOAL.md';current.write_text(current.read_text(encoding='utf-8')+'\n\n§41.866：原865 terminalEXIT1，3初始化PASS但首M0在build前统计reset失败，更新0/formal0。只移动GPU1统计调用至build/partition之后；去统计语句AST相等。修订866待一次启动freshprepare/M0/full，旧失败不重跑改判。Goal ACTIVE/UNMET。\n',encoding='utf-8')
changed={doc,'refine-logs/CURRENT_GOAL.md','tools/run_signal_selection_reference.py',*['refine-logs/signal_selection_reference_v1/'+n for n in ('EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','LOCAL_SOURCE_REVIEW.md','MANIFEST.json','SOURCE_SCOPE.json')]}
changed.update(str(p.relative_to(repo)).replace('\\','/') for p in archive.rglob('*') if p.is_file())
protected=json.loads((proof/'publication865_local.json').read_bytes())['protected_files'];assert all(hashlib.sha256((repo/n).read_bytes()).hexdigest()==d for n,d in protected.items())
record=dict(previous_head=previous['head'],files=sorted(changed),protected_files=protected,immutable_archive_roots=[str(archive.relative_to(repo)).replace('\\','/')],remote_sparse_roots=[str(archive.relative_to(repo)).replace('\\','/')])
(proof/'publication866_local.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status='STATS_ONLY_REVISION_REGISTERED_NOT_PUSHED',files=len(changed),source_count=len(sources),original3_prepare=True,m0_updates=0)))
