from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import shutil
import subprocess

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp')
evidence=base/'independent_evidence_draft'
proof=base/'foundation_recipe_v1_20261002'
prior=json.loads((proof/'four_copy871_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==prior['head']
packet=evidence/'role_head_preparation872'
ready=json.loads((packet/'REMOTE.json').read_bytes())
assert ready['retired']==4 and ready['retired_bytes']==1407383008
assert ready['cpu_witness']['status']=='CPU_HEAD_COPY_AND_OWNERSHIP_PASS'
assert ready['free_after']>=ready['required_start_bytes']==3758096384
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
old_scope=repo/'refine-logs/signal_selection_reference_v1/SOURCE_SCOPE.json'
old_scope_sha=sha(old_scope)
protected={n:sha(repo/n) for n in ('.aris/meta/events.jsonl','tools/run_trifusion_experiment.py')}
for p in (repo/'refine-logs/cross_depth_role_state_v1/initialization_source_check_666_20260929').rglob('*'):
    if p.is_file(): protected[p.relative_to(repo).as_posix()]=sha(p)
assert len(protected)==10
changed=set(ready['source_sha256'])
assert all(sha(repo/n)==d for n,d in ready['source_sha256'].items())
folder=repo/'refine-logs/independent_role_heads_v1'
plan=folder/'EXPERIMENT_PLAN_20261006_230600.md'
shutil.copyfile(plan,folder/'EXPERIMENT_PLAN.md')
initial_tracker=folder/'EXPERIMENT_TRACKER_20261006_230600.md'
tracker=initial_tracker.read_text(encoding='utf-8').replace('| 初值、梯度、BN、重载 | NOT_RUN |', '| 初值、梯度、BN、重载 | CPU_PASS |')
new_tracker=folder/'EXPERIMENT_TRACKER_20261006_231200.md'
assert not new_tracker.exists()
new_tracker.write_text(tracker,encoding='utf-8')
shutil.copyfile(new_tracker,folder/'EXPERIMENT_TRACKER.md')
review=folder/'REVIEW.md'
assert not review.exists()
review.write_text('''# 源码自审：独立fused训练头

审阅来源：root self-review，same-family/provisional；没有子代理、跨家族复现或论文评分。

主干、原semantic角色读取/桥接、1536维推理与原RAW身份目标均不改。只有角色分类BN/线性头从原头deepcopy并独立学习；不加一个新的局部辅助loss。原global头继续只学global，角色g/stages仍detach。两任务梯度与buffer归属经一次CPUdirect/vehicle合成见证验证。

新增头的可训练容量与独立BN运行统计明确披露，不能称严格同容量或原创模块。它不保证未知身份检索改善，不能从源SIM车辆增益推断头就是唯一原因。

CPU检查一次通过：初值全部输出/logits一致，任务梯度分离，新2/6张量累计活动/实际变化及完整组件state重载通过。合成fixture为4训练类，因此7680参数是fixture预算，不能冒充生产数据集参数量。真实新增参数由initializer按1536×(生产训练类+1)核算。

新M0记录所有参数活动、原global与新role各BN8次、新头实际更新及完整重载；保留旧AMP/full-batch/first-strict合同。全模型初始化、M0、正式训练均尚未执行，旧parity/读取detach/joint-L2失败不追认通过。

队列在CPU报告启动前将COMPLETE/report1持久化；这避免重复已观测的旧报告状态写入时序问题，未修改原失败。新三端完成后只一次六配对报告，不重复旧控制或搜索官方超参。各端验收后只退役自己的M0，全部共享依赖保留。

五个新Python AST通过，CPU输出和实际四份清理journal归档。该自审仅确认登记范围内的接入准备；生产验证与检索有效性均未获得。
''',encoding='utf-8')
new_sources=list(sorted(changed)) + [p.relative_to(repo).as_posix() for p in folder.iterdir() if p.is_file()]
sources=json.loads(old_scope.read_text())['source_sha256']
assert len(sources)==374
for n in new_sources:
    assert n not in sources,n
    sources[n]=sha(repo/n)
scope=folder/'SOURCE_SCOPE.json'
assert not scope.exists()
scope.write_text(json.dumps(dict(schema='trifusion-independent-role-heads-v1',
    parent_source_scope_sha256=old_scope_sha,source_sha256=sources,
    control_seal='refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json',
    boundary='Old374 scope unchanged; only five new Python files and own plan/tracker/review. Raw187 protected. Source self-review/CPU preparation, not production M0 or retrieval evidence.'),indent=2)+'\n',encoding='utf-8')
changed.update(p.relative_to(repo).as_posix() for p in folder.iterdir() if p.is_file())
archive=repo/'logs/independent_role_head_preparation872_20261006'
assert not archive.exists()
for name in ('role_head_preflight872','role_head_preparation872'):
    for p in (evidence/name).rglob('*'):
        if not p.is_file(): continue
        assert p.suffix.lower() not in ('.pt','.pth','.npy','.npz')
        out=archive/name/p.relative_to(evidence/name)
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,out)
        changed.add(out.relative_to(repo).as_posix())
for n in ('inspect_role_head_preflight872.py','prepare_role_head_components872.py','build_role_head_queue872.py'):
    out=archive/'producers'/n;out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(base/n,out);changed.add(out.relative_to(repo).as_posix())
at=datetime.now().astimezone().isoformat()
doc=repo/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
body=doc.read_text(encoding='utf-8')
assert '## §41.872 ' not in body
body += f'''

## §41.872 — 独立fused身份头三端登记：只检验训练头限制，尚无真实M0（{at}）

已读完整§857–871及实际源码。近容量来源三端、视觉anchor重建六端、按槽位消息分配六端、源SIM纯mask九端均已闭合且未晋级。上一goal轮完成这些证据范围核对，不再将容量/重建/部分分配当尚未尝试的创新。本轮不回放旧队列，也不在失败结构上叠N1/N2/N3。

当前RAW角色经当前global头的detached参数与cloneBN buffer训练，头只接受global目标；源SIM另有可学习var头。但源SIM还改变聚合、容量、目标和3072维输出，因此不能把车辆增益唯一归因于分类头。下一窄假设：原semantic角色是否受到共用global训练头的限制。

新`IndependentRoleHeads`保持原semantic角色、shared/role输入detach、raw作者损失、L_g+L_f、gain和1536L2部署。global头只接受L_g，fused使用从原global头deepcopy的独立BN/分类头，只接受L_f；不是给修正c追加局部ID辅助loss，也没有新外部资源。201一套1536头、车辆三套512头；新增活动参数1536×(训练类+1)、张量数2/6，明确多出训练容量及独立运行统计，不计为论文原创模块。初值clone不消耗新随机数。

一次CPU合成direct/vehicle见证通过：共同state与raw/L2/global/logits初值一致，两任务梯度归属分离，新头累计非零有限梯度/实际更新及组件strict重载一致。fixture为4类，7680是合成预算而非生产参数量。新五文件AST/源码自审为root self-review、same-family/provisional，不是独立审计、全模型M0或检索成功。

唯一计划`refine-logs/independent_role_heads_v1/EXPERIMENT_PLAN.md`：201→MSVR→100，各新semantic端prepare/真实完整batch初值对照/8M0/fresh50/首次strict，然后一次全query六配对报告。旧RAWsemantic/global九行187依赖复用，不重训；新三端batch顺序与对应旧控制逐字节比较。原seed42、作者采样/Adam/raw目标/日程/AMP及first6 GPU1/last6与heads GPU0不改。主要新−RAWsemantic与对独立global推进均为mAP≥+0.5且R1不降，保持单mAP-best全部CMC、完整gallery与原过滤。负结果不救LR/gain/margin/Triplet/seed，也不立即加其他模块。

23:00:49只读核实本项目producer为空，GPU0/1各18MiB，free2,493,485,056B。启动需3best+1probe各384MiB加2GiB，固定3,758,096,384B。23:10:08资格清理完成四份刚闭合且被保留RAWglobal四指标支配的源选择best：201global/masked/allpatch及100global；每份原50轮、首strict回执/实际PTH/训练和评价距离SHA及保留winner已核。实退役1,407,383,008B，free3,895,394,304B；原374来源与RAW187前后SHA一致。四份旧binary直接重放能力退役，原history/receipt/距离/所有分数与既有失败不改；源MSVRglobal及其他未支配best、作者/public/当前对照仍保留。完整资格、CPU与退役journal在`logs/independent_role_head_preparation872_20261006/`，没有模型/NPY回传。

当前真实prepare/pair/M0/full三端均NOT_RUN。只26物理GPU0/1、无25/其他GPU/功率温度动作、无环境重建或消息。预计约3.5–4小时，按实际里程碑/180–300秒观察；NN期间不做source/Git同步。发布完成后才首次启动，旧计算parity失败仍封存。整体Goal ACTIVE/UNMET，三集强净收益、必要机制、完整多种子和同资源SOTA仍缺；2025原I/O pending不探测。
'''
doc.write_text(body,encoding='utf-8')
changed.add(doc.relative_to(repo).as_posix())
desktop=Path('C:/Users/gb/Desktop/document')/doc.name
desktop.write_bytes(doc.read_bytes())
manifest=repo/'MANIFEST.md'
original_manifest_sha=sha(manifest) if manifest.exists() else None
if not manifest.exists():
    manifest.write_text('# Research Output Manifest\n\n| Timestamp | Skill | File | Stage | Description |\n|-----------|-------|------|-------|-------------|\n',encoding='utf-8')
with manifest.open('a',encoding='utf-8') as f:
    for p in sorted(folder.glob('EXPERIMENT_*.md')):
        f.write(f'| {at} | /experiment-plan | {p.relative_to(repo).as_posix()} | implementation | 独立fused训练头的三端固定对照与执行状态 |\n')
changed.add('MANIFEST.md')
assert sha(old_scope)==old_scope_sha
assert all(sha(repo/n)==d for n,d in protected.items())
for n in ready['source_sha256']: ast.parse((repo/n).read_text(encoding='utf-8'))
publication=dict(previous_head=prior['head'],files=sorted(changed),protected_files=protected,
    original_manifest_sha256=original_manifest_sha,uploaded_new_sources=ready['source_sha256'],
    source_count=len(sources),source_scope_sha256=sha(scope),
    remote_sparse_roots=['refine-logs/independent_role_heads_v1','logs/independent_role_head_preparation872_20261006'],
    immutable_archive_roots=['logs/independent_role_head_preparation872_20261006'],
    doc_sha256=sha(doc),section='41.872',at=at)
path=proof/'publication872_local.json';assert not path.exists()
path.write_text(json.dumps(publication,indent=2)+'\n')
print(json.dumps(dict(changed=len(changed),source_count=len(sources),doc_sha256=sha(doc),at=at)))
