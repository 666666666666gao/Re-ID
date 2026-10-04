from datetime import datetime
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import subprocess

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private=Path('C:/Users/gb/.codex_tmp')
proof=private/'foundation_recipe_v1_20261002'
previous=json.loads((proof/'five_copy819.json').read_bytes())
assert previous['section']=='41.819'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
assert subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo)==b''
assert not (proof/'publication820_local.json').exists()
evidence=private/'independent_evidence_draft'
retired=evidence/'closed_m3_probe_retirement_v2_20261004'
assert json.loads((retired/'EXIT.json').read_bytes())['exit_code']==0
record=json.loads((retired/'RETIREMENT.json').read_bytes())
assert record==json.loads((retired/'stdout.json').read_bytes())
assert record['status']=='RETIRED_EXACT_12_CLOSED_M3_M0_PROBES'
assert len(record['rows'])==12 and all(row['deleted'] for row in record['rows'])
assert record['retired_bytes']==162103728
assert record['current_required_probes_preserved']==5
assert record['control_artifact_count']==61 and record['source_file_count']==322
archive_name='logs/closed_m3_probe_retirement820_20261004'
archive=repo/archive_name
assert not archive.exists()
archive.mkdir()
for folder in ('closed_m0_inventory_20261004','closed_m3_probe_dependencies_20261004',
               'closed_m3_probe_retirement_preflight_20261004','closed_m3_probe_retirement_preflight_v2_20261004',
               'closed_m3_probe_retirement_20261004','closed_m3_probe_retirement_v2_20261004'):
    source=evidence/folder
    assert source.is_dir()
    shutil.copytree(source,archive/folder)
(archive/'helpers').mkdir()
for name in ('inventory_closed_m0_weights_20261004.py','inspect_closed_m3_probe_dependencies_20261004.py',
             'prepare_closed_m3_probe_retirement_20261004.py','prepare_closed_m3_probe_retirement_v2_20261004.py',
             'retire_exact_closed_m3_probes_20261004.py','retire_exact_closed_m3_probes_v2_20261004.py',
             'register_closed_m3_probe_retirement820.py'):
    shutil.copyfile(private/name,archive/'helpers'/name)
intro='旧M3预测12端已完成后，精确退役12份临时M0探针，实删162,103,728B；12份正式best、结果回执、当前5份必需M0和61件固定best诊断依赖均SHA不变。未新增训练或检索成绩，继续等待原RGBNT100队列的14:05观察。仅26GPU0/1，不执行功率温度操作，Goal active/unmet。'
docname='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
docpath=repo/docname
assert hashlib.sha256(docpath.read_bytes()).hexdigest()==previous['doc_sha256']
doc=docpath.read_text(encoding='utf-8')
header=next(line for line in doc.splitlines() if line.startswith('**当前进度：'))
assert '41.819' in header and '## 41.820 ' not in doc
doc=doc.replace(header,'**当前进度：§41.820（2026-10-04更新）** '+intro)
doc+='\n\n## 41.820 已闭合旧M3预测面板的临时探针退役与当前训练保护\n\n'+intro+'\n\n'
doc+=f'实际退役完成于{record["completed_at"]}。旧面板logs/correspondence_m3_prediction_20260929的12个条件均完成M0、完整50轮和首次独立评价；原accepted_complete_656矩阵为12/12 VERIFIED_COMPLETE。清理前逐项复核正式best、距离、官方回执、M0回执与原存档SHA，仅移除下表12个m0_reload_probe.pth；未删除任一正式mAP-best。\n\n'
doc+='| 探针所属run | 实删字节 | 删除前SHA256 |\n|---|---:|---|\n'
for row in record['rows']:
    doc+=f'| {Path(row["path"]).parent.name} | {row["bytes"]} | {row["sha256"]} |\n'
doc+=f'\n实删合计{record["retired_bytes"]}B（{record["retired_bytes"]/1024**2:.2f}MiB）。删除前磁盘free为{record["free_bytes_before"]}B，删除后为{record["free_bytes_after"]}B；训练仍可能写日志，磁盘差值不能替代逐文件实删字节。远端保留logs/closed_m3_probe_retirement_20261004下的PLAN、逐文件DELETION_EVENTS与RETIREMENT。\n\n'
doc+='本轮初始库存共有99份探针，当前队列5份仍是原一次CPU报告的必需依赖，全部保留。其余旧探针未因文件名或年龄自动删除；本轮只处理具有12端闭合证据的这一个面板。所有必要初始化、旧61件固定best依赖及当前322源文件均在删除前后核对SHA一致。旧M3收集器读取M0文本回执，不依赖探针二进制；但历史M0二进制重载探针本身已不能直接重放，不得补训生成探针来冒充原物。\n\n'
doc+='保留两个清理辅助检查失败：第一版preflight把子campaign父目录误写成logs，实际源码child_campaign明确位于父campaign内；第二个传输辅助脚本嵌套换行发生语法错误。两次均在任何unlink之前失败，没有模型forward、训练或权重变化。修正仅涉及清理辅助脚本路径合同与字符串构造，第二版通过，原失败记录随本节归档。此项不计为算法或训练修复。\n\n'
doc+='训练状态沿用§819的12:27观察和12:52原进程身份记录，不伪称本节又取得训练终态。既有14:05观察器保持单份，未重启训练、未提前执行新固定best诊断。后续依原队列收齐RGBNT100 semantic/native的50轮与首次严格评价、原一次CPU报告，再封存新固定best诊断输入。科学目标与已封存负结果不变。\n'
docpath.write_text(doc,encoding='utf-8')
goalpath=repo/'refine-logs/CURRENT_GOAL.md'
goal=goalpath.read_text(encoding='utf-8')
updated=next(line for line in goal.splitlines() if line.startswith('更新：'))
goal=goal.replace(updated,'更新：2026-10-04 §41.820；'+intro)
goal+='\n\n§41.820存储收尾：旧M3闭合12端仅退役12份临时M0探针，实删162,103,728B；12正式best/回执、当前5必需探针、61件固定best依赖、322源文件不变。剩余未证实无用的旧探针保留。原RGBNT100队列及14:05单次观察不变，无新分数，Goal active/unmet。证据见logs/closed_m3_probe_retirement820_20261004。\n'
goalpath.write_text(goal,encoding='utf-8')
desktop=Path('C:/Users/gb/Desktop/document')/docpath.name
assert hashlib.sha256(desktop.read_bytes()).hexdigest()==previous['doc_sha256']
desktop.write_bytes(docpath.read_bytes())
owned=[docname,'refine-logs/CURRENT_GOAL.md']+sorted(path.relative_to(repo).as_posix() for path in archive.rglob('*') if path.is_file())
protected=json.loads((proof/'publication819_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest for name,digest in protected.items())
(proof/'publication820_local.json').write_text(json.dumps({'previous_head':previous['head'],'section':'41.820','files':owned,'protected_files':protected,
    'immutable_archive_roots':[archive_name],'doc_sha256':hashlib.sha256(docpath.read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
source=(private/'publish_rodi_protocol819.py').read_text(encoding='utf-8')
replacements={
    'publication819_local.json':'publication820_local.json','five_copy818.json':'five_copy819.json',
    'commit819.json':'commit820.json','target819.bundle':'target820.bundle',
    'Verify RoDI primary paper protocol boundaries and CPU diagnosis entry':'Retire exact twelve closed M3 probe weights and preserve current queue',
    'range(739,820)':'range(739,821)','logs/rodi_protocol_and_cpu_help819_20261004':archive_name,
    '/tmp/trifusion_target819_20261004.bundle':'/tmp/trifusion_target820_20261004.bundle',
    "'section':'41.819'":"'section':'41.820'","five_copy819.json').write_text":"five_copy820.json').write_text"}
assert all(key in source for key in replacements)
source=re.sub('|'.join(re.escape(key) for key in sorted(replacements,key=len,reverse=True)),lambda match:replacements[match.group()],source)
source=re.sub(r"'boundary':'Primary RoDI paper[^']+'", "'boundary':'Exact12 closed old M3 M0 probes retired, all12 formal best and current5 required probes preserved. Source322 and sealed61 unchanged. Formal4/6 remains historical until14:05 scheduled observation; original training untouched. Only26GPU0/1,25textonly,no power/temp. Scientific Goal active/unmet.'",source)
ast.parse(source)
target=private/'publish_closed_m3_probe_retirement820.py'
assert not target.exists()
target.write_text(source,encoding='utf-8')
wrapper=(private/'run_rodi_protocol819_publication.py').read_text(encoding='utf-8').replace('publication819_attempt','publication820_attempt').replace('publish_rodi_protocol819.py','publish_closed_m3_probe_retirement820.py')
ast.parse(wrapper)
wrapper_target=private/'run_closed_m3_probe_retirement820_publication.py'
assert not wrapper_target.exists()
wrapper_target.write_text(wrapper,encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\n'+datetime.now().astimezone().isoformat()+' TriFusion exact12 closed oldM3 M0 probes retired162103728B, current5 required probes/formal12best/source322/sealed61 preserved. Cleanup helper failures before unlink retained. Section820 publication prepared, not published yet. RGBNT100 original queue remains, single14:05 observer, no power/temp.\n')
print(json.dumps({'status':'EXACT_12_CLOSED_M3_PROBE_RETIREMENT_PUBLICATION_PREPARED','owned_files':len(owned),'retired_bytes':record['retired_bytes']}))
