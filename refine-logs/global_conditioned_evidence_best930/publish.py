"""Publish the accepted finite fixed-best diagnosis, without launching work."""
import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess

w = Path(__file__).resolve().parent
r = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
m = Path('C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
git = lambda *a: subprocess.run(['git','-C',str(r),*a], capture_output=True, check=True).stdout
prior = json.loads((w.parent / 'EVIDENCE_TERMINAL_FINAL_929.json').read_text())
assert git('rev-parse','HEAD').decode().strip() == prior['commit']
assert git('rev-parse','origin/main').decode().strip() == prior['commit']
assert not git('diff','--cached','--name-only')
assert sha(m) == sha(r/'docs'/m.name) == prior['master_sha256']
audit = json.loads((w/'review/RUNTIME_AUDIT.json').read_text())
assert audit['fixed_best_decomposition_evidence_accepted'] and not audit['unresolved_blocking_evidence_defects']
approval = json.loads((w/'review/PUBLISH_SOURCE_CHECK.json').read_text())
assert approval['source_ready_for_publication'] and not approval['unresolved_blocking_source_defects']
assert sha(Path(__file__)) == approval['audited_source_sha256']['C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/gcf930/publish.py']
prepared = json.loads((w/'PREPARED_PUBLICATION.json').read_text())
assert all(sha(r/n) == h for n,h in prepared['file_sha256'].items())
assert sha(w/'prepare_publication.py') == prepared['summary_source_sha256']
for p,h in audit['audited_input_hashes'].items():
    assert sha(Path(p)) == h, p
base = 'refine-logs/global_conditioned_evidence_best930/'
names = list(prepared['file_sha256'])
for name in ('RUNTIME_AUDIT.json','RUNTIME_AUDIT.md','PUBLISH_SOURCE_CHECK.json','PUBLISH_SOURCE_CHECK.md'):
    target = r/base/name
    assert not target.exists()
    target.write_bytes((w/'review'/name).read_bytes())
    names.append(base+name)
target = r/base/'publish.py'
assert not target.exists()
target.write_bytes(Path(__file__).read_bytes())
names.append(base+'publish.py')
report = (r/base/'RESULT_REPORT.md').read_text(encoding='utf-8')
before = m.read_bytes()
assert not (w/'MASTER_BEFORE_930.md').exists()
(w/'MASTER_BEFORE_930.md').write_bytes(before)
head, sep, rest = before.partition(b'\r\n')
assert sep and b'41.929' in head
header = '> **当前状态（2026-10-13，§41.930）：原gradient/random两端50轮已结束；各自E8 best的G/C/F补测完成、原F距离差0、0更新。own-global均74.2967，修正分别仅+0.0678/+0.1729 mAP；梯度机制未胜随机，原失败不改判。仅26GPU0/1，全部自有训练/诊断进程已终止，不启动新训练。主Goal ACTIVE_UNMET。**'
section = '\n\n## 41.930 每条新增证据在最终比较中的实际净用途\n\n' + report
section += '\n本节覆盖§929的“同模型G/C/F尚未测量”状态，旧节保留为历史记录。源审计PASS_SOURCE_ONLY/总体WARN，实际运行完整，文本审计同家族provisional；审查者未SSH或重读tensor。下列公开附件保留原seal、两份真实DIAGNOSIS、两个child EXIT、controller EXIT、一次180秒延迟观察及报告生成源码。没有新权重、模型/loss/层号/比例/gain/margin/seed变化，没有新比较器或车辆队列。训练目标及三数据集超过完整Signal仍未达成。\n'
m.write_bytes(header.encode()+b'\r\n'+rest.rstrip(b'\r\n')+section.replace('\n','\r\n').encode().rstrip(b'\r\n')+b'\r\n')
(r/'docs'/m.name).write_bytes(m.read_bytes())
names.append('docs/'+m.name)
note = '\n2026-10-13 §41.930 authoritative continuation: original two full50 remain closed. One fixed-E8 G/C/F extraction per arm complete, F distance maxdiff0, state/buffers unchanged, 0updates. Both own-global match independent saved query AP/first-rank and mAP74.2966636; own-global→fused gain +0.0677956/+0.1728813. Original gradient−random remains negative/not advanced. New diagnostic supervisor3646738/worker3646739 EXIT0 and absent at15:09:45. Only26physical0/1, no25/T-P. No new training/variant/comparator/seed/vehicle queue. Goal ACTIVE_UNMET. Master930 and global_conditioned_evidence_best930/RESULT_REPORT.md authoritative; no causal/vector-equality/stability claim.\n'
for name in ('AGENTS.md','refine-logs/CURRENT_GOAL.md'):
    p = r/name
    p.write_bytes(p.read_bytes().rstrip(b'\r\n')+note.encode())
    names.append(name)
with (r/'MANIFEST.md').open('ab') as f:
    f.write(f"| {datetime.now().astimezone().isoformat()} | fixed-best diagnosis | {base}RESULT_REPORT.md | two original selected-best G/C/F extractions; zero updates | own-global retained, very thin fusion gain; mechanism still failed |\n".encode())
names.append('MANIFEST.md')
pins = {n:sha(r/n) for n in names}
assert not set(pins).intersection(prior['source_sha256'])
for n in names:
    oid = git('hash-object','-w','--no-filters',str(r/n)).decode().strip()
    git('update-index','--add','--cacheinfo','100644',oid,n)
git('-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
git('commit','-m','Record fixed-best evidence decomposition and retain failed control result')
git('push','origin','main')
commit = git('rev-parse','HEAD').decode().strip()
assert commit == git('rev-parse','origin/main').decode().strip()
remote_ref = git('ls-remote','origin','refs/heads/main').decode().split()[0]
assert remote_ref == commit
assert all(hashlib.sha256(git('show',commit+':'+n)).hexdigest()==h for n,h in pins.items())
assert not git('diff','--cached','--name-only')
seal = json.loads((w/'results/INPUT_SEAL.json').read_text())
observation = json.loads((w/'TERMINAL_OBSERVATION.json').read_text())
request = {'commit':commit,'master_sha256':sha(m),'file_sha256':pins,
           'files_base64':{n:base64.b64encode((r/n).read_bytes()).decode() for n in names},
           'seal_sha256':sha(w/'results/INPUT_SEAL.json'),
           'collected_files':observation['files'], 'physical_gpus':[0,1],
           'supervisor_pid':observation['supervisor_pid'],'worker_pid':observation['worker_pid']}
code = 'request='+repr(request)+'\n'+r'''
import base64,hashlib,json
from pathlib import Path
from datetime import datetime
root=Path('/data/gaob/Re-ID/Trifusion')
admin=root/'logs/global_conditioned_evidence_best930'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(admin/'INPUT_SEAL.json')==request['seal_sha256']
seal=json.loads((admin/'INPUT_SEAL.json').read_text())
assert all(sha(root/n)==h for n,h in seal['source_sha256'].items())
assert all(sha(Path(n))==h for n,h in seal['artifact_sha256'].items())
assert not Path('/proc/'+str(request['supervisor_pid'])).exists()
assert not Path('/proc/'+str(request['worker_pid'])).exists()
assert json.loads((admin/'CONTROLLER_EXIT.json').read_text())['exit_code']==0
assert json.loads((admin/'campaign.json').read_text())['status']=='COMPLETE_TWO_FIXED_BEST_DIAGNOSES'
for n,row in request['collected_files'].items():assert sha(Path(row['remote_path']))==row['sha256']
for exposure in ('gradient','random'):
 d=json.loads((admin/exposure/'DIAGNOSIS.json').read_text())
 assert d['status']=='COMPLETE' and d['optimizer_updates']==0
 for n,row in d['artifacts'].items():assert sha(admin/exposure/n)==row['sha256']
for n,h in request['file_sha256'].items():
 p=root/n;assert p.resolve().is_relative_to(root.resolve())
 data=base64.b64decode(request['files_base64'][n]);assert hashlib.sha256(data).hexdigest()==h
 if n in seal['source_sha256']:assert h==seal['source_sha256'][n]
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
assert all(sha(root/n)==h for n,h in request['file_sha256'].items())
assert all(sha(root/n)==h for n,h in seal['source_sha256'].items())
assert all(sha(Path(n))==h for n,h in seal['artifact_sha256'].items())
value={k:v for k,v in request.items() if k!='files_base64'}
value.update(status='FIXED_BEST_GCF_GIT_DESKTOP_26_EXACT',at=datetime.now().astimezone().isoformat(),
 new_training_started=False,optimizer_updates=0,goal_status='ACTIVE_UNMET',
 source_files_unchanged=len(seal['source_sha256']),input_artifacts_unchanged=len(seal['artifact_sha256']),
 original_bests_and_diagnostic_arrays_retained=True,controller_and_worker_absent=True)
(admin/'PUBLICATION_SYNC.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
'''
opts = ['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run = subprocess.run(['ssh',*opts,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
for name,data in (('PUBLISH.stdout',run.stdout),('PUBLISH.stderr',run.stderr)):
    (w/name).write_text(data,encoding='utf-8')
(w/'PUBLISH_TRANSPORT_EXIT.json').write_text(json.dumps({'at':datetime.now().astimezone().isoformat(),'exit_code':run.returncode},indent=2)+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout)
assert sha(m)==sha(r/'docs'/m.name)==value['master_sha256']
(w/'FINAL.json').write_text(json.dumps(value,indent=2)+'\n')
memo=f"\nTriFusion {value['at']} section41.930: two fixed-E8 G/C/F diagnoses complete, 0updates/Fdistance maxdiff0. Both ownG mAP74.2966636/query AP+first-rank match savedindependentG, not vector/distance bitwise proof. G→F gain gradient+0.0677956/random+0.1728813; originalprimary negative, no mechanismadvance. Diagnostic3646738/3646739 EXIT0/dead, originaltraining stopped. Git{commit}/master{value['master_sha256']} exactGit/Desktop/26;472source/44inputartifacts/newarrays retained, no newweight/cleanupneeded. No newtraining/seed/tuning/vehicles; only26physical01/noT-P. GoalACTIVE_UNMET. Usernorthstar: every added identity evidence must help finalcomparison; externalmethods answerone link, no three-module/+10 naming-first design.\n"
for p in (Path('C:/Users/gb/memory/2026-10-13.md'),Path('C:/Users/gb/MEMORY.md')):
    with p.open('a',encoding='utf-8') as f:f.write(memo)
print(json.dumps({k:value[k] for k in ('status','at','commit','master_sha256','source_files_unchanged','input_artifacts_unchanged','new_training_started')}))
