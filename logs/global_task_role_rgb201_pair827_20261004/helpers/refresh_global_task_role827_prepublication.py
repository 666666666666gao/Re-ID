"""Append newer original-queue observation to the unpublished section."""
from pathlib import Path
import hashlib,json,shutil
p=Path('C:/Users/gb/.codex_tmp');repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
manifest=p/'foundation_recipe_v1_20261002/publication827_local.json'
record=json.loads(manifest.read_bytes())
observer=p/'independent_evidence_draft/global_task_role_observations/204524_299592'
r=json.loads((observer/'stdout.json').read_bytes())
assert json.loads((observer/'EXIT.json').read_bytes())['exit_code']==0
assert sum(j['phase']=='full' and j['status']=='COMPLETE' for j in r['jobs'])==3
assert sum(j['phase']=='m0' and j['status']=='COMPLETE' for j in r['jobs'])==4
assert any(x['pid']==2934764 and x['mode']=='train' for x in r['active_processes'])
archive=repo/'logs/global_task_role_rgb201_pair827_20261004'
shutil.copytree(observer,archive/'publication_time_vehicle_observation')
shutil.copyfile(Path(__file__),archive/'helpers'/Path(__file__).name)
note='20:45发布前补查：原队列正式3/6、M0 4/6；MSVR310 semantic完成50轮并首次严格重载，E38为50.5422mAP/67.8511R1，完整CPU文本与配对接收待做。native已完成23轮，原训练PID2934764/ticks37257914状态R，supervisor原PID/ticks不变。仅26GPU0/1，不管功温；剩余6,224,576,512B。上述20:09的2/6状态为历史观察，当前以此为准；科学source330不变，Goal active/unmet。'
doc=repo/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
assert hashlib.sha256(doc.read_bytes()).hexdigest()==record['doc_sha256']
doc.open('a',encoding='utf-8').write('\n\n'+note+'\n')
goal=repo/'refine-logs/CURRENT_GOAL.md';s=goal.read_text(encoding='utf-8')
line=next(x for x in s.splitlines() if x.startswith('更新：'))
goal.write_text(s.replace(line,'更新：2026-10-04 §41.827。'+note,1)+'\n\n§41.827补查 '+note+'\n',encoding='utf-8')
(repo/'refine-logs/global_task_role_v1/EXPERIMENT_TRACKER.md').open('a',encoding='utf-8').write('\n\n'+note+'\n')
(Path('C:/Users/gb/Desktop/document')/doc.name).write_bytes(doc.read_bytes())
record['files']=record['files']+[x.relative_to(repo).as_posix() for x in (archive/'publication_time_vehicle_observation').rglob('*') if x.is_file()]+[(archive/'helpers'/Path(__file__).name).relative_to(repo).as_posix()]
record['doc_sha256']=hashlib.sha256(doc.read_bytes()).hexdigest()
manifest.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'UNPUBLISHED827_CURRENT_OBSERVATION_APPENDED','files':len(record['files'])}))
