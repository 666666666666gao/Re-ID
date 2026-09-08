"""Receive complete verified Q1 texts once; leave binary arrays and models remote."""
from pathlib import Path
import hashlib,json,shlex
tmp=Path('C:/Users/gb/.codex_tmp')
launch=json.loads((tmp/'history_gradient_training_launch_20260908.json').read_bytes())
dest=tmp/'history_gradient_q1_complete_20260908';assert not dest.exists()
exec((tmp/'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
script=r'''
from pathlib import Path
import hashlib,json
root=Path(__ROOT__)
pipeline=json.loads((root/'pipeline.json').read_bytes())
assert pipeline['status'] in ('COMPLETE_VERIFIED_Q1_PASS','COMPLETE_VERIFIED_Q1_FAIL')
assert pipeline['code_commit']==__HEAD__ and pipeline['config_sha256']==__CONFIG__
assert all(r['exit_code']==0 for r in pipeline['stages']) and len(pipeline['stages'])==5
s=json.loads((root/'q1/summary.json').read_bytes());v=json.loads((root/'q1_cpu.json').read_bytes())
assert s['optimizer_steps']==v['checked_training_steps']==1560
assert s['official_image_reads']==0 and s['heldout_record_forwards']==2064
assert s['seed']==42 and s['history_anchors']==0
assert v['status']=='PASS_COMPLETE_HISTORY_GRADIENT_Q1'
assert pipeline['terminal_summary_sha256']==v['summary_sha256']==hashlib.sha256((root/'q1/summary.json').read_bytes()).hexdigest()
assert pipeline['terminal_cpu_sha256']==hashlib.sha256((root/'q1_cpu.json').read_bytes()).hexdigest()
paths=[root/x for x in ('pipeline.json','q1/summary.json','q1_cpu.json','q1.log','q1_cpu.log')]
assert len(s['folds'])==3
for fold in s['folds']:
 assert set(fold['endpoints'])=={'control','history_gradient'}
 for end,r in fold['endpoints'].items():
  assert r['training']['optimizer_steps']==260
  directory=root/'q1'/f'fold_{fold["fold"]}_{end}'
  paths += [directory/x for x in ('memory_steps.jsonl','rankings.json','receipt.json','training.json')]
assert len(paths)==29 and all(p.is_file() for p in paths)
print(json.dumps([dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
'''.replace('__ROOT__',repr(launch['root'])).replace('__HEAD__',repr(launch['code_commit'])).replace('__CONFIG__',repr(launch['config_sha256']))
_,out,err=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -c '+shlex.quote(script))
data=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
rows=json.loads(data);dest.mkdir();sftp=c.open_sftp()
for row in rows:
 target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
 sftp.get(launch['root']+'/'+row['path'],str(target))
 b=target.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
(dest/'intake_manifest.json').write_text(json.dumps(dict(run=launch['root'],files=rows),indent=2)+'\n',encoding='utf-8')
sftp.close();c.close()
print(json.dumps(dict(status='RECEIVED_COMPLETE_VERIFIED_Q1_TEXT',files=len(rows),bytes=sum(r['bytes'] for r in rows),destination=str(dest))))
