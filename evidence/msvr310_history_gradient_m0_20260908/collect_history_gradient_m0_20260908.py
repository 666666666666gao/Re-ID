from pathlib import Path
import hashlib,json,shlex
tmp=Path('C:/Users/gb/.codex_tmp')
launch=json.loads((tmp/'history_gradient_training_launch_20260908.json').read_bytes())
dest=tmp/'history_gradient_m0_complete_20260908';assert not dest.exists()
exec((tmp/'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
script=r'''
from pathlib import Path
import hashlib,json
root=Path(__ROOT__)
s=json.loads((root/'m0/summary.json').read_bytes());v=json.loads((root/'m0_cpu.json').read_bytes())
assert s['status']=='PASS_ENGINEERING_ONLY' and s['optimizer_steps']==248
assert v['status']=='PASS_COMPLETE_HISTORY_GRADIENT_M0'
assert v['summary_sha256']==hashlib.sha256((root/'m0/summary.json').read_bytes()).hexdigest()
paths=[root/'t0.json',root/'m0_cpu.json',root/'m0/summary.json',root/'m0.log',root/'m0_cpu.log']
paths+=sorted((root/'m0').glob('*/receipt.json'))+sorted((root/'m0').glob('*/training.json'))+sorted((root/'m0').glob('*/memory_steps.jsonl'))
assert len(list((root/'m0').glob('*/receipt.json')))==6
assert len(list((root/'m0').glob('*/training.json')))==8
print(json.dumps([dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
'''.replace('__ROOT__',repr(launch['root']))
_,out,err=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -c '+shlex.quote(script))
data=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
rows=json.loads(data);dest.mkdir();sftp=c.open_sftp()
for row in rows:
 target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
 sftp.get(launch['root']+'/'+row['path'],str(target))
 b=target.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
(dest/'intake_manifest.json').write_text(json.dumps(dict(run=launch['root'],files=rows),indent=2)+'\n',encoding='utf-8')
sftp.close();c.close()
s=json.loads((dest/'m0/summary.json').read_bytes());v=json.loads((dest/'m0_cpu.json').read_bytes())
result=dict(m0=s['status'],completed_at=s['completed_at'],cpu=v['status'],cpu_steps=v['checked_training_steps'],distance_elements=v['checked_memory_distance_elements'],vjp_record_forwards=v['checked_history_vjp_record_forwards'],overfit={e:r['gate'] for e,r in s['overfit'].items()},intake_files=len(rows),intake_bytes=sum(r['bytes'] for r in rows))
(dest/'completion_brief.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
