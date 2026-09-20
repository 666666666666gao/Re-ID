"""Receive completed phase text only; no images, model or distance arrays."""
from pathlib import Path
import sys,json,hashlib,shlex

destination=Path(sys.argv[1]);mode=sys.argv[2]
assert mode in ('m0','q1') and not destination.exists()
exec(Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
script=r'''
from pathlib import Path
import json,hashlib,datetime
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
mode=MODE
pipeline=json.loads((root/'pipeline.json').read_bytes())
stages={r['stage']:r for r in pipeline['stages']}
assert stages[mode]['exit_code']==stages[mode+'_cpu']['exit_code']==0
summary=json.loads((root/mode/'summary.json').read_bytes())
proof=json.loads((root/(mode+'_cpu.json')).read_bytes())
assert summary['status'] in (('PASS_ENGINEERING_ONLY',) if mode=='m0' else ('Q1_PASS','Q1_FAIL'))
assert proof['status']=='PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_'+mode.upper()
assert proof['checked_training_steps']==(248 if mode=='m0' else 1560)
assert proof['summary_sha256']==hashlib.sha256((root/mode/'summary.json').read_bytes()).hexdigest()
if mode=='q1':assert pipeline['status']=='COMPLETE_VERIFIED_'+summary['status']
paths=[p for p in (root/mode).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.log')]
paths += [root/'t0.json',root/'t0.log',root/(mode+'.log'),root/(mode+'_cpu.json'),root/(mode+'_cpu.log')]
if mode=='q1':paths.append(root/'pipeline.json')
rows=[]
for p in sorted(paths):
    data=p.read_bytes();data.decode('utf-8')
    rows.append(dict(remote=str(p),path=p.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(dict(status='COMPLETE_PHASE_TEXT_INVENTORY',mode=mode,pipeline_snapshot=pipeline,
 observed_at=datetime.datetime.now().astimezone().isoformat(),files=rows)))
'''.replace('MODE',repr(mode))
_,out,err=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -B -c '+shlex.quote(script))
inventory=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
value=json.loads(inventory);destination.mkdir(parents=True)
(destination/'inventory.json').write_bytes(inventory)
sftp=c.open_sftp()
for row in value['files']:
    with sftp.open(row['remote'],'rb') as stream:
        stream.prefetch();data=stream.read()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    p=destination/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
sftp.close();c.close()
receipt=dict(status='COMPLETE_VERIFIED_PHASE_TEXT_INTAKE',mode=mode,files=len(value['files']),
             bytes=sum(r['bytes'] for r in value['files']),inventory_sha256=hashlib.sha256(inventory).hexdigest(),remote_binary_files_downloaded=0)
(destination/'intake.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
