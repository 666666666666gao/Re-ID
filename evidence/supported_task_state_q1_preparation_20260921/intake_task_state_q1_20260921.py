"""Receive complete original Q1 text after its original full CPU verification."""
from pathlib import Path
import json,hashlib,shlex
destination=Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_task_state_q1_complete_20260921')
assert not destination.exists()
exec(Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
script=r'''
from pathlib import Path
import json,hashlib,datetime
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_task_state_v1_seed42_cb4f4c3')
pipeline=json.loads((root/'pipeline.json').read_bytes())
stages={r['stage']:r for r in pipeline['stages']}
assert stages['q1']['exit_code']==stages['q1_cpu']['exit_code']==0
summary=json.loads((root/'q1/summary.json').read_bytes())
proof=json.loads((root/'q1_cpu.json').read_bytes())
assert summary['status'] in ('Q1_PASS','Q1_FAIL')
assert pipeline['status']=='COMPLETE_VERIFIED_'+summary['status']
assert proof['status']=='PASS_COMPLETE_SUPPORTED_TASK_STATE_Q1'
assert proof['checked_training_steps']==summary['optimizer_steps']==1560
assert proof['summary_sha256']==hashlib.sha256((root/'q1/summary.json').read_bytes()).hexdigest()
assert proof['model_forwards']==proof['optimizer_updates']==0
assert summary['official_image_reads']==0 and summary['heldout_record_forwards']==2064
assert len(summary['folds'])==3
assert all(set(f['endpoints'])=={'control','split'} for f in summary['folds'])
paths=[p for p in (root/'q1').rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.log')]
paths += [root/name for name in ('pipeline.json','t0.json','q1.log','q1_cpu.log','q1_cpu.json')]
rows=[]
for p in sorted(paths):
    data=p.read_bytes();data.decode('utf-8')
    rows.append(dict(remote=str(p),path=p.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(dict(status='COMPLETE_Q1_TEXT_INVENTORY',pipeline_snapshot=pipeline,
 observed_at=datetime.datetime.now().astimezone().isoformat(),files=rows)))
'''
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
receipt=dict(status='COMPLETE_VERIFIED_Q1_TEXT_INTAKE',files=len(value['files']),
    bytes=sum(r['bytes'] for r in value['files']),inventory_sha256=hashlib.sha256(inventory).hexdigest(),
    remote_binary_files_downloaded=0)
(destination/'intake.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
