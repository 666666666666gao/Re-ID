from pathlib import Path
import hashlib,json
tmp=Path('D:/Program Files/UserCache/gb/codex/tmp')
destination=tmp/'trifusion_supported_task_state_m0_complete_20260921'
assert not destination.exists()
exec(Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
root='/root/trifusion-storage/artifacts/msvr310_supported_task_state_v1_seed42_cb4f4c3'
sftp=c.open_sftp()
def read(name):
    with sftp.open(root+'/'+name,'rb') as stream:return stream.read()
pipeline_bytes=read('pipeline.json');pipeline=json.loads(pipeline_bytes)
for name in ('m0','m0_cpu'):
    stage=[s for s in pipeline['stages'] if s['stage']==name]
    assert len(stage)==1 and stage[0].get('exit_code')==0,(name,stage)
summary_bytes=read('m0/summary.json');summary=json.loads(summary_bytes)
verification_bytes=read('m0_cpu.json');verification=json.loads(verification_bytes)
assert summary['status']=='PASS_ENGINEERING_ONLY' and summary['optimizer_steps']==248
assert verification['status']=='PASS_COMPLETE_SUPPORTED_TASK_STATE_M0'
assert verification['summary_sha256']==hashlib.sha256(summary_bytes).hexdigest()
assert verification['checked_training_steps']==248
assert summary['official_image_reads']==summary['heldout_record_forwards']==0
destination.mkdir()
names=['t0.json','m0.log','m0_cpu.log']
for fold in range(3):
    for end in ('control','split'):
        names.extend(f'm0/fold_{fold}_{end}/{name}' for name in ('receipt.json','training.json','memory_steps.jsonl'))
for end in ('control','split'):
    names.extend(f'm0/overfit_{end}/{name}' for name in ('training.json','memory_steps.jsonl'))
data={'pipeline_at_intake.json':pipeline_bytes,'m0/summary.json':summary_bytes,'m0_cpu.json':verification_bytes}
data.update({name:read(name) for name in names})
rows=[]
for name,payload in data.items():
    target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    digest=hashlib.sha256(payload).hexdigest()
    remote_binding=verification.get('files',{}).get(root+'/'+name)
    if remote_binding:
        assert digest==remote_binding['sha256'] and len(payload)==remote_binding['bytes']
    rows.append(dict(path=name,bytes=len(payload),sha256=digest,cpu_bound=remote_binding is not None))
inventory=dict(status='COMPLETE_M0_TEXT_INTAKE',remote_run=root,files=rows,
              model_tensor_downloads=0,scope='Full M0 text; binary checkpoints and distance tensors remain remote.')
(destination/'intake_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
sftp.close();c.close()
print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),output=str(destination))))
