from pathlib import Path
import hashlib,json,paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
plan=(base/'selection_legacy20_qualification868/PLAN.json').read_bytes()
packet=base/'selection_legacy20_retirement868'
assert not packet.exists();packet.mkdir()
code=r'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
plan=json.loads(PLAN)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
launch=root/'logs/signal_selection_reference_launch_20261006_867'
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1
for name in ('LAUNCH.json','CHILD.json'):
    r=json.loads((launch/name).read_text());p=Path('/proc')/str(r['pid'])
    assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
assert len(protected)==187 and all(sha(p)==s for p,s in protected.items())
keep={r['path']:r for r in plan['retained']}
assert len(plan['targets'])==11 and sum(r['target']['bytes'] for r in plan['targets'])==379316962
proofs=[]
for pair in plan['targets']:
    target,winner=pair['target'],pair['winner']
    assert winner['path'] in keep and target['group']==winner['group']
    assert all(winner['metrics'][k]>=target['metrics'][k] for k in ('mAP','Rank-1','Rank-5','Rank-10'))
    assert winner['metrics']['mAP']>target['metrics']['mAP']
    for row in (target,winner):
        p=Path(row['path']).resolve();assert p.is_relative_to(root/'trained-model') and p.name=='roles_epoch20.pth'
        assert p.stat().st_size==row['bytes'] and sha(p)==row['checkpoint_sha256']
        assert sha(p.parent/'training.json')==row['training_sha256']
        assert sha(p.parent/'official_metrics.json')==row['receipt_sha256']
        t=json.loads((p.parent/'training.json').read_text())
        receipt=json.loads((p.parent/'official_metrics.json').read_text())
        assert t['status']=='FIXED_EPOCH20_TRAINING_COMPLETE' and t['training']['epochs']==20
        assert sha(row['distance_arrays'])==row['distance_arrays_sha256']
        assert sha(t['protocol'])==t['protocol_sha256']
        assert sha(t['initializer']['author_checkpoint'])==t['initializer']['author_checkpoint_sha256']
        assert receipt['independent_upstream_metrics_equal']
    assert target['path'] not in protected
    proofs.append(pair)
for row in keep.values():assert sha(row['path'])==row['checkpoint_sha256']
probe=root/'trained-model/signal_selection_reference_v1_20261006_867_m0_global_only_MSVR310/m0_reload_probe.pth'
assert sha(probe)=='23c4e50e21ec5a4b5f3f297d04398617180ce6689e9eb20bc0c75788e4adf8e9'
out=root/'logs/selection_legacy20_retirement_20261006_868'
assert not out.exists();out.mkdir()
before=shutil.disk_usage(root).free
qualified=dict(status='QUALIFIED_BEFORE_DELETE',at=datetime.now().astimezone().isoformat(),plan_sha256=hashlib.sha256(PLAN.encode()).hexdigest(),
    proofs=proofs,retained_checkpoint_sha256={p:r['checkpoint_sha256'] for p,r in keep.items()},free_before_bytes=before,
    boundary='Only11 closed fixed20 inferior role weights; all4metric dominated within matched method/dataset/protocol/author+Signal init. Official receipts/logs/distances/protocols retained. Authors/current50winners/currentMSVRprobe/RAW187 preserved. Legacybinary replays for11 now retired; reconstruct via recorded sources/recipes ifneeded. NoNN/eval/GPU/25/temp/power action.')
(out/'QUALIFIED.json').write_text(json.dumps(qualified,indent=2)+'\n')
for pair in proofs:
    row=pair['target'];p=Path(row['path'])
    record=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,winner=pair['winner']['path'],at=datetime.now().astimezone().isoformat())
    assert record['sha256']==row['checkpoint_sha256'] and record['bytes']==row['bytes']
    with (out/'RETIREMENTS.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    p.unlink()
assert all(not Path(p['target']['path']).exists() for p in proofs)
assert all(sha(p)==r['checkpoint_sha256'] for p,r in keep.items())
assert all(sha(p)==s for p,s in protected.items())
assert sha(probe)=='23c4e50e21ec5a4b5f3f297d04398617180ce6689e9eb20bc0c75788e4adf8e9'
result=dict(status='COMPLETE',at=datetime.now().astimezone().isoformat(),retired=11,logical_retired_bytes=379316962,free_before_bytes=before,free_after_bytes=shutil.disk_usage(root).free,
    retained_weights=len(keep),protected_artifacts=187,current_msvr_probe_retained=True,out_dir=str(out))
(out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(result=result,files={str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in out.iterdir()})))
'''.replace('PLAN',repr(plan.decode()),1)
(packet/'SOURCE.py').write_text(code,encoding='utf-8');compile(code,'remote-retirement','exec')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(180)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error);(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n')
assert status==0,error.decode()
r=json.loads(data);sftp=c.open_sftp()
for name,info in r['files'].items():
    p=packet/'received'/name;p.parent.mkdir(parents=True,exist_ok=True)
    sftp.get('/data/gaob/Re-ID/Trifusion/'+name,str(p));assert p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
sftp.close();c.close();print(json.dumps(r['result'],indent=2))
