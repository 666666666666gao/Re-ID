from pathlib import Path
import json
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/closed_m3_probe_retirement_preflight_20261004')
assert not packet.exists()
packet.mkdir()
code='''
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion').resolve()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
closed=root/'logs/correspondence_m3_accepted_complete_656_20260929.json'
matrix=json.loads(closed.read_text())
assert matrix['schema']=='trifusion-correspondence-m3-panel-verification-v1'
assert matrix['expected_endpoints']==matrix['verified_complete']==len(matrix['rows'])==12
parent=root/'logs/correspondence_m3_prediction_20260929'
state=json.loads((parent/'campaign.json').read_text())
assert state['status']=='COMPLETE' and len(state['jobs'])==12
assert all(job['status']=='COMPLETE' and job['exit_code']==0 for job in state['jobs'])
current=root/'logs/role_input_detach_v1_20261004_813/manifest.json'
manifest=json.loads(current.read_text())
assert len(manifest['source_sha256'])==322
assert all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())
seal_path=root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
seal=json.loads(seal_path.read_text())
assert len(seal['artifact_sha256'])==61
assert all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
rows=[]
protected={str(current):sha(current),str(closed):sha(closed),str(parent/'campaign.json'):sha(parent/'campaign.json'),str(seal_path):sha(seal_path)}
for row in matrix['rows']:
    assert row['status']=='VERIFIED_COMPLETE'
    child=Path(row['campaign_dir']).resolve()
    assert child.parent==root/'logs' and child.name.startswith('correspondence_m3_prediction_20260929_m3_')
    record=json.loads((child/'campaign.json').read_text())
    assert record['status']=='COMPLETE' and [job['mode'] for job in record['jobs']]==['m0','train','evaluate']
    assert all(job['status']=='COMPLETE' and job['exit_code']==0 for job in record['jobs'])
    m0=Path(record['jobs'][0]['output_dir']).resolve()
    full=Path(row['run_dir']).resolve()
    assert full==Path(record['jobs'][1]['output_dir']).resolve()==Path(record['jobs'][2]['output_dir']).resolve()
    assert m0.parent==full.parent==root/'trained-model'
    assert m0.name.startswith('correspondence_m3_prediction_20260929_m3_') and m0.name.endswith('_seed42_m0')
    assert full.name==m0.name[:-3]+'_full'
    training=json.loads((full/'training.json').read_text())
    official=json.loads((full/'official_metrics.json').read_text())
    probe_receipt=json.loads((m0/'training.json').read_text())
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and [item['epoch'] for item in training['history']]==list(range(1,51))
    assert official['status']=='COMPLETE' and official['training_epochs']==50
    assert probe_receipt['status']=='M0_PASS' and probe_receipt['initializer']==training['initializer']
    assert official['metrics']==row['metrics'] and official['selected_epoch']==training['best_epoch']==row['best_epoch']
    assert [path.name for path in full.glob('*.pth')]==['best_map.pth']
    for name,key in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('official_metrics.json','receipt_sha256')):
        path=full/name
        assert sha(path)==row[key]
        protected[str(path)]=row[key]
    for path in (child/'campaign.json',full/'training.json',m0/'training.json'):
        protected[str(path)]=sha(path)
    probe=(m0/'m0_reload_probe.pth').resolve()
    assert probe.parent==m0 and not (m0/'m0_reload_probe.pth').is_symlink()
    digest=sha(probe)
    assert digest==probe_receipt['m0']['reload_probe_sha256']
    assert str(probe) not in seal['artifact_sha256']
    rows.append({'path':str(probe),'bytes':probe.stat().st_size,'sha256':digest,'dataset':row['dataset'],'variant':row['variant'],'closed_full_run':str(full)})
current_probes=sorted((root/'trained-model').glob('role_input_detach_v1_20261004_813*/m0_reload_probe.pth'))
assert len(current_probes)==5
for path in current_probes:
    protected[str(path)]=sha(path)
assert len({row['path'] for row in rows})==12 and not {row['path'] for row in rows}.intersection(protected)
print(json.dumps({'schema':'closed-m3-m0-probe-retirement-v1','status':'READY','at':datetime.now().astimezone().isoformat(),
    'root':str(root),'rows':rows,'total_bytes':sum(row['bytes'] for row in rows),'protected_sha256':protected,
    'control_artifacts_sha256':seal['artifact_sha256'],'current_source_sha256':manifest['source_sha256'],'current_probe_count':len(current_probes),
    'disk_free_bytes':shutil.disk_usage(root).free,
    'boundary':'Only these 12 successful closed M3 M0 probe binaries may be removed. Preserve all full50 best checkpoints, evaluation distances, JSON receipts, current five required M0 probes, current 322 source files and original 61 sealed artifacts. Historical M0 binary replay becomes unavailable; no retraining or regenerated probes.'}))
'''
(packet/'remote.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data,error=stdout.read(),stderr.read()
exit_code=stdout.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data)
(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
plan=json.loads(data)
client.close()
print(json.dumps({key:plan[key] for key in ('schema','status','at','total_bytes','current_probe_count','disk_free_bytes','boundary')},indent=2))
