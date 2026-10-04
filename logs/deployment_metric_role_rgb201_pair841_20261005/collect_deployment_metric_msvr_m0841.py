"""Receive first actual M0 evidence once; do not invoke a new model."""
from datetime import datetime
from pathlib import Path,PurePosixPath
import hashlib,json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'deployment_metric_msvr_semantic_m0_intake841';assert not packet.exists()
code=r'''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/deployment_metric_role_v1_20261005_837'
state=json.loads((campaign/'campaign.json').read_text())
job=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('MSVR310','semantic','m0'))
assert job['status']=='COMPLETE' and job['exit_code']==0
binding=json.loads((campaign/'initialization/MSVR310_semantic.json').read_text())['binding']
m0=root/'trained-model/deployment_metric_role_v1_20261005_837_m0_semantic_MSVR310'
value=json.loads((m0/'training.json').read_text())
assert job['result']==value and value['status']=='M0_PASS'
assert value['schema']=='trifusion-deployment-metric-role-v1' and value['initializer']==binding
assert binding['role_metric_policy']=='role_triplet_joint_1536_l2_h_global_triplet_original_raw_parts'
assert value['production_m0_diagnostics']['effective_optimizer_updates']==8
assert all(count==8 for count in value['production_m0_diagnostics']['author_bn_batches_tracked'].values())
assert value['m0']['trainable_parameters']==value['m0']['nonzero_gradient_parameters']==binding['trainable_parameter_tensors']
probe=m0/'m0_reload_probe.pth'
assert hashlib.sha256(probe.read_bytes()).hexdigest()==value['m0']['reload_probe_sha256']
names=['logs/deployment_metric_role_v1_20261005_837/manifest.json','logs/deployment_metric_role_v1_20261005_837/campaign.json',
 'logs/deployment_metric_role_v1_20261005_837/initialization/MSVR310_semantic.json',
 'logs/deployment_metric_role_v1_20261005_837/prepare_MSVR310_semantic.log','logs/deployment_metric_role_v1_20261005_837/MSVR310_semantic_m0.log',
 'logs/deployment_metric_role_launch_20261005_837/LAUNCH.json','logs/deployment_metric_role_launch_20261005_837/supervisor.py']
names += [p.relative_to(root).as_posix() for p in m0.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.txt')]
files={name:dict(sha256=hashlib.sha256((root/name).read_bytes()).hexdigest(),bytes=(root/name).stat().st_size,text=(root/name).read_text()) for name in names}
full=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('MSVR310','semantic','full'))
active=state['active_command']
assert active['mode']=='train' and active['status']=='RUNNING'
assert Path('/proc/'+str(active['pid'])).exists()
print(json.dumps(dict(status='FIRST_REAL_DEPLOYMENT_METRIC_ROLE_M0_VERIFIED',at=datetime.now().astimezone().isoformat(),
 binding=binding,actual_m0=value,full_phase=full,active_command=active,files=files,probe=dict(path=str(probe),sha256=value['m0']['reload_probe_sha256'],bytes=probe.stat().st_size),
 disk_free_bytes=shutil.disk_usage(root).free,boundary='First8realM0 verified and fresh50 started,not formal performance;no model replay or power/temperature action. Probe retained until ownfull50/firststrict acceptance.')))
'''
compile(code,'remote_msvr_m0_intake841.py','exec')
packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
client.close();assert rc==0,error.decode()
record=json.loads(data)
for name,row in record.pop('files').items():
    target=packet/'texts'/str(PurePosixPath(name));target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(row['text'].encode('utf-8'))
    assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
(packet/'SUMMARY.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],at=record['at'],gradients=record['actual_m0']['m0']['nonzero_gradient_parameters'],
 effective_updates=record['actual_m0']['production_m0_diagnostics']['effective_optimizer_updates'],active_mode=record['active_command']['mode'],active_pid=record['active_command']['pid'],disk_free_bytes=record['disk_free_bytes'])))
