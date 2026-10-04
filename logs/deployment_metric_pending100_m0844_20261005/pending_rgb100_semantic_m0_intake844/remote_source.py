
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/deployment_metric_role_pending100_20261005_842'
state=json.loads((campaign/'campaign.json').read_text())
job=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('RGBNT100','semantic','m0'))
assert job['status']=='COMPLETE' and job['exit_code']==0
binding=json.loads((campaign/'initialization/RGBNT100_semantic.json').read_text())['binding']
m0=root/'trained-model/deployment_metric_role_pending100_20261005_842_m0_semantic_RGBNT100'
value=json.loads((m0/'training.json').read_text())
assert job['result']==value and value['status']=='M0_PASS'
assert value['schema']=='trifusion-deployment-metric-role-v1' and value['initializer']==binding
assert binding['role_metric_policy']=='role_triplet_joint_1536_l2_h_global_triplet_original_raw_parts'
assert value['production_m0_diagnostics']['effective_optimizer_updates']==8
assert all(count==8 for count in value['production_m0_diagnostics']['author_bn_batches_tracked'].values())
assert value['m0']['trainable_parameters']==value['m0']['nonzero_gradient_parameters']==binding['trainable_parameter_tensors']
probe=m0/'m0_reload_probe.pth'
assert hashlib.sha256(probe.read_bytes()).hexdigest()==value['m0']['reload_probe_sha256']
names=['logs/deployment_metric_role_pending100_20261005_842/manifest.json','logs/deployment_metric_role_pending100_20261005_842/campaign.json',
 'logs/deployment_metric_role_pending100_20261005_842/initialization/RGBNT100_semantic.json',
 'logs/deployment_metric_role_pending100_20261005_842/prepare_RGBNT100_semantic.log','logs/deployment_metric_role_pending100_20261005_842/RGBNT100_semantic_m0.log',
 'logs/deployment_metric_pending100_launch_20261005_843/LAUNCH.json','logs/deployment_metric_pending100_launch_20261005_843/supervisor.py']
names += [p.relative_to(root).as_posix() for p in m0.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.txt')]
files={name:dict(sha256=hashlib.sha256((root/name).read_bytes()).hexdigest(),bytes=(root/name).stat().st_size,text=(root/name).read_text()) for name in names}
full=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('RGBNT100','semantic','full'))
active=state['active_command']
assert active['mode']=='train' and active['status']=='RUNNING'
assert Path('/proc/'+str(active['pid'])).exists()
print(json.dumps(dict(status='FIRST_REAL_DEPLOYMENT_METRIC_ROLE_M0_VERIFIED',at=datetime.now().astimezone().isoformat(),
 binding=binding,actual_m0=value,full_phase=full,active_command=active,files=files,probe=dict(path=str(probe),sha256=value['m0']['reload_probe_sha256'],bytes=probe.stat().st_size),
 disk_free_bytes=shutil.disk_usage(root).free,boundary='First8realM0 verified and fresh50 started,not formal performance;no model replay or power/temperature action. Probe retained until ownfull50/firststrict acceptance.')))
