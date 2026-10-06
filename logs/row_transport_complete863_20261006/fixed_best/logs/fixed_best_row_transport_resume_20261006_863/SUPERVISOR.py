settings={'root': '/data/gaob/Re-ID/Trifusion', 'python': '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', 'launch': '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_resume_20261006_863', 'diagnosis': '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862', 'report': '/data/gaob/Re-ID/Trifusion/results/fixed_best_row_transport_diagnosis_20261006_862', 'old': '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_launch_20261006_862', 'source': 'tools/diagnose_row_mass_fixed_best_single.py', 'source_sha256': '60416fafb6b97415edf9bcc24d8065aec61a6ff4b435fbc0b3824f26c8bf2c13'}

from datetime import datetime
import hashlib,json,os,subprocess
from pathlib import Path
launch=Path(settings['launch']);diag=Path(settings['diagnosis']);root=Path(settings['root'])
def write(name,value):(launch/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
steps=[]
for variant in ('semantic','native'):
 command=[settings['python'],'-B',str(root/settings['source']),'--variant',variant,'--diagnosis',str(diag)]
 with (launch/(variant+'.log')).open('x') as log:
  child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
  step=dict(stage='diagnose_'+variant,command=command,pid=child.pid,status='RUNNING',started_at=datetime.now().astimezone().isoformat())
  steps.append(step);write('CHILD.json',dict(status='RUNNING',steps=steps))
  code=child.wait()
 step.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',completed_at=datetime.now().astimezone().isoformat())
 write('CHILD.json',dict(status=step['status'],steps=steps))
 if code:
  write('EXIT.json',dict(exit_code=code,steps=steps,completed_at=datetime.now().astimezone().isoformat()));raise SystemExit(code)
 assert sha(root/settings['source'])==settings['source_sha256']
seal=json.loads((diag/'INPUT_SEAL.json').read_text())
matrix=json.loads((Path(seal['original_campaign'])/'accepted_matrix.json').read_text())
rows=[]
for row in matrix['rows']:
 folder=diag/(row['dataset']+'_'+row['variant'])
 endpoint=json.loads((folder/'ENDPOINT.json').read_text())
 assert len(endpoint['modes'])==3
 if row['dataset']=='RGBNT100':
  receipt=json.loads((folder/'FRESH_PROCESS_RECEIPT.json').read_text())
  assert receipt['status']=='FRESH_PROCESS_ENDPOINT_COMPLETE' and receipt['source_sha256']==settings['source_sha256']
  assert receipt['endpoint_sha256']==sha(folder/'ENDPOINT.json')
 rows.append(endpoint)
assert all(sha(root/n)==h for n,h in seal['original_source_sha256'].items())
assert all(sha(root/n)==h for n,h in seal['diagnostic_source_sha256'].items())
summary=dict(schema=seal['schema'],status='COMPLETE',accepted_models=6,deployment_modes=18,
 completed_at=datetime.now().astimezone().isoformat(),rows=rows,
 original_failed_launch=settings['old'],original_failed_exit_sha256=sha(Path(settings['old'])/'EXIT.json'),
 fresh_process_resume_launch=str(launch),resume_source_sha256=settings['source_sha256'],
 boundary='Four prior accepted endpoints plus two missing fresh-process endpoints. Old failure unchanged; no original12mode reruns, training or tolerance changes.')
assert not (diag/'SUMMARY.json').exists()
(diag/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
command=[settings['python'],'-B',str(root/'tools/report_row_mass_fixed_best.py'),'--diagnosis',str(diag),'--output',settings['report']]
with (launch/'report.log').open('x') as log:
 child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
 step=dict(stage='report',command=command,pid=child.pid,status='RUNNING',started_at=datetime.now().astimezone().isoformat());steps.append(step)
 write('CHILD.json',dict(status='RUNNING',steps=steps));code=child.wait()
step.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',completed_at=datetime.now().astimezone().isoformat())
write('CHILD.json',dict(status=step['status'],steps=steps))
write('EXIT.json',dict(exit_code=code,steps=steps,completed_at=datetime.now().astimezone().isoformat()))
raise SystemExit(code)
