"""Seal and launch one finite two-best diagnosis after exact source review."""
import hashlib
import json
from pathlib import Path
import subprocess

w = Path(__file__).resolve().parent
evidence = w.parent
r = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
review = json.loads((w / 'review/SOURCE_AUDIT.json').read_text())
assert review['source_ready_for_fixed_best_diagnosis'] and not review['unresolved_blocking_source_defects']
logical_w = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/gcf930')
sources = {'tools/diagnose_global_conditioned_evidence_best.py': r / 'tools/diagnose_global_conditioned_evidence_best.py',
           'logs/global_conditioned_evidence_best930/controller.py': logical_w / 'controller.py',
           'refine-logs/global_conditioned_evidence_best930/EXPERIMENT_PLAN.md': r / 'refine-logs/global_conditioned_evidence_best930/EXPERIMENT_PLAN.md'}
for p in sources.values():
    assert sha(p) == review['audited_source_sha256'][p.as_posix()]
previous = json.loads((evidence / 'EVIDENCE_TERMINAL_FINAL_929.json').read_text())
first = json.loads((evidence / 'EVIDENCE_FIRST_FINAL_928.json').read_text())
assert previous['controller_and_worker_absent'] and previous['new_neural_calls'] == 0
assert not (w / 'LAUNCH.json').exists()
request = {'source_sha256': previous['source_sha256'], 'new_sources': {name: p.read_text(encoding='utf-8') for name, p in sources.items()},
           'new_source_sha256': {name: sha(p) for name, p in sources.items()},
           'original_file_sha256': {row['remote_path']: row['sha256'] for c in (first['collection'], previous['collection']) for row in c['files'].values()},
           'audit_sha256': sha(w / 'review/SOURCE_AUDIT.json')}
opts = ['-o', 'ProxyCommand=none', '-o', 'ProxyJump=none', '-o', 'BatchMode=yes', '-i', 'C:/Users/gb/.ssh/id_ed25519']
code = 'request=' + repr(request) + '\n' + r'''
import hashlib,json
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/n)==h for n,h in request['source_sha256'].items())
assert all(sha(Path(n))==h for n,h in request['original_file_sha256'].items())
old=root/'logs/global_conditioned_evidence_full_20261013_926'
full=json.loads((old/'INPUTS.json').read_text())
assert full['runtime_source_sha256']==request['source_sha256']
assert all(sha(root/n)==row['sha256'] for n,row in full['m0_file_sha256'].items())
state=json.loads((old/'campaign.json').read_text())
assert state['status']=='COMPLETE_TWO_FULL50_FIRST_STRICT'
assert json.loads((old/'CONTROLLER_EXIT.json').read_text())['exit_code']==0
assert not Path('/proc/3190961').exists() and not Path('/proc/3190963').exists()
admin=root/'logs/global_conditioned_evidence_best930'
assert not admin.exists()
control_path=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(control_path)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
control=json.loads(control_path.read_text())
global_row=next(row for row in control['rows'] if row['dataset']=='RGBNT201' and row['variant']=='global_only')
global_distance=Path(global_row['run_dir'])/'official_distances.pt'
assert sha(global_distance)==global_row['distance_sha256']==control['artifact_sha256'][str(global_distance)]
artifact=dict(request['original_file_sha256'])
artifact.update({str(root/n):row['sha256'] for n,row in full['m0_file_sha256'].items()})
artifact[str(control_path)]=sha(control_path)
artifact[str(global_distance)]=sha(global_distance)
rows=[]
for accepted in state['accepted_endpoints']:
 exposure=accepted['exposure'];run=Path(accepted['output_dir'])
 receipt=json.loads((run/'official_metrics.json').read_text())
 training=json.loads((run/'training.json').read_text())
 init=root/f'logs/global_conditioned_evidence_m0_20261013_925/initialization/RGBNT201_{exposure}.json'
 assert sha(run/'official_metrics.json')==accepted['official_metrics_sha256']
 assert sha(run/'best_map.pth')==receipt['checkpoint_sha256']
 artifact[str(run/'best_map.pth')]=receipt['checkpoint_sha256']
 rows.append({'exposure':exposure,'run_dir':str(run),'initialization':str(init),
              'initializer':training['initializer'],'metrics':receipt['metrics'],
              'selected_epoch':receipt['selected_epoch'],'checkpoint_sha256':receipt['checkpoint_sha256'],
              'distance_sha256':receipt['distance_sha256']})
protocol=root/'logs/training_feature_scale_protocols_20261002/RGBNT201.json'
clip=root/'pertrained-model/ViT-B-16.pt'
assert sha(protocol)==full['protocol_sha256'] and sha(clip)==full['public_clip_sha256']
artifact[str(protocol)]=full['protocol_sha256'];artifact[str(clip)]=full['public_clip_sha256']
admin.mkdir()
for n,text in request['new_sources'].items():
 p=root/n;assert p.resolve().is_relative_to(root.resolve())
 assert hashlib.sha256(text.encode()).hexdigest()==request['new_source_sha256'][n]
 if p.exists():assert sha(p)==request['new_source_sha256'][n]
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(text.encode())
assert all(sha(root/n)==h for n,h in request['source_sha256'].items())
seal={'schema':'trifusion-global-conditioned-best-decomposition-v1','registered_at':datetime.now().astimezone().isoformat(),
      'source_sha256':dict(request['source_sha256'],**request['new_source_sha256']),
      'artifact_sha256':artifact,'rows':rows,'campaign':str(old/'campaign.json'),
      'controller_exit':str(old/'CONTROLLER_EXIT.json'),'protocol':str(protocol),
      'clip_weight':str(clip),'signal_source':str(root/'comparators/Signal-cd1b0a6'),
      'independent_global_distance':str(global_distance),'independent_global_metrics':global_row['metrics'],
      'source_review_sha256':request['audit_sha256'],'optimizer_updates':0,'physical_gpus':[0,1]}
(admin/'INPUT_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
print(json.dumps({'status':'SEALED_NEW_FIXED_BEST_DIAGNOSIS_NO_NN','seal_sha256':sha(admin/'INPUT_SEAL.json'),
                  'sources':len(seal['source_sha256']),'artifacts':len(artifact),'admin':str(admin)}))
'''
prep = subprocess.run(['ssh', *opts, '-p', '2026', 'gaob@172.19.12.138', '/usr/bin/python3', '-'], input=code, text=True, capture_output=True)
for suffix, data in [('stdout', prep.stdout), ('stderr', prep.stderr)]:
    (w / ('PREPARE.' + suffix)).write_text(data, encoding='utf-8')
(w / 'PREPARE_EXIT.json').write_text(json.dumps({'exit_code': prep.returncode}) + '\n')
assert prep.returncode == 0, prep.stderr
print(prep.stdout, end='', flush=True)
launch = subprocess.run(['ssh', *opts, '-p', '2026', 'gaob@172.19.12.138', '/usr/bin/python3',
    '/data/gaob/Re-ID/Trifusion/logs/global_conditioned_evidence_best930/controller.py'], capture_output=True, text=True)
for suffix, data in [('stdout', launch.stdout), ('stderr', launch.stderr)]:
    (w / ('LAUNCH.' + suffix)).write_text(data, encoding='utf-8')
(w / 'LAUNCH_TRANSPORT_EXIT.json').write_text(json.dumps({'exit_code': launch.returncode}) + '\n')
assert launch.returncode == 0, launch.stderr
(w / 'LAUNCH.json').write_text(launch.stdout, encoding='utf-8')
print(launch.stdout, end='', flush=True)
