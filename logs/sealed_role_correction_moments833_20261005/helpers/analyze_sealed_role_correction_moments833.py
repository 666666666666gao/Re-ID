"""Describe existing fixed-best feature moments, without any new ranking or model call."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'sealed_role_correction_moments833'
assert not packet.exists()
inputs = []
for group, folder in (('original_v6', 'native_fixed_best812_complete'),
                      ('read_input_detach', 'role_input_detach_fixed_best_complete')):
    x = json.loads((base / folder / 'stdout.json').read_bytes())
    assert x['status'] == 'SIX_FIXED_BEST_READ_ONLY_DIAGNOSES_VERIFIED'
    assert x['all_model_parameter_and_buffer_sha_unchanged'] and x['all_original_formal_inputs_sha_unchanged']
    records = {name: record for name, record in x['files'].items() if name.endswith('/DIAGNOSIS.json')}
    assert len(records) == 6
    inputs.extend(dict(group=group, path=name, **record) for name, record in records.items())
scope_path = repo / 'refine-logs/global_task_role_v1/SOURCE_SCOPE.json'
scope_sha = hashlib.sha256(scope_path.read_bytes()).hexdigest()
assert len(json.loads(scope_path.read_bytes())['source_sha256']) == 330
plan = dict(status='PREDEFINED_CPU_STATISTICS_OF_TWELVE_CLOSED_FIXED_BEST_MODELS',
    registered_at=datetime.now().astimezone().isoformat(), inputs=inputs,
    current_scientific_source_scope_sha256=scope_sha,
    observations=['For each query/gallery split: mean vector energy divided by mean per-sample squared norm for g and c.',
                  'RMS energy of centered versus common correction, scaled correction versus global RMS, and mean-direction cosine.',
                  'Check original h=g+gain*c and f=L2(h), artifact bytes/SHA before and after; retain existing scores without recalculation.'],
    hypothesis='In exact arithmetic, training-mode BN classification and raw Euclidean triplet are invariant to a common additive feature offset, while final L2 retrieval is not. This structural observation does not prove current corrections contain harmful offsets.',
    boundary='All twelve inputs previously closed; current sixth training endpoint continues unchanged. No new neural forward, test statistics update, mean subtraction in deployed representation, distance/scoring computation, new checkpoint selection or training decision. Descriptive moments are not causal attribution.')
packet.mkdir()
(packet / 'EXPERIMENT_PLAN.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')

code = '''
from datetime import datetime
from pathlib import Path
import hashlib,json,csv
import torch
import torch.nn.functional as F
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=INPUTS
scope=root/'refine-logs/global_task_role_v1/SOURCE_SCOPE.json'
assert sha(scope)==SCOPE_SHA
sources=json.loads(scope.read_text())['source_sha256']
assert len(sources)==330 and all(sha(root/name)==digest for name,digest in sources.items())
torch.set_num_threads(1)
output=root/'results/sealed_role_correction_moments_20261004_v1'
assert not output.exists()
output.mkdir()
rows=[]
bound={}
def moments(value):
 mean=value.mean(dim=0)
 energy=value.square().sum(dim=1).mean()
 mean_energy=mean.square().sum()
 centered=(value-mean).square().sum(dim=1).mean()
 assert energy>0 and abs(float((mean_energy+centered-energy)/energy))<1e-10
 return dict(samples=len(value),rms_norm=float(energy.sqrt()),mean_norm=float(mean_energy.sqrt()),
  centered_rms_norm=float(centered.sqrt()),common_mean_energy_fraction=float(mean_energy/energy)),mean
for item in inputs:
 path=root/item['path']
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
 report=json.loads(path.read_text())
 assert report['status']=='COMPLETE' and report['model_state_before_sha256']==report['model_state_after_sha256']
 bound[str(path)]=item['sha256']
 gain=report['readout_gain']
 for split in ('query','gallery'):
  name=split+'_features.pt'
  p=path.parent/name
  artifact=report['artifacts'][name]
  assert p.stat().st_size==artifact['bytes'] and sha(p)==artifact['sha256']
  bound[str(p)]=artifact['sha256']
  features=torch.load(p,map_location='cpu',weights_only=False)['features']
  assert set(features)=={'shared_global','correction','raw_fused','fused'}
  assert all(v.device.type=='cpu' and v.ndim==2 and v.shape[1]==1536 and bool(torch.isfinite(v).all()) for v in features.values())
  g,c,h,f=(features[k] for k in ('shared_global','correction','raw_fused','fused'))
  assert torch.allclose(h,g+gain*c,atol=1e-6,rtol=1e-6)
  assert torch.allclose(f,F.normalize(h,dim=1),atol=1e-6,rtol=1e-6)
  gm,mean_g=moments(g.double())
  cm,mean_c=moments(c.double())
  rows.append(dict(group=item['group'],dataset=report['dataset'],variant=report['variant'],split=split,
   selected_epoch=report['selected_epoch'],readout_gain=gain,
   global_moments=gm,correction_moments=cm,
   scaled_correction_rms_to_global_rms=abs(gain)*cm['rms_norm']/gm['rms_norm'],
   scaled_common_correction_to_global_rms=abs(gain)*cm['mean_norm']/gm['rms_norm'],
   scaled_centered_correction_rms_to_global_rms=abs(gain)*cm['centered_rms_norm']/gm['rms_norm'],
   global_and_correction_mean_cosine=float(F.cosine_similarity(mean_g[None],mean_c[None]).item()),
   original_same_model_global_to_fused=report['comparisons']['same_model_global_to_fused']['delta_metrics'],
   original_input_checkpoint_sha256=report['input_checkpoint_sha256']))
assert len(rows)==24
assert {(r['group'],r['dataset'],r['variant'],r['split']) for r in rows}=={
 (g,d,v,s) for g in ('original_v6','read_input_detach') for d in ('RGBNT201','MSVR310','RGBNT100')
 for v in ('semantic','native') for s in ('query','gallery')}
assert all(sha(Path(p))==digest for p,digest in bound.items())
assert sha(scope)==SCOPE_SHA and all(sha(root/name)==digest for name,digest in sources.items())
result=dict(status='ALL_TWELVE_CLOSED_FIXED_BEST_CORRECTION_MOMENTS_COMPLETE',
 completed_at=datetime.now().astimezone().isoformat(),models=12,splits=24,input_sha256=bound,rows=rows,
 current_scientific_source_count=330,input_features_unchanged=True,
 boundary='CPU float64 describes saved tensor moments only; original neural inference/metrics not rerun, reselected or modified. Means are descriptive query/gallery statistics and are never applied to deployment. High common energy alone does not prove irrelevance, harmfulness, identity collapse or a unique training failure cause.')
(output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\\n')
fields=['group','dataset','variant','split','selected_epoch','readout_gain','correction_common_mean_energy_fraction',
 'global_common_mean_energy_fraction','scaled_correction_rms_to_global_rms','scaled_common_correction_to_global_rms',
 'scaled_centered_correction_rms_to_global_rms','original_global_to_fused_map_delta']
with (output/'MOMENTS.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
 for r in rows:
  writer.writerow(dict(**{k:r[k] for k in fields[:6]},
   correction_common_mean_energy_fraction=r['correction_moments']['common_mean_energy_fraction'],
   global_common_mean_energy_fraction=r['global_moments']['common_mean_energy_fraction'],
   **{k:r[k] for k in fields[8:11]},original_global_to_fused_map_delta=r['original_same_model_global_to_fused']['mAP']))
print(json.dumps(dict(summary=result,files={p.name:p.read_text() for p in output.iterdir()})))
'''
code = code.replace('INPUTS', repr(inputs)).replace('SCOPE_SHA', repr(scope_sha))
compile(code, 'remote_describe_sealed_correction_moments.py', 'exec')
(packet / 'remote_describe_sealed_correction_moments.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command("cd /data/gaob/Re-ID/Trifusion && CUDA_VISIBLE_DEVICES='' /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B -")
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(180)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=exit_code,at=datetime.now().astimezone().isoformat())) + '\n', encoding='utf-8')
client.close()
assert exit_code == 0, error.decode()
received = json.loads(data)
for name, content in received['files'].items():
    (packet / name).write_text(content, encoding='utf-8')
print(json.dumps({k:received['summary'][k] for k in ('status','completed_at','models','splits','input_features_unchanged')}))
