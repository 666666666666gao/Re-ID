
from datetime import datetime
from pathlib import Path
import hashlib,json,csv
import torch
import torch.nn.functional as F
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[{'group': 'original_v6', 'path': 'results/native_fixed_best_20261004_811/RGBNT201_semantic/DIAGNOSIS.json', 'sha256': '92c585b0707c73d3554f2d7b1c8e587f2bf1cdd336249dc43b4ada0df3af0dc9', 'bytes': 717440}, {'group': 'original_v6', 'path': 'results/native_fixed_best_20261004_811/RGBNT201_native/DIAGNOSIS.json', 'sha256': '6a10906af1a934558e742a860be65651dcffc45f22feac17c37f7f8b617361f3', 'bytes': 719577}, {'group': 'original_v6', 'path': 'results/native_fixed_best_20261004_811/MSVR310_semantic/DIAGNOSIS.json', 'sha256': '92c1cbfec345d198318a2a8d7f2d556edcf614f630c3b2f47ab68f5b3056ef5a', 'bytes': 544931}, {'group': 'original_v6', 'path': 'results/native_fixed_best_20261004_811/MSVR310_native/DIAGNOSIS.json', 'sha256': 'bf834ec0b8f60d4fc001b9bc81ac5c145933f189e1fdc301a2dabc7cf8460f7c', 'bytes': 546062}, {'group': 'original_v6', 'path': 'results/native_fixed_best_20261004_811/RGBNT100_semantic/DIAGNOSIS.json', 'sha256': 'ad40a3767ce7c6a244a54ec7e97e042de7a76cc7508765d7fc06587c33e8fa76', 'bytes': 1478141}, {'group': 'original_v6', 'path': 'results/native_fixed_best_20261004_811/RGBNT100_native/DIAGNOSIS.json', 'sha256': 'ca76b6a919b69b2d0f5def35b98f63c475de7e9ba40d2b2b050980d046ef33d3', 'bytes': 1476647}, {'group': 'read_input_detach', 'path': 'results/role_input_detach_fixed_best_20261004_v1/RGBNT201_semantic/DIAGNOSIS.json', 'sha256': '5155cef646e800f93e2f16164fc3bd6ff518df5ba65d2cde44a9ca1228bb1687', 'bytes': 718779}, {'group': 'read_input_detach', 'path': 'results/role_input_detach_fixed_best_20261004_v1/RGBNT201_native/DIAGNOSIS.json', 'sha256': '16a2c95e28f7082212839ea4102abe753c9b3309da9b447bb75b7b18f6064872', 'bytes': 723317}, {'group': 'read_input_detach', 'path': 'results/role_input_detach_fixed_best_20261004_v1/MSVR310_semantic/DIAGNOSIS.json', 'sha256': 'cbe281ebbf13138f937a6476de000fa9a5b0d94d12b37ba918c58fd4033a13e0', 'bytes': 545215}, {'group': 'read_input_detach', 'path': 'results/role_input_detach_fixed_best_20261004_v1/MSVR310_native/DIAGNOSIS.json', 'sha256': 'cad8297b2624c3dc34cc80c99d8a39c4ca027f4c781fb4e6e1b06af90cfec3b2', 'bytes': 546477}, {'group': 'read_input_detach', 'path': 'results/role_input_detach_fixed_best_20261004_v1/RGBNT100_semantic/DIAGNOSIS.json', 'sha256': '581815e0275d7230518ab8baccb10564989713cc540197da768818c0114e1a0a', 'bytes': 1484342}, {'group': 'read_input_detach', 'path': 'results/role_input_detach_fixed_best_20261004_v1/RGBNT100_native/DIAGNOSIS.json', 'sha256': '69a549ea997e3cd4c0b9c0421c0ca83a8c77efb8d342ca38a0bc0b5a586c4553', 'bytes': 1480157}]
scope=root/'refine-logs/global_task_role_v1/SOURCE_SCOPE.json'
assert sha(scope)=='5ac2b38e74cb837d902205482ecfdc32686d3698739300ce94c68b68ab70eb34'
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
assert sha(scope)=='5ac2b38e74cb837d902205482ecfdc32686d3698739300ce94c68b68ab70eb34' and all(sha(root/name)==digest for name,digest in sources.items())
result=dict(status='ALL_TWELVE_CLOSED_FIXED_BEST_CORRECTION_MOMENTS_COMPLETE',
 completed_at=datetime.now().astimezone().isoformat(),models=12,splits=24,input_sha256=bound,rows=rows,
 current_scientific_source_count=330,input_features_unchanged=True,
 boundary='CPU float64 describes saved tensor moments only; original neural inference/metrics not rerun, reselected or modified. Means are descriptive query/gallery statistics and are never applied to deployment. High common energy alone does not prove irrelevance, harmfulness, identity collapse or a unique training failure cause.')
(output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
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
