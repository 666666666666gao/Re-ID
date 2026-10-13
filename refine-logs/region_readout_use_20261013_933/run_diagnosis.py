"""Admit and run the two frozen-best readout diagnoses once, sequentially."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (w/'LAUNCH_EXIT.json').exists()
old_local=w.parent/'gcf930/results/INPUT_SEAL.json'
old=json.loads(old_local.read_text())
source=r/'tools/diagnose_region_readout_use.py'
plan=r/'refine-logs/region_readout_use_20261013_933/EXPERIMENT_PLAN.md'
audit=json.loads((w/'SOURCE_AUDIT.json').read_text())
assert audit['source_readiness']=='PASS' and audit['source_blockers']==[]
assert all(sha(Path(p))==h.removeprefix('sha256:') for p,h in audit['audited_input_hashes'].items())
artifacts={}
cached='/data/gaob/Re-ID/Trifusion/logs/global_conditioned_evidence_best930'
for exposure in ('gradient','random'):
    p=w.parent/'gcf930/results'/exposure/'DIAGNOSIS.json'
    d=json.loads(p.read_text())
    assert d['status']=='COMPLETE' and d['selected_epoch']==8
    artifacts[cached+'/'+exposure+'/DIAGNOSIS.json']=sha(p)
    for name in ('query_features.pt','gallery_features.pt'):
        artifacts[cached+'/'+exposure+'/'+name]=d['artifacts'][name]['sha256']
request={'source':source.read_text(encoding='utf-8'),'plan':plan.read_text(encoding='utf-8'),
    'source_sha256':{'tools/diagnose_region_readout_use.py':sha(source),
        'refine-logs/region_readout_use_20261013_933/EXPERIMENT_PLAN.md':sha(plan)},
    'prior_seal':'/data/gaob/Re-ID/Trifusion/logs/global_conditioned_evidence_best930/INPUT_SEAL.json',
    'prior_seal_sha256':sha(old_local),'artifact_sha256':artifacts,
    'cached_features_root':cached,'storage_required_bytes':2684354560,
    'source_audit_sha256':sha(w/'SOURCE_AUDIT.json'),'schema':'trifusion-fixed-region-readout-use-v1'}
(w/'LAUNCH_REQUEST.json').write_text(json.dumps(request,indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
from datetime import datetime
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path(request['prior_seal']))==request['prior_seal_sha256']
prior=json.loads(Path(request['prior_seal']).read_text())
assert all(sha(root/p)==h for p,h in prior['source_sha256'].items())
assert all(sha(Path(p))==h for p,h in prior['artifact_sha256'].items())
assert all(sha(Path(p))==h for p,h in request['artifact_sha256'].items())
assert json.loads(Path(prior['controller_exit']).read_text())['exit_code']==0
assert json.loads(Path(prior['campaign']).read_text())['status']=='COMPLETE_TWO_FULL50_FIRST_STRICT'
free=shutil.disk_usage(root).free
assert free>=request['storage_required_bytes'],free
gpu=subprocess.run(['nvidia-smi','-i','0,1','--query-gpu=index,memory.used,memory.total',
    '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
gpus=[tuple(int(v.strip()) for v in line.split(',')) for line in gpu.strip().splitlines()]
assert [v[0] for v in gpus]==[0,1] and all(v[1]<500 for v in gpus),gpus
admin=root/'logs/region_readout_use933';assert not admin.exists();admin.mkdir()
for key,name in (('source','tools/diagnose_region_readout_use.py'),
                 ('plan','refine-logs/region_readout_use_20261013_933/EXPERIMENT_PLAN.md')):
    data=request.pop(key).encode();assert hashlib.sha256(data).hexdigest()==request['source_sha256'][name]
    p=root/name;assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
(admin/'INPUT_SEAL.json').write_text(json.dumps(request,indent=2)+'\n')
(admin/'ADMISSION.json').write_text(json.dumps({'observed_at':datetime.now().astimezone().isoformat(),
    'free_bytes':free,'gpu_memory_rows':gpus,'environment':'reuse installed tri_reid; no install/rebuild',
    'training_started':False},indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
exits=[]
started=time.perf_counter()
for exposure in ('gradient','random'):
    assert shutil.disk_usage(root).free>=2415919104  # 2GiB reserve + one 256MiB arm budget.
    cmd=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',
        str(root/'tools/diagnose_region_readout_use.py'),'--seal',str(admin/'INPUT_SEAL.json'),
        '--output-dir',str(admin/exposure),'--exposure',exposure]
    began=datetime.now().astimezone().isoformat()
    with (admin/(exposure+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        exit_code=child.wait()
    item={'exposure':exposure,'child_pid':child.pid,'command':cmd,'started_at':began,
          'completed_at':datetime.now().astimezone().isoformat(),'exit_code':exit_code,'optimizer_updates':0}
    exits.append(item)
    (admin/(exposure+'_EXIT.json')).write_text(json.dumps(item,indent=2)+'\n')
    assert exit_code==0,(item,(admin/(exposure+'.log')).read_text())
assert all(sha(root/p)==h for p,h in prior['source_sha256'].items())
assert all(sha(Path(p))==h for p,h in prior['artifact_sha256'].items())
assert all(sha(Path(p))==h for p,h in request['artifact_sha256'].items())
record={'status':'TWO_FROZEN_BEST_DIAGNOSES_COMPLETE','completed_at':datetime.now().astimezone().isoformat(),
    'exits':exits,'elapsed_seconds':time.perf_counter()-started,'optimizer_updates':0,'new_training':False}
(admin/'EXIT.json').write_text(json.dumps(record,indent=2)+'\n')
result={'exit':record,'seal':request,'admission':json.loads((admin/'ADMISSION.json').read_text()),
    'diagnoses':{e:json.loads((admin/e/'DIAGNOSIS.json').read_text()) for e in ('gradient','random')}}
print(json.dumps(result))
'''
opts=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*opts,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],
    input=code,text=True,capture_output=True)
(w/'LAUNCH.stdout').write_text(run.stdout,encoding='utf-8')
(w/'LAUNCH.stderr').write_text(run.stderr,encoding='utf-8')
(w/'LAUNCH_EXIT.json').write_text(json.dumps({'exit_code':run.returncode,
    'at':datetime.now().astimezone().isoformat()},indent=2)+'\n')
assert run.returncode==0,run.stderr
result=json.loads(run.stdout)
(w/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['exit']['status'],'exits':result['exit']['exits'],
    'metrics':{e:{name:score['metrics'] for name,score in d['scores'].items()}
               for e,d in result['diagnoses'].items()}}))
