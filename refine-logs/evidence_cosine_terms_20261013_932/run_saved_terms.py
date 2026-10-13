"""One CPU saved-vector relation analysis; no neural or training launch."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
gcf=w.parent/'gcf930'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (w/'CPU_TERMS_EXIT.json').exists()
source=r/'tools/analyze_evidence_cosine_terms.py'
old=json.loads((gcf/'results/INPUT_SEAL.json').read_text())
protocol=old['protocol']
artifacts={protocol:old['artifact_sha256'][protocol]}
rows=[]
for exposure in ('gradient','random'):
    p=gcf/'results'/exposure/'DIAGNOSIS.json'
    diagnosis=json.loads(p.read_text())
    assert diagnosis['status']=='COMPLETE' and diagnosis['selected_epoch']==8
    root='/data/gaob/Re-ID/Trifusion/logs/global_conditioned_evidence_best930/'+exposure
    artifacts[root+'/DIAGNOSIS.json']=sha(p)
    for name in ('query_features.pt','gallery_features.pt','diagnostic_distances.pt'):
        artifacts[root+'/'+name]=diagnosis['artifacts'][name]['sha256']
    rows.append({'exposure':exposure,'diagnosis':root+'/DIAGNOSIS.json',
        'query':root+'/query_features.pt','gallery':root+'/gallery_features.pt',
        'distances':root+'/diagnostic_distances.pt'})
request={'source':source.read_text(encoding='utf-8'),'source_sha256':sha(source),
         'protocol':protocol,'artifact_sha256':artifacts,'rows':rows}
(w/'REQUEST.json').write_text(json.dumps(request,indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
from datetime import datetime
import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(Path(p))==h for p,h in request['artifact_sha256'].items())
campaign=root/'logs/global_conditioned_evidence_full_20261013_926'
assert json.loads((campaign/'CONTROLLER_EXIT.json').read_text())['exit_code']==0
assert json.loads((campaign/'campaign.json').read_text())['status']=='COMPLETE_TWO_FULL50_FIRST_STRICT'
admin=root/'logs/evidence_cosine_terms932';assert not admin.exists();admin.mkdir()
source=root/'tools/analyze_evidence_cosine_terms.py'
payload=request.pop('source').encode();assert hashlib.sha256(payload).hexdigest()==request['source_sha256']
assert not source.exists();source.write_bytes(payload)
(admin/'INPUT_SEAL.json').write_text(json.dumps(request,indent=2)+'\n')
cmd=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(source),
     '--seal',str(admin/'INPUT_SEAL.json'),'--output',str(admin/'TERMS.json')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
started=datetime.now().astimezone().isoformat()
with (admin/'CPU.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    status=child.wait()
record={'command':cmd,'child_pid':child.pid,'started_at':started,
    'completed_at':datetime.now().astimezone().isoformat(),'exit_code':status,
    'neural_forward_calls':0,'optimizer_updates':0,'retrieval_rankings_computed':0}
(admin/'EXIT.json').write_text(json.dumps(record,indent=2)+'\n')
assert status==0,(status,(admin/'CPU.log').read_text())
assert all(sha(Path(p))==h for p,h in request['artifact_sha256'].items())
print(json.dumps({'exit':record,'seal':request,'terms':json.loads((admin/'TERMS.json').read_text()),
    'terms_sha256':sha(admin/'TERMS.json'),'seal_sha256':sha(admin/'INPUT_SEAL.json')}))
'''
opts=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*opts,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],
    input=code,text=True,capture_output=True)
(w/'CPU_TERMS.stdout').write_text(run.stdout,encoding='utf-8')
(w/'CPU_TERMS.stderr').write_text(run.stderr,encoding='utf-8')
(w/'CPU_TERMS_EXIT.json').write_text(json.dumps({'exit_code':run.returncode,
    'at':datetime.now().astimezone().isoformat()},indent=2)+'\n')
assert run.returncode==0,run.stderr
result=json.loads(run.stdout)
(w/'CPU_TERMS_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'COMPLETE','exit':result['exit'],'rows':[
    {k:row[k] for k in ('exposure','counts','closure','equal_query_weighted_margins')}
    for row in result['terms']['rows']]}))
