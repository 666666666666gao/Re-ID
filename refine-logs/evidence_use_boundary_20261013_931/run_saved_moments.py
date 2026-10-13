"""One CPU-only saved-feature intake; no GPU/model/optimizer invocation."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime

w=Path(__file__).resolve().parent
r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
gcf=w.parent/'gcf930'
assert not (w/'COMMON_COMPONENT_EXIT.json').exists()
source=r/'tools/analyze_evidence_common_component.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];artifacts={}
for exposure in ('gradient','random'):
    p=gcf/'results'/exposure/'DIAGNOSIS.json'
    diagnosis=json.loads(p.read_text())
    assert diagnosis['status']=='COMPLETE' and diagnosis['selected_epoch']==8
    remote='/data/gaob/Re-ID/Trifusion/logs/global_conditioned_evidence_best930/'+exposure
    artifacts[remote+'/DIAGNOSIS.json']=sha(p)
    for split in ('query','gallery'):
        name=split+'_features.pt'
        artifacts[remote+'/'+name]=diagnosis['artifacts'][name]['sha256']
        rows.append({'exposure':exposure,'split':split,'path':remote+'/'+name})
request={'source':source.read_text(encoding='utf-8'),'source_sha256':sha(source),
         'artifact_sha256':artifacts,'rows':rows}
code='request='+repr(request)+'\n'+r'''
import hashlib,json,os,subprocess
from pathlib import Path
from datetime import datetime
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(Path(p))==h for p,h in request['artifact_sha256'].items())
admin=root/'logs/evidence_common_component931';assert not admin.exists();admin.mkdir()
source=root/'tools/analyze_evidence_common_component.py'
data=request.pop('source').encode();assert hashlib.sha256(data).hexdigest()==request['source_sha256']
assert not source.exists();source.write_bytes(data)
(admin/'INPUT_SEAL.json').write_text(json.dumps(request,indent=2)+'\n')
cmd=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(source),
     '--seal',str(admin/'INPUT_SEAL.json'),'--output',str(admin/'STATISTICS.json')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
began=datetime.now().astimezone().isoformat()
with (admin/'CPU.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 code=child.wait()
exit_record={'command':cmd,'child_pid':child.pid,'started_at':began,
 'completed_at':datetime.now().astimezone().isoformat(),'exit_code':code,'neural_forward_calls':0}
(admin/'EXIT.json').write_text(json.dumps(exit_record,indent=2)+'\n')
assert code==0,(code,(admin/'CPU.log').read_text())
assert all(sha(Path(p))==h for p,h in request['artifact_sha256'].items())
print(json.dumps({'exit':exit_record,'seal':request,
 'statistics':json.loads((admin/'STATISTICS.json').read_text()),
 'statistics_sha256':sha(admin/'STATISTICS.json')}))
'''
opts=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*opts,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],
                    input=code,text=True,capture_output=True)
(w/'COMMON_COMPONENT.stderr').write_text(run.stderr,encoding='utf-8')
(w/'COMMON_COMPONENT.stdout').write_text(run.stdout,encoding='utf-8')
(w/'COMMON_COMPONENT_EXIT.json').write_text(json.dumps({'exit_code':run.returncode,'at':datetime.now().astimezone().isoformat()},indent=2)+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout)
(w/'COMMON_COMPONENT_STATISTICS.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'exit_code':value['exit']['exit_code'],'child_pid':value['exit']['child_pid'],
 'neural_forward_calls':0,'statistics':[{k:row[k] for k in ('exposure','split','global','correction')} for row in value['statistics']['rows']]}))
