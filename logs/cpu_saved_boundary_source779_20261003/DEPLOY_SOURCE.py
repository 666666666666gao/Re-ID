from pathlib import Path
from datetime import datetime
import ast,hashlib,json,paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
target=base/'cpu_boundary_deploy779'
assert not target.exists()
publication=json.loads((proof/'five_copy779.json').read_bytes())
review=json.loads((repo/'refine-logs/cpu_saved_boundary_diagnostic_v1/EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'].upper() in ('PASS','WARN') and not review['blocking_issues']
intake=json.loads((base/'cpu_saved_terminal_intake778/INTAKE.json').read_bytes())
sources=dict(intake['source_sha256'])
for name in ('tools/diagnose_cpu_saved_boundaries.py',
             'refine-logs/cpu_saved_boundary_diagnostic_v1/EXPERIMENT_PLAN.md',
             'refine-logs/cpu_saved_boundary_diagnostic_v1/EXPERIMENT_CODE_REVIEW.md'):
    sources[name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
assert len(sources)==304
target.mkdir()
controller=r'''from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
campaign=Path(sys.argv[1]);path=campaign/'JOB.json';job=json.loads(path.read_text())
job['controller_pid']=os.getpid()
with (campaign/'diagnostic.log').open('x') as log:
 child=subprocess.Popen(job['command'],cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['physical_gpu'])),stdout=log,stderr=subprocess.STDOUT)
 job.update(status='RUNNING',child_pid=child.pid);path.write_text(json.dumps(job,indent=2)+'\n')
 code=child.wait()
job.update(status='FAILED' if code else 'COMPLETE',exit_code=code,completed_at=datetime.now().astimezone().isoformat())
path.write_text(json.dumps(job,indent=2)+'\n');sys.exit(code)
'''
ast.parse(controller)
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=={publication['head']!r}
expected={sources!r}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in expected.items())
old=root/'logs/independent_native_evidence_20261003_v4'
state=json.loads((old/'campaign.json').read_text())
assert state['status']=='INITIALIZING' and state['jobs']==[]
assert not (Path('/proc')/str(state['controller_pid'])).exists()
source_initial=old/'initialization/RGBNT201_semantic.json'
assert hashlib.sha256(source_initial.read_bytes()).hexdigest()=={intake['files']['logs/independent_native_evidence_20261003_v4/initialization/RGBNT201_semantic.json']['sha256']!r}
assert shutil.disk_usage(root).free>=2*1024**3
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
available=[int(row.split(',')[0]) for row in memory.splitlines() if int(row.split(',')[0]) in range(4) and int(row.split(',')[1])<500]
assert available
gpu=available[0]
campaign=root/'logs/cpu_saved_boundary_diagnostic_20261003_v1'
assert not campaign.exists()
campaign.mkdir()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/diagnose_cpu_saved_boundaries.py'),'--output',str(campaign/'observations')]
job={{'status':'STARTING','command':command,'physical_gpu':gpu,'source_sha256':expected,'source_head':{publication['head']!r},'started_at':datetime.now().astimezone().isoformat(),'boundary':'One instrumented semantic201 original/CPU-save diagnostic, two backwards, no update/M0/formal/official/weights. Does not promote sealedv4.'}}
(campaign/'JOB.json').write_text(json.dumps(job,indent=2)+'\\n')
(campaign/'CONTROLLER_SOURCE.py').write_text({controller!r})
with (campaign/'controller.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(campaign/'CONTROLLER_SOURCE.py'),str(campaign)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'status':'CPU_BOUNDARY_DIAGNOSTIC_ACTUAL_POPEN','controller_pid':process.pid,'physical_gpu':gpu,'port':2026,'campaign':str(campaign),'command':command,'source_head':{publication['head']!r},'source_sha256':expected,'launched_at':datetime.now().astimezone().isoformat(),'boundary':'Single actual launch only; no result inferred, no unchangedv4restart.'}}))
'''
ast.parse(code)
i,o,e=client.exec_command('/usr/bin/python3 -B -')
i.write(code)
i.channel.shutdown_write()
data,error=o.read(),e.read()
rc=o.channel.recv_exit_status()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':rc,'at':datetime.now().astimezone().isoformat()})+'\n')
client.close()
assert rc==0,error.decode()
value=json.loads(data)
print(json.dumps({k:value[k] for k in ('status','controller_pid','physical_gpu','port','campaign','source_head','launched_at')}))
