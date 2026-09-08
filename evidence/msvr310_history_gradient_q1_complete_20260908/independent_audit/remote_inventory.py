import hashlib, json, os, subprocess, sys
from pathlib import Path
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run=Path('/root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_gradient_v1_seed42_a1b4777')
inputs={}
def load(p):
    p=Path(p); b=p.read_bytes(); inputs[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}; return json.loads(b)
pipeline=load(run/'pipeline.json'); cpu=load(run/'q1_cpu.json')
files={}
paths=[run/x for x in ['t0.json','m0/summary.json','m0_cpu.json','q1/summary.json','q1.log','q1_cpu.log']]
paths += [Path(p) for p in cpu['files']]
for p in dict.fromkeys(paths):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    files[str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
spec=load(repo/'configs/MSVR310/TriFusion-history-gradient-paired-v1.json')
inventory=load(repo/'evidence/msvr310_history_gradient_training_preexecution_20260908/audited_input_hashes.json')
code={}
for name in inventory:
    p=repo/name
    if not p.is_file(): continue
    b=p.read_bytes()
    result=subprocess.run(['git','-C',str(repo),'show',pipeline['code_commit']+':'+name],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    code[name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'lf_sha256':hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest(),'execution_commit_exists':result.returncode==0,'execution_commit_raw_sha256':hashlib.sha256(result.stdout).hexdigest() if result.returncode==0 else None,'execution_commit_raw_match':result.returncode==0 and result.stdout==b,'execution_commit_match':result.returncode==0 and result.stdout.replace(b'\r\n',b'\n')==b.replace(b'\r\n',b'\n')}
print(json.dumps({'identity':{'uid':os.getuid(),'hostname':os.uname().nodename,'cwd':str(Path.cwd()),'python':sys.version},'head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'status':subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True),'pipeline':pipeline,'process_presence':{str(pid):Path('/proc/'+str(pid)).exists() for pid in [pipeline['wrapper_pid']]+[s['original_pid'] for s in pipeline['stages']]},'files':files,'project_files':code,'read_hashes':inputs},indent=2))
