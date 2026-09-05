from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess
base=Path('/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v22_initialization_full_gallery_824fcfd')
output=Path(str(base)+'.json');log=Path(str(base)+'.log');exit_file=Path(str(base)+'.exit')
proc=Path('/proc/42325/cmdline')
live=proc.exists() and b'tools/diagnose_v22_initialization_full_gallery.py' in proc.read_bytes().split(b'\0')
raw=output.read_bytes();d=json.loads(raw)
status={'observed_at':datetime.now().astimezone().isoformat(),'original_pid':42325,'original_process_live':live,
        'status':d['status'],'completed_folds':len(d['folds']),'elapsed_seconds':d['elapsed_seconds'],
        'exit_code':exit_file.read_text().strip() if exit_file.exists() else None,
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()}
if not live:
    assert status['exit_code']=='0' and d['status']=='COMPLETE_READONLY_DIAGNOSTIC'
    repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    files=[]
    expected={Path(p):h for p,h in d['source_file_sha256'].items()}
    expected[repo/'tools/diagnose_v22_initialization_full_gallery.py']=d['script_sha256']
    expected[repo/'refine-logs/v22/INITIALIZATION_FULL_GALLERY_DIAGNOSTIC_PLAN.md']=d['plan_sha256']
    for p,wanted in expected.items():
        if not p.is_absolute():p=repo/p
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024**2),b''):digest.update(chunk)
        actual=digest.hexdigest();assert actual==wanted,str(p)
        files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':actual,'expected_sha256':wanted,'matches':True})
    status.update({'source_file_verification':files,'summary_path':str(output),'summary_bytes':len(raw),
                   'summary_sha256':hashlib.sha256(raw).hexdigest(),'log_path':str(log),
                   'log_bytes':log.stat().st_size,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
                   'initialization_aggregate':d['initialization_aggregate'],
                   'terminal_minus_initial_mAP':d['terminal_minus_initial_mAP'],
                   'new_verifier_model_loads':0,'new_verifier_optimizer_steps':0})
record=Path(str(base)+'.observation_'+datetime.now().strftime('%H%M%S')+'.json')
assert not record.exists();record.write_bytes((json.dumps(status,indent=2)+'\n').encode())
print(json.dumps({'observation_path':str(record),**status}))
