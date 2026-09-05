from pathlib import Path
from datetime import datetime
import hashlib, json, subprocess
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
directory=Path('/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v22_camera_negative_seed42_5ae096b')
assert Path(str(directory)+'.exit').read_text().strip()=='0'
summary=json.loads((directory/'run_summary.json').read_bytes())
assert summary['status'] in ('Q1_PASS','Q1_FAIL') and len(summary['folds'])==3
assert summary['repository_commit']=='5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36'
output=directory/'terminal_file_verification.json'
assert not output.exists()
result=subprocess.run(['/root/miniconda3/envs/tri_reid/bin/python',str(repo/'tools/verify_v22_terminal_files.py'),
                       '--repo',str(repo),'--run-dir',str(directory),'--output',str(output)],
                      cwd=repo,capture_output=True,text=True,check=True)
verification=json.loads(output.read_bytes())
assert verification['verified']
pairs=[
(directory/'run_summary.json','trifusion_v22_q1_seed42_5ae096b.json'),
(Path(str(directory)+'.log'),'trifusion_v22_complete_run_20260905.log'),
(output,'trifusion_v22_terminal_file_verification_20260905.json')]
for fold in summary['folds']:
    for endpoint in ('batch_hard_residual','camera_negative_residual'):
        stem=f"fold_{fold['fold']}_{endpoint}_receipt.json"
        pairs.append((directory/stem,'trifusion_v22_'+stem))
files=[]
for source,target in pairs:
    raw=source.read_bytes()
    files.append({'remote':str(source),'local_evidence_name':target,'bytes':len(raw),
                  'sha256':hashlib.sha256(raw).hexdigest()})
record={'captured_at':datetime.now().astimezone().isoformat(),'status':summary['status'],'exit_code':0,
        'execution_commit':summary['repository_commit'],'file_verification_stdout':result.stdout.strip(),
        'files':files,'new_optimizer_steps':0,'new_retrieval_evaluations':0}
path=directory/'terminal_transfer_manifest.json'
assert not path.exists()
path.write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps({'manifest_path':str(path),**record}))
