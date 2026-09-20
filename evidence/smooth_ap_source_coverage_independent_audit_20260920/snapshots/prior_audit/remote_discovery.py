from pathlib import Path
import os, json, hashlib, subprocess, datetime
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def emit(kind,**values):print(json.dumps(dict(kind=kind,**values),ensure_ascii=False),flush=True)
emit('environment',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cwd=os.getcwd(),pid=os.getpid(),user=subprocess.check_output(['whoami'],text=True).strip(),hostname=subprocess.check_output(['hostname'],text=True).strip(),python=subprocess.check_output(['/root/miniconda3/envs/tri_reid/bin/python','-V'],text=True).strip())
spec=json.loads((repo/'configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json').read_bytes())
qpath=Path(spec['q1_summary'])
emit('q1_input_metadata',path=str(qpath),bytes=qpath.stat().st_size,sha256=sha(qpath),expected=spec['q1_summary_sha256'])
q=json.loads(qpath.read_bytes())
for f in q['folds']:
 for end in ('control','smooth_ap'):
  row=f['endpoints'][end]
  mp=Path(row['checkpoint']).parent/'memory_steps.jsonl'
  emit('endpoint_inputs',fold=f['fold'],endpoint=end,checkpoint=row['checkpoint'],checkpoint_bytes=Path(row['checkpoint']).stat().st_size,checkpoint_sha256=sha(row['checkpoint']),expected_checkpoint_sha256=row['checkpoint_sha256'],initialization=row['initialization'],final_state_sha256=row['training']['final_state_sha256'],memory_path=str(mp),memory_bytes=mp.stat().st_size,memory_sha256=sha(mp),memory_expected=row['training']['audit_files']['memory_steps.jsonl'])
for name in ('summary.json','coverage.json','coverage_verification.json'):
 p=root/name;o=json.loads(p.read_bytes())
 emit('diagnostic_file',path=str(p),sha256=sha(p),bytes=p.stat().st_size,keys=list(o),small_fields={k:v for k,v in o.items() if not isinstance(v,(list,dict))})
files=[p for p in root.rglob('*') if p.is_file()]
emit('diagnostic_inventory',files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in files],total_bytes=sum(p.stat().st_size for p in files))
emit('git',head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True),execution_commit=subprocess.check_output(['git','-C',str(repo),'show','-s','--format=%H %cI %s','f7a0590'],text=True))

