from pathlib import Path
import os, json, datetime, subprocess, hashlib
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def emit(kind,**v):print(json.dumps(dict(kind=kind,**v)),flush=True)
emit('observation',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid())
for base in (Path('/root'),Path('/root/trifusion-storage/artifacts'),Path('/tmp')):
    found=[dict(path=str(p),directory=p.is_dir(),bytes=p.stat().st_size) for p in sorted(base.iterdir()) if 'coverage' in p.name.lower() or ('audit' in p.name.lower() and 'smooth' in p.name.lower())]
    emit('relevant_paths',base=str(base),paths=found)
for base in (Path('/root'),Path('/root/trifusion-storage/artifacts'),Path('/tmp')):
    for p in sorted(base.iterdir()):
        if p.is_dir() and 'coverage' in p.name.lower() and 'audit' in p.name.lower():
            emit('prior_audit_inventory',directory=str(p),files=[dict(path=str(x),bytes=x.stat().st_size) for x in sorted(p.rglob('*')) if x.is_file()])
ps=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True)
emit('coverage_processes',lines=[l for l in ps.splitlines() if 'coverage' in l and ('python' in l or 'screen' in l)])
for name in ('summary.json','coverage.json','coverage_verification.json'):
    p=root/name
    emit('scientific_artifact',path=str(p),bytes=p.stat().st_size,sha256=sha(p))
emit('run_inventory',files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file()])
