from pathlib import Path
import sys,hashlib,json,datetime
out=Path(__file__).parent
attempt=sys.argv[1]
assert attempt in ('failed_attempt01','attempt02_terminal','source_dependencies')
env={}
exec(Path(r'C:\Users\gb\.codex_tmp\trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),env)
c=env['c'];sftp=c.open_sftp()
remote='/root/trifusion-storage/artifacts/smooth_ap_source_coverage_independent_audit_20260920'
files=[]
if attempt=='source_dependencies':
    inv=json.loads((out/'dependency_inventory.json').read_bytes())
    rows=[json.loads(s) for s in (out/'remote_source_binding.jsonl').read_text().splitlines()]
    current={x['path']:x for r in rows if r['kind']=='source_hashes' for x in r['files']}
    for row in inv['files']:
        name=row['path']
        if row['sha256']!=current[name]['sha256']:
            files.append(('/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'+name,name,current[name]['sha256']))
else:
    if attempt=='attempt02_terminal':remote+='/attempt02'
    for name in ('independent_remote_cpu_verify.py','wrapper.py','launch.json','terminal.json','independent_verify.jsonl','independent_verify.stderr','wrapper.stdout','wrapper.stderr'):
        files.append((remote+'/'+name,name,None))
dest=out/'remote_artifacts'/attempt;dest.mkdir(parents=True,exist_ok=False)
receipts=[]
for src,name,expected in files:
    with sftp.open(src,'rb') as f:data=f.read()
    sha=hashlib.sha256(data).hexdigest()
    if expected:assert sha==expected
    p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    receipts.append(dict(remote=src,local=str(p.relative_to(out)),bytes=len(data),sha256=sha))
sftp.close();c.close()
if attempt!='source_dependencies':
    terminal=json.loads((dest/'terminal.json').read_bytes())
    assert terminal['stdout_sha256']==hashlib.sha256((dest/'independent_verify.jsonl').read_bytes()).hexdigest()
    assert terminal['stderr_sha256']==hashlib.sha256((dest/'independent_verify.stderr').read_bytes()).hexdigest()
    assert terminal['script_sha256']==hashlib.sha256((dest/'independent_remote_cpu_verify.py').read_bytes()).hexdigest()
(dest/'intake.json').write_text(json.dumps(dict(status='BYTE_HASH_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=receipts),indent=2),encoding='utf-8')
print(json.dumps(dict(attempt=attempt,files=len(files),bytes=sum(r['bytes'] for r in receipts))))
