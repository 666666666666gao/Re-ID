from pathlib import Path
import json

t=Path('C:/Users/gb/.codex_tmp')
plan=json.loads((t/'trifusion_transport_retention_20260909.json').read_bytes())
assert len(plan['files'])==27 and all(x['eligible'] and x['local_copy_identical'] for x in plan['files'])
script='''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
plan=json.loads(PLAN)
root=Path('/root/autodl-tmp/trifusion-v2');repo=root/'TriFusion-ReID';folder=root/'transport'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==plan['head']
for x in plan['files']:
 p=Path(x['path'])
 assert p.resolve().parent==folder.resolve() and not p.is_symlink() and p.suffix=='.bundle'
 assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 for ref in x['refs']:
  subprocess.run(['git','cat-file','-e',ref['oid']+'^{commit}'],cwd=repo,check=True,capture_output=True)
  subprocess.run(['git','merge-base','--is-ancestor',ref['oid'],plan['head']],cwd=repo,check=True,capture_output=True)
out=root/'artifacts/transport_cleanup_20260909';out.mkdir()
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\\n')
receipt=dict(status='DELETING_VERIFIED_REDUNDANT_TRANSPORT',started_at=datetime.now().astimezone().isoformat(),head=plan['head'],free_bytes_before=shutil.disk_usage(root).free,deleted=[],weight_files_deleted=0)
for x in plan['files']:
 p=Path(x['path']);p.unlink();assert not p.exists();receipt['deleted'].append(x)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
receipt.update(status='COMPLETE_27_REDUNDANT_TRANSPORT_BUNDLES_DELETED',ended_at=datetime.now().astimezone().isoformat(),logical_bytes_deleted=sum(x['bytes'] for x in receipt['deleted']),free_bytes_after=shutil.disk_usage(root).free)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''.replace('PLAN',repr(json.dumps(plan)),1)
env={};exec((t/'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),env)
c=env['c'];i,o,e=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -B -',timeout=60)
i.write(script);i.channel.shutdown_write();raw=o.read();err=e.read();status=o.channel.recv_exit_status();c.close()
assert status==0,err.decode()
out=t/'trifusion_transport_cleanup_receipt_20260909.json';assert not out.exists();out.write_bytes(raw)
r=json.loads(raw);print(json.dumps({k:v for k,v in r.items() if k!='deleted'}))
