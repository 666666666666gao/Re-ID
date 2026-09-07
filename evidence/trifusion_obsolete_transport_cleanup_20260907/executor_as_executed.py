from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
base=Path('/root/autodl-tmp/trifusion-v2/transport').resolve()
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
plan_path=base/'obsolete_bundle_cleanup_plan_20260907.json'
assert hashlib.sha256(plan_path.read_bytes()).hexdigest()=='e9bc085253232ba89a34506abb1f8b6690470ac636abdb749f1e11284c1d9127'
plan=json.loads(plan_path.read_bytes())
assert plan['status']=='VERIFIED_OBSOLETE_IMPORTED_BUNDLES_ONLY'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==plan['repo_head']
assert len(plan['files'])==182 and sum(x['bytes'] for x in plan['files'])==637229963
for row in plan['files']:
 p=Path(row['path'])
 assert p.resolve().parent==base and p.parent==base and not p.is_symlink()
 assert p.suffix=='.bundle' and not p.name.startswith('msvr_freshness')
 st=p.stat()
 assert st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
for pid in [185622,186000]:
 cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\x00',b' ')
 assert b'msvr_freshness' in cmd
receipt_path=base/'obsolete_bundle_cleanup_receipt_20260907.json'
receipt=dict(status='DELETION_STARTED',started_at=datetime.now().astimezone().isoformat(),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),files=plan['files'],free_before=shutil.disk_usage(base).free,source_original_pid=186000,wrapper_original_pid=185622,weight_or_evidence_deletions=0)
with receipt_path.open('x') as f: f.write(json.dumps(receipt,indent=2)+'\n')
for row in plan['files']: Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in plan['files'])
assert all(Path(p).is_file() for p in plan['retained_current_freshness_bundles'])
check=subprocess.run(['git','fsck','--connectivity-only','--no-dangling'],cwd=repo,text=True,capture_output=True)
assert check.returncode==0,check.stderr
for pid in [185622,186000]: assert (Path('/proc')/str(pid)/'cmdline').is_file()
receipt.update(status='COMPLETE_VERIFIED_OBSOLETE_BUNDLE_CLEANUP',completed_at=datetime.now().astimezone().isoformat(),removed_files=len(plan['files']),removed_bytes=sum(x['bytes'] for x in plan['files']),all_targets_absent=True,current_freshness_bundles_retained=True,git_reachable_connectivity_after=True,free_after=shutil.disk_usage(base).free,original_processes_present_after=True)
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
