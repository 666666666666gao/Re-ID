"""Exact D archives before removal of only closed remote duplicates."""
import hashlib,json,shutil,subprocess
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (w/'CLOSED_ARCHIVE_REQUEST_951.json').exists()
assert json.loads((w/'COMPLETE_CLOSURE_950.json').read_bytes())['status']=='NATIVE_PATCH_GIT_DESKTOP_26_EXACT'
assert json.loads((w/'REGION_COMPARISON_COLLECTED_949.json').read_bytes())['exit_code']==0
root='/data/gaob/Re-ID/Trifusion'
inventory=[]
for mode in ('joint_delta','add_delta','joint_absolute'):
    diagnosis=json.loads((w/f'diagnosis948/results/joint_spectral_diagnosis948/{mode}/DIAGNOSIS.json').read_bytes())
    for name in ('query_features.pt','gallery_features.pt'):
        inventory.append(dict(path=f'{root}/results/joint_spectral_diagnosis948/{mode}/{name}',
            archive_name=mode+'_'+name,**diagnosis['artifacts'][name]))
report=json.loads((w/'REPORT_COLLECTED_948.json').read_bytes())
for mode in ('add_delta','joint_absolute'):
    row=next(row for row in report['weight_inventory'] if row['path']==f'{root}/trained-model/joint_spectral_training943_{mode}_full_semantic_RGBNT201/best_map.pth')
    inventory.append(dict(row,archive_name=mode+'_best_map.pth'))
archive=(w/'closed_inputs951').resolve();assert archive.drive=='D:' and not archive.exists()
assert archive.is_relative_to(w.resolve()) and shutil.disk_usage(archive.parent).free>sum(row['bytes'] for row in inventory)+1024**3
archive.mkdir();request=dict(inventory=inventory,archive=str(archive),archives={},
    canonical_source_sha256=json.loads((w/'REGISTRATION_FINAL.json').read_bytes())['source_sha256'])
(w/'CLOSED_ARCHIVE_REQUEST_951.json').write_text(json.dumps(request,indent=2)+'\n')
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
for row in inventory:
    target=archive/row['archive_name']
    run=subprocess.run(['scp','-q',*options,'-P','2026','gaob@172.19.12.138:'+row['path'],str(target)],capture_output=True)
    (w/(row['archive_name']+'_COPY_951.json')).write_text(json.dumps(dict(exit_code=run.returncode,stderr=run.stderr.decode()))+'\n')
    assert run.returncode==0,run.stderr
    assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
    request['archives'][row['path']]=dict(path=str(target),bytes=row['bytes'],sha256=row['sha256'])
(w/'CLOSED_ARCHIVES_VERIFIED_951.json').write_text(json.dumps(request,indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
import hashlib,json,shutil
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((root/'logs/joint_spectral_training943/campaign.json').read_text())['status']=='COMPLETE'
assert json.loads((root/'logs/native_clip_patch950/EXIT.json').read_text())['exit_code']==0
assert json.loads((root/'logs/saved_region_comparison949/EXIT.json').read_text())['exit_code']==0
assert all(sha(root/name)==digest for name,digest in request['canonical_source_sha256'].items())
protected=root/'trained-model/joint_spectral_training943_joint_delta_full_semantic_RGBNT201/best_map.pth'
protected_sha=sha(protected)
for row in request['inventory']:
    path=Path(row['path']);assert path.resolve().is_relative_to(root.resolve())
    assert path.name in ('query_features.pt','gallery_features.pt','best_map.pth') and path!=protected
    assert path.stat().st_size==row['bytes'] and sha(path)==request['archives'][row['path']]['sha256']==row['sha256']
before=shutil.disk_usage(root).free
for row in request['inventory']:Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in request['inventory']) and sha(protected)==protected_sha
value=dict(status='CLOSED_INPUT_D_ARCHIVES_VERIFIED_REMOTE_DUPLICATES_RETIRED',archives=request['archives'],retired=request['inventory'],
    retired_bytes=sum(row['bytes'] for row in request['inventory']),remote_free_before=before,remote_free_after=shutil.disk_usage(root).free,
    retained_joint_delta_best_sha256=protected_sha,at=datetime.now().astimezone().isoformat(),
    boundary='All source/best diagnoses and saved-region comparisons terminal. Exact D size/SHA verified before removing only remote copies. Restore archived bests/feature arrays before any original consumer/verifier/report needing them. All source/datasets/unique records and selected joint_delta best remain.')
print(json.dumps(value))
'''
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'CLOSED_RETENTION_951.stdout').write_bytes(run.stdout.encode());(w/'CLOSED_RETENTION_951.stderr').write_bytes(run.stderr.encode())
(w/'CLOSED_RETENTION_951_EXIT.json').write_text(json.dumps(dict(exit_code=run.returncode))+'\n');assert run.returncode==0,run.stderr
value=json.loads(run.stdout);assert all(sha(Path(row['path']))==row['sha256'] for row in value['archives'].values())
(w/'CLOSED_RETENTION_951.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(status=value['status'],retired_bytes=value['retired_bytes'],remote_free_after=value['remote_free_after'])))
