"""Archive three completed M0 probes before retiring only remote duplicates."""
import hashlib, json, shutil, subprocess
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((w/'REPORT_COLLECTED_948.json').read_bytes())
assert report['exit_code']==0
assert not (w/'RETENTION_REQUEST_948.json').exists()
modes=('joint_delta','add_delta','joint_absolute')
inventory=report['weight_inventory']
probes=[next(row for row in inventory if row['path']==f'/data/gaob/Re-ID/Trifusion/trained-model/joint_spectral_training943_{mode}_m0_semantic_RGBNT201/m0_reload_probe.pth') for mode in modes]
bests=[next(row for row in inventory if row['path']==f'/data/gaob/Re-ID/Trifusion/trained-model/joint_spectral_training943_{mode}_full_semantic_RGBNT201/best_map.pth') for mode in modes]
archive=(w/'retired_m0_948').resolve()
assert archive.is_relative_to(w.resolve()) and archive.drive=='D:' and not archive.exists()
assert shutil.disk_usage(w.resolve()).free>sum(row['bytes'] for row in probes)
archive.mkdir()
request=dict(at=datetime.now().astimezone().isoformat(),modes=list(modes),probes=probes,bests=bests,archives={},
    parent='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_training943',report_admin='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_report948',
    canonical_source_sha256=json.loads((w/'REGISTRATION_FINAL.json').read_bytes())['source_sha256'])
(w/'RETENTION_REQUEST_948.json').write_text(json.dumps(request,indent=2)+'\n')
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
for mode,probe in zip(modes,probes):
    target=archive/(mode+'_m0_reload_probe.pth')
    run=subprocess.run(['scp','-q',*options,'-P','2026','gaob@172.19.12.138:'+probe['path'],str(target)],capture_output=True)
    (w/('M0_ARCHIVE_COPY_'+mode+'_948.json')).write_text(json.dumps(dict(exit_code=run.returncode,stderr=run.stderr.decode(),at=datetime.now().astimezone().isoformat()))+'\n')
    assert run.returncode==0,run.stderr
    assert target.stat().st_size==probe['bytes'] and sha(target)==probe['sha256']
    request['archives'][mode]=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target))
(w/'M0_ARCHIVES_VERIFIED_948.json').write_text(json.dumps(request,indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
import hashlib,json,shutil
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=Path(request['parent']);report=Path(request['report_admin'])
state=json.loads((parent/'campaign.json').read_text())
assert state['status']=='COMPLETE' and state['accepted']==state['expected']==3
assert json.loads((report/'EXIT.json').read_text())['exit_code']==0
assert json.loads((root/'results/joint_spectral_report948/SUMMARY.json').read_text())['status']=='COMPLETE_SAVED_DISTANCE_DIAGNOSIS'
assert not (Path('/proc')/str(state['controller_pid'])).exists()
assert all(sha(root/n)==v for n,v in request['canonical_source_sha256'].items())
for mode,probe,best in zip(request['modes'],request['probes'],request['bests']):
    target=Path(probe['path']);expected=root/f'trained-model/joint_spectral_training943_{mode}_m0_semantic_RGBNT201/m0_reload_probe.pth'
    assert target.resolve()==expected.resolve() and target.resolve().is_relative_to((root/'trained-model').resolve())
    assert target.stat().st_size==probe['bytes'] and sha(target)==request['archives'][mode]['sha256']==probe['sha256']
    assert sha(Path(best['path']))==best['sha256']
before=shutil.disk_usage(root).free
for probe in request['probes']:Path(probe['path']).unlink()
assert all(not Path(row['path']).exists() for row in request['probes'])
assert all(sha(Path(row['path']))==row['sha256'] for row in request['bests'])
assert all(sha(root/n)==v for n,v in request['canonical_source_sha256'].items())
value=dict(status='THREE_M0_D_ARCHIVES_VERIFIED_REMOTE_DUPLICATES_RETIRED',archives=request['archives'],
    retired_probes=request['probes'],retained_bests=request['bests'],retired_bytes=sum(row['bytes'] for row in request['probes']),
    remote_free_before=before,remote_free_after=shutil.disk_usage(root).free,at=datetime.now().astimezone().isoformat(),
    boundary='Original3x50/firststrict and once CPU report completed. D copies were size/SHA verified before exact remote probe removal. Restore exact archived probes before any later M0-dependent verifier; all bests, datasets, source and unique records retained.')
(report/'M0_RETENTION.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
'''
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'M0_RETENTION_948.stdout').write_bytes(run.stdout.encode());(w/'M0_RETENTION_948.stderr').write_bytes(run.stderr.encode())
(w/'M0_RETENTION_EXIT_948.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout)
assert all(sha(Path(row['path']))==row['sha256'] for row in value['archives'].values())
(w/'M0_RETENTION_948.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(status=value['status'],retired_bytes=value['retired_bytes'],retained_bests=len(bests))))
