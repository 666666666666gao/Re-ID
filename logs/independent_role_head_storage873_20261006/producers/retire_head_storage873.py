"""Retire one dominated source baseline and one closed unaccepted N1 binary."""
from pathlib import Path
import json,hashlib,paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_head_storage_retirement873')
assert not packet.exists();packet.mkdir()
code=r'''from pathlib import Path
from datetime import datetime
import json,shutil,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_independent_role_heads as panel
from tools import queue_signal_selection_reference as selection
panel.source_map();seal=panel.previous.require_controls()
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        assert '/data/gaob/Re-ID/Trifusion/tools/' not in cmd,cmd
journal=root/'logs/independent_role_head_storage_retirement_20261006_873'
assert not journal.exists()
target=root/'trained-model/signal_selection_reference_v1_20261006_867_full_global_only_MSVR310/best_map.pth'
failed=root/'trained-model/native_detail_20261002_v1_clean_clip_low_MSVR310_seed42_full/best_map.pth'
keeper=root/'trained-model/metric_feature_scale_20261003_v1_full_metric_raw_MSVR310/best_map.pth'
r=json.loads((target.parent/'official_metrics.json').read_text());k=json.loads((keeper.parent/'official_metrics.json').read_text())
t=json.loads((target.parent/'training.json').read_text());kt=json.loads((keeper.parent/'training.json').read_text())
assert r['status']==k['status']=='COMPLETE'
assert t['status']==kt['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
assert len(t['history'])==len(kt['history'])==50 and t['epochs']==kt['epochs']==50
keys=('mAP','Rank-1','Rank-5','Rank-10');assert all(k['metrics'][key]>=r['metrics'][key] for key in keys)
assert panel.base.sha(target)==r['checkpoint_sha256'] and panel.base.sha(keeper)==k['checkpoint_sha256']
for folder,receipt in ((target.parent,r),(keeper.parent,k)):
    assert panel.base.sha(folder/'official_distances.pt')==receipt['distance_sha256']
    assert panel.base.sha(folder/'best_epoch_distances.pt')==receipt['training_best_distance_sha256']
ft=json.loads((failed.parent/'training.json').read_text())
assert ft['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and ft['epochs']==50 and len(ft['history'])==50
assert not (failed.parent/'official_metrics.json').exists()
assert (failed.parent/'best_epoch_distances.pt').is_file()
for p in (target,failed):
    assert p.resolve().is_relative_to((root/'trained-model').resolve()) and p.stat().st_nlink==1
    assert str(p) not in seal['artifact_sha256'] and str(p.relative_to(root)) not in seal['artifact_sha256']
proofs=[dict(path=str(target),sha256=panel.base.sha(target),bytes=target.stat().st_size,
    reason='Accepted full50 source baseline all four CMC/mAP dominated by retained F3 metric-raw best',
    original_receipt=r,retained_winner_receipt=k),
    dict(path=str(failed),sha256=panel.base.sha(failed),bytes=failed.stat().st_size,
    reason='Closed full50 N1 low MSVR, original first-strict failed and no accepted official receipt; unused by new public-CLIP semantic study and protected187. No accepted score inferred, no dominance claim.',
    original_training_sha256=panel.base.sha(failed.parent/'training.json'),
    original_training_best_distance_sha256=panel.base.sha(failed.parent/'best_epoch_distances.pt'),
    original_training_status=ft['status'],original_training_best_epoch=ft['best_epoch'])]
qualified=dict(at=datetime.now().astimezone().isoformat(),free_before=shutil.disk_usage(root).free,
    proofs=proofs,retained_checkpoint_sha256={str(keeper):panel.base.sha(keeper)},
    boundary='Original numerical acceptance failure is not repaired/reclassified. These two binary reload paths retire; their original histories, best arrays, failures and metrics remain. No current input/public/author/RAW187 removed.')
journal.mkdir();(journal/'QUALIFIED.json').write_text(json.dumps(qualified,indent=2)+'\n')
(journal/'original_unaccepted_N1_training.json').write_bytes((failed.parent/'training.json').read_bytes())
with (journal/'RETIREMENT.jsonl').open('x') as f:
    for proof in proofs:
        p=Path(proof['path']);assert panel.base.sha(p)==proof['sha256'];p.unlink()
        f.write(json.dumps(dict(retired_at=datetime.now().astimezone().isoformat(),**proof))+'\n');f.flush()
assert all(not Path(p['path']).exists() for p in proofs)
panel.source_map();panel.previous.require_controls()
assert panel.base.sha(keeper)==qualified['retained_checkpoint_sha256'][str(keeper)]
complete=dict(status='TWO_CLOSED_UNUSED_BINARY_RETIREMENTS_COMPLETE',at=datetime.now().astimezone().isoformat(),
    retired=2,retired_bytes=sum(p['bytes'] for p in proofs),free_after=shutil.disk_usage(root).free,
    required_start_bytes=panel.STORAGE_BYTES,boundary=qualified['boundary'])
(journal/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
print(json.dumps(dict(**complete,files={str(p.relative_to(root)):panel.base.sha(p) for p in journal.iterdir() if p.is_file()})))
'''
compile(code,'retire_head_storage873','exec');(packet/'SOURCE.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(300)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n')
assert status==0,error.decode();r=json.loads(data);s=c.open_sftp()
for n,d in r['files'].items():
    p=packet/'received'/n;p.parent.mkdir(parents=True,exist_ok=True)
    s.get('/data/gaob/Re-ID/Trifusion/'+n,str(p));assert hashlib.sha256(p.read_bytes()).hexdigest()==d
s.close();c.close();print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
