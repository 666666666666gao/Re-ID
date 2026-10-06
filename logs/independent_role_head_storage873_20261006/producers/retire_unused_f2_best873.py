from pathlib import Path
import json,hashlib,paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_head_unused_f2_retirement873')
assert not packet.exists();packet.mkdir()
code=r'''from pathlib import Path
from datetime import datetime
import json,shutil,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_independent_role_heads as panel
panel.source_map();seal=panel.previous.require_controls()
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        assert '/data/gaob/Re-ID/Trifusion/tools/' not in cmd,cmd
j=root/'logs/independent_role_head_unused_f2_retirement_20261006_873';assert not j.exists()
target=root/'trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/best_map.pth'
keeper=root/'trained-model/metric_feature_scale_20261003_v1_full_metric_raw_MSVR310/best_map.pth'
receipts=[]
for p in (target,keeper):
    t=json.loads((p.parent/'training.json').read_text());r=json.loads((p.parent/'official_metrics.json').read_text())
    assert t['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and t['epochs']==50 and len(t['history'])==50
    assert r['status']=='COMPLETE' and panel.base.sha(p)==r['checkpoint_sha256']
    assert panel.base.sha(p.parent/'official_distances.pt')==r['distance_sha256']
    assert panel.base.sha(p.parent/'best_epoch_distances.pt')==r['training_best_distance_sha256']
    receipts.append(r)
assert receipts[1]['metrics']['mAP']>receipts[0]['metrics']['mAP']
assert target.resolve().is_relative_to((root/'trained-model').resolve()) and target.stat().st_nlink==1
assert str(target) not in seal['artifact_sha256'] and str(target.relative_to(root)) not in seal['artifact_sha256']
qualified=dict(at=datetime.now().astimezone().isoformat(),path=str(target),sha256=panel.base.sha(target),
    bytes=target.stat().st_size,original_receipt=receipts[0],retained_receipt=receipts[1],
    free_before=shutil.disk_usage(root).free,
    boundary='Unused closed F2 raw-scale binary retired under user best-mAP retention instruction. F3 retained has higher mAP/R1/R10 but lower R5; not all-CMC dominance, not a matched causal comparison. Original histories, accepted receipts and both distance arrays remain. Current RAW187/public/author inputs unchanged.')
j.mkdir();(j/'QUALIFIED.json').write_text(json.dumps(qualified,indent=2)+'\n')
assert panel.base.sha(target)==qualified['sha256'];target.unlink()
(j/'RETIREMENT.json').write_text(json.dumps(qualified,indent=2)+'\n')
panel.source_map();panel.previous.require_controls()
assert panel.base.sha(keeper)==receipts[1]['checkpoint_sha256']
complete=dict(status='UNUSED_CLOSED_F2_BEST_BINARY_RETIRED',at=datetime.now().astimezone().isoformat(),
    retired_bytes=qualified['bytes'],free_after=shutil.disk_usage(root).free,
    required_start_bytes=panel.STORAGE_BYTES,boundary=qualified['boundary'])
(j/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
print(json.dumps(dict(**complete,files={str(p.relative_to(root)):panel.base.sha(p) for p in j.iterdir() if p.is_file()})))
'''
compile(code,'retire_unused_f2_best873','exec');(packet/'SOURCE.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(300)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n');assert status==0,error.decode()
r=json.loads(data);s=c.open_sftp()
for n,d in r['files'].items():
    p=packet/'received'/n;p.parent.mkdir(parents=True,exist_ok=True);s.get('/data/gaob/Re-ID/Trifusion/'+n,str(p))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==d
s.close();c.close();print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
