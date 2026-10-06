from pathlib import Path
import json,paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_head_storage_preflight873')
assert not packet.exists();packet.mkdir()
code=r'''from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
for name in ('logs/independent_role_heads_launch_20261006_872','logs/independent_role_heads_v1_20261006_872','results/independent_role_heads_v1_20261006_872'):
    assert not (root/name).exists(),name
missing=[]
for p in (root/'trained-model').glob('*/best_map.pth'):
    if p.stat().st_size<200*1024**2:continue
    if not (p.parent/'official_metrics.json').exists():
        t=json.loads((p.parent/'training.json').read_text())
        missing.append(dict(path=str(p),bytes=p.stat().st_size,nlink=p.stat().st_nlink,
                            status=t['status'],dataset=t['dataset'],epochs=t['epochs']))
keeper=root/'trained-model/metric_feature_scale_20261003_v1_full_metric_raw_MSVR310'
target=root/'trained-model/signal_selection_reference_v1_20261006_867_full_global_only_MSVR310'
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,
    keeper_receipt=json.loads((keeper/'official_metrics.json').read_text()),
    target_receipt=json.loads((target/'official_metrics.json').read_text()),unaccepted_large_bests=missing,
    own_campaign_not_created=True)))
'''
(packet/'SOURCE.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(180)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n')
assert status==0,error.decode();r=json.loads(data)
print(json.dumps(dict(at=r['at'],free_bytes=r['free_bytes'],unaccepted_large_bests=r['unaccepted_large_bests'],
    keeper_metrics=r['keeper_receipt']['metrics'],target_metrics=r['target_receipt']['metrics']),indent=2))
c.close()
