from pathlib import Path
import json
import paramiko

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/closed_m0_inventory_20261004')
assert not packet.exists()
packet.mkdir()
code = '''
from datetime import datetime
from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
models=root/'trained-model'
rows=[]
for weight in sorted(models.rglob('m0_reload_probe.pth')):
    training=weight.parent/'training.json'
    item={'weight':str(weight),'bytes':weight.stat().st_size,'run_name':weight.parent.name,'training_receipt_exists':training.is_file()}
    if training.is_file():
        value=json.loads(training.read_text())
        item.update(schema=value.get('schema'),status=value.get('status'),dataset=value.get('dataset'),recipe=value.get('recipe'),variant=value.get('variant'),completed_at=value.get('completed_at'),
            probe_recorded_sha256=value.get('m0',{}).get('reload_probe_sha256'),initializer=value.get('initializer'))
    rows.append(item)
reports=[]
for folder in sorted((root/'results').iterdir()):
    path=folder/'SUMMARY.json'
    if path.is_file() and any(name in folder.name for name in ('foundation','feature_scale','semantic_native','role_input','native_research')):
        value=json.loads(path.read_text())
        reports.append({'folder':str(folder),'schema':value.get('schema'),'status':value.get('status'),'accepted':value.get('accepted'),
            'formal_epochs':value.get('formal_epochs'),'rows':[{key:item.get(key) for key in ('dataset','variant','recipe','run_dir','m0_dir','status','checkpoint_sha256','receipt_sha256')} for item in value.get('rows',[])]})
print(json.dumps({'status':'READ_ONLY_M0_WEIGHT_AND_CLOSED_REPORT_INVENTORY','at':datetime.now().astimezone().isoformat(),
 'weights':rows,'reports':reports,'total_probe_bytes':sum(row['bytes'] for row in rows),'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'Read-only directory/receipt inventory. No weight deletion, hash changes, model import/execution, training, evaluation, GPU or power/temperature query. Current six-endpoint probes remain required until their report closes.'}))
'''
(packet / 'remote.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(30)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code}) + '\n', encoding='utf-8')
assert exit_code == 0, error.decode()
record = json.loads(data)
client.close()
print(json.dumps({key:value for key,value in record.items() if key not in ('weights','reports')}))
print(json.dumps({'weight_count':len(record['weights']),'weights':[{key:row.get(key) for key in ('run_name','bytes','schema','status','completed_at')} for row in record['weights']],
    'reports':[{key:row.get(key) for key in ('folder','schema','status','accepted','formal_epochs')} for row in record['reports']]},indent=2))
