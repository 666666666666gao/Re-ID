"""One CPU-only storage inventory while the existing F2 queue runs; no deletion."""
from pathlib import Path
import json
import paramiko

out = Path('C:/Users/gb/.codex_tmp/training_feature_scale_storage754')
assert not out.exists()
out.mkdir()
code = '''
from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/training_feature_scale_20261003_v2'
state=json.loads((campaign/'campaign.json').read_text())
owned={str(root/'trained-model'/f"{campaign.name}_{j['phase']}_{j['variant']}_{j['dataset']}") for j in state['jobs']}
rows=[]
for path in sorted((root/'trained-model').rglob('*.pth')):
 stat=path.stat()
 rows.append({'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'current_F2_run':str(path.parent) in owned})
probes=[root/'trained-model'/f'{campaign.name}_m0_{variant}_{dataset}'/'m0_reload_probe.pth'
        for variant in ('normalized','raw') for dataset in ('RGBNT201','RGBNT100','MSVR310')]
assert len(probes)==6 and all(p.is_file() for p in probes)
proc=Path('/proc')/str(state['controller_pid'])
print(json.dumps({'observed_at':datetime.now().astimezone().isoformat(),'port':2026,
 'controller_pid':state['controller_pid'],'controller_live':proc.is_dir(),
 'controller_cmdline':(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode() if proc.is_dir() else None,
 'campaign_status':state['status'],'report_invocations':state['report_invocations'],
 'jobs':[{k:v for k,v in j.items() if k in ('phase','dataset','variant','status','gpu','pid')} for j in state['jobs']],
 'disk_free_bytes':shutil.disk_usage(root).free,'weight_rows':rows,
 'm0_probe_files':[str(p) for p in probes],
 'boundary':'Read-only CPU file metadata and original process/receipt snapshot. Current six M0 probes are dependencies of original verify and final report. No weights hashed or deleted, model/scoring/benchmark/restart invoked.'}))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
client.close()
(out / 'stdout.txt').write_bytes(data)
(out / 'stderr.txt').write_bytes(error)
assert exit_code == 0, error.decode()
record = json.loads(data)
(out / 'STORAGE.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps({'observed_at':record['observed_at'],'controller_live':record['controller_live'],
 'free_bytes':record['disk_free_bytes'],'weights':len(record['weight_rows']),
 'weights_bytes':sum(r['bytes'] for r in record['weight_rows']),
 'F2_weight_files':sum(r['current_F2_run'] for r in record['weight_rows']),
 'm0_probe_files_present':len(record['m0_probe_files']),
 'full_jobs':[j for j in record['jobs'] if j['phase']=='full'],
 'report_invocations':record['report_invocations']}),flush=True)
