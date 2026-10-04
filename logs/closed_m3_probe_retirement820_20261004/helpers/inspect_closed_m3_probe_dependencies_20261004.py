from pathlib import Path
import json
import paramiko

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/closed_m3_probe_dependencies_20261004')
assert not packet.exists()
packet.mkdir()
code = '''
from datetime import datetime
from pathlib import Path
import json
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/correspondence_m3_prediction_20260929'
manifest=json.loads((campaign/'manifest.json').read_text())
state=json.loads((campaign/'campaign.json').read_text())
candidates=[]
for folder in (root/'logs',root/'refine-logs',root/'results'):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix=='.json' and ('m3' in path.name or 'role_prediction' in path.name):
            candidates.append({'path':str(path),'bytes':path.stat().st_size})
print(json.dumps({'status':'READ_ONLY_CLOSED_M3_DEPENDENCY_INSPECTION','at':datetime.now().astimezone().isoformat(),
    'campaign':str(campaign),'campaign_state':state,'manifest':manifest,'candidate_reports':candidates}))
'''
(packet/'remote.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(30)
data,error=stdout.read(),stderr.read()
exit_code=stdout.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data)
(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
value=json.loads(data)
client.close()
print(json.dumps({'status':value['status'],'at':value['at'],'campaign_state':value['campaign_state'],'candidate_reports':value['candidate_reports']},indent=2))
