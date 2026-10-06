from pathlib import Path
import json,paramiko
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'signal_selection_prepare_failure865';assert not packet.exists();packet.mkdir()
o=json.loads((base/'signal_selection_launch865/OBSERVATION.json').read_bytes())
assert o['supervisor_exit']['exit_code']==1 and not o['actual_running_steps'] and not o['training']
step=o['campaign']['failed_preparation']
name=f'prepare_{step["dataset"]}_{step["selection"]}.log'
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
s=c.open_sftp();s.get('/data/gaob/Re-ID/Trifusion/logs/signal_selection_reference_v1_20261006_865/'+name,str(packet/name));s.close();c.close()
(packet/'OBSERVATION.json').write_bytes((base/'signal_selection_launch865/OBSERVATION.json').read_bytes())
print((packet/name).read_text()[-7000:])
