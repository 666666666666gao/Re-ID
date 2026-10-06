from pathlib import Path
import json,paramiko
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
failed=base/'signal_selection_prepare_failure865'
(failed/'FAILURE.json').write_text(json.dumps(dict(stage='local_log_collector_field_lookup',ssh_connected=False,error='Expected failed_preparation but actual campaign failed_job is M0 after three successful prepares.'),indent=2)+'\n')
packet=base/'signal_selection_m0_entry_failure865';assert not packet.exists();packet.mkdir()
observation=json.loads((base/'signal_selection_launch865/OBSERVATION.json').read_bytes())
job=observation['campaign']['failed_job'];assert job['phase']=='m0'
name=f'{job["dataset"]}_{job["selection"]}_m0.log'
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
s=c.open_sftp();s.get('/data/gaob/Re-ID/Trifusion/logs/signal_selection_reference_v1_20261006_865/'+name,str(packet/name));s.close();c.close()
(packet/'OBSERVATION.json').write_bytes((base/'signal_selection_launch865/OBSERVATION.json').read_bytes())
print((packet/name).read_text()[-7000:])
