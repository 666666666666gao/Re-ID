"""Once-only text intake after original870 NN and distinct871 CPU report terminate."""
from pathlib import Path
import hashlib,json,paramiko
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/signal_selection_complete871')
assert not packet.exists();packet.mkdir()
code=r'''from pathlib import Path
from datetime import datetime
import json,hashlib,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_signal_selection_reference as panel
launch=root/'logs/signal_selection_reference_launch_20261006_870';campaign=root/'logs/signal_selection_reference_v1_20261006_870';report=root/'results/signal_selection_reference_complete_20261006_871'
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1
for n in ('LAUNCH.json','CHILD.json'):
 r=json.loads((launch/n).read_text());p=Path('/proc')/str(r['pid'])
 assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
state=json.loads((campaign/'campaign.json').read_text());assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==1
assert len(state['jobs'])==18 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert state['reused_formal']==7 and state['new_formal_expected']==2 and state['reused_m0']==8 and state['new_m0_expected']==1
summary=json.loads((report/'SUMMARY.json').read_text());assert summary['status']=='COMPLETE' and summary['accepted']==9 and summary['formal_epochs']==450 and summary['formal_steps']==19452
assert len(summary['pairs'])==9 and all(len(p['paired_diagnosis']['query_changes'])==({'RGBNT201':836,'MSVR310':591,'RGBNT100':1715}[p['dataset']]) for p in summary['pairs'])
matrix=json.loads((campaign/'accepted_matrix.json').read_text());origins=json.loads((campaign/'endpoint_origins.json').read_text());metadata=json.loads((campaign/'batch_metadata_paths.json').read_text())
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==9
for row in matrix['rows']:
 key=f'{row["dataset"]}:{row["variant"]}'
 assert panel.accepted_row(Path(origins[key]),row['dataset'],row['variant'],batch_metadata_path=Path(metadata[key]))==row
panel.require_sources();panel.require_protected()
overlay=json.loads((root/'refine-logs/signal_selection_disk_continuation_v2/SOURCE_SCOPE.json').read_text())['source_sha256'];assert len(overlay)==4 and all(panel.sha(root/n)==s for n,s in overlay.items())
for version in (866,867,868,869):assert json.loads((root/f'logs/signal_selection_reference_launch_20261006_{version}/EXIT.json').read_text())['exit_code']==1
retirements=[]
for version in (866,867,868,870):retirements.extend(map(json.loads,(root/f'logs/signal_selection_reference_v1_20261006_{version}/retired_m0.jsonl').read_text().splitlines()))
assert len(retirements)==9 and len({r['path'] for r in retirements})==9 and all(not Path(r['path']).exists() for r in retirements)
cpu=root/'logs/selection_report_completion_20261006_871'
assert json.loads((cpu/'EXIT.json').read_text())['exit_code']==0
for n in ('LAUNCH.json','CHILD.json'):
 r=json.loads((cpu/n).read_text());p=Path('/proc')/str(r['pid']);assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
qualification=json.loads((cpu/'QUALIFIED.json').read_text())
assert all(panel.sha(p)==d for p,d in qualification['original_immutable_sha256'].items())
assert not (root/'results/signal_selection_reference_complete_20261006_870').exists()
texts=[]
texts.extend(p for p in cpu.rglob('*') if p.is_file())
for folder in (launch,campaign,report):texts.extend(p for p in folder.rglob('*') if p.is_file())
for row in matrix['rows']:
 folder=Path(row['run_dir']);texts.extend(folder/n for n in ('training.json','official_metrics.json','training_steps.jsonl','training_batch_metadata.jsonl'))
texts.extend(Path(p) for p in metadata.values())
assert all(p.suffix.lower() not in ('.pt','.pth','.npy','.npz') for p in texts)
files={str(p.relative_to(root)):panel.sha(p) for p in texts}
result=dict(status='NINE_FULL50_ONCE_ORIGINAL_REPORT_VERIFIED',at=datetime.now().astimezone().isoformat(),formal_epochs=450,formal_steps=19452,rows=matrix['rows'],primary_pairs=[dict(dataset=p['dataset'],advance=p['phase_progress'],delta=p['paired_diagnosis']['delta_metrics']) for p in summary['pairs'] if p['control']=='all_patch'],retired_m0=len(retirements),files=files,boundary='Text intake only; no NN, evaluator/report/score replay. Old866/867/868/869 failures preserved, seven existing endpoints not retrained, all single-seed consumed-official limits retained. Weights and matrices remain remote. Not SOTA or goal completion.')
print(json.dumps(result))
'''
compile(code,'complete_selection870_gate','exec');(packet/'SOURCE.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts');c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(900)
data,error=o.read(),e.read();status=o.channel.recv_exit_status();(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error);(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n');assert status==0,error.decode()
r=json.loads(data);s=c.open_sftp()
for n,d in r['files'].items():
 p=packet/'received'/n;p.parent.mkdir(parents=True,exist_ok=True);s.get('/data/gaob/Re-ID/Trifusion/'+n,str(p));assert hashlib.sha256(p.read_bytes()).hexdigest()==d
s.close();c.close();(packet/'COMPLETE.json').write_text(json.dumps(dict(status=r['status'],files=len(r['files']),at=r['at']))+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('rows','files')},indent=2))
