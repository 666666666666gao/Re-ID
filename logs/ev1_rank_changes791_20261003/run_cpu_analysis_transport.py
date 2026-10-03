from datetime import datetime
from pathlib import Path
import hashlib
import json
import shlex
import paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
target=base/'ev1_rank_changes791_cpu'
assert not target.exists()
target.mkdir()
source=repo/'tools/analyze_ev1_rank_changes.py'
reference=Path('C:/Users/gb/.codex_tmp/semantic_native_complete734_20261002/raw/results/semantic_native_evidence_complete_20261002/SUMMARY.json')
root='/data/gaob/Re-ID/Trifusion'
remote_source=root+'/tools/analyze_ev1_rank_changes.py'
output=root+'/results/ev1_rank_changes_20261003_v1'
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
check=f'''from pathlib import Path
import hashlib,json
root=Path({root!r})
assert not Path({remote_source!r}).exists()
assert not Path({output!r}).exists()
assert hashlib.sha256((root/'results/semantic_native_evidence_complete_20261002/SUMMARY.json').read_bytes()).hexdigest()=={hashlib.sha256(reference.read_bytes()).hexdigest()!r}
print(json.dumps({{'status':'EXISTING_CLOSED_EV1_REPORT_CONFIRMED','new_model_runs':0}}))
'''
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(check);stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read();code=stdout.channel.recv_exit_status()
(target/'precheck.stdout.json').write_bytes(data)
(target/'precheck.stderr.txt').write_bytes(error)
assert code==0,error.decode()
sftp=client.open_sftp();sftp.put(str(source),remote_source);sftp.close()
command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1',
         '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',remote_source,'--output-dir',output]
started=datetime.now().astimezone().isoformat()
_,stdout,stderr=client.exec_command(shlex.join(command))
data,error=stdout.read(),stderr.read();code=stdout.channel.recv_exit_status()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'started_at':started,'completed_at':datetime.now().astimezone().isoformat(),'exit_code':code,'command':command,'server_port':2026,'output_dir':output,'model_runs':0,'training_runs':0})+'\n')
assert code==0,error.decode()
sftp=client.open_sftp()
for name in ('SUMMARY.json','RGBNT201_QUERY_CHANGES.csv','RGBNT100_QUERY_CHANGES.csv','MSVR310_QUERY_CHANGES.csv'):
    sftp.get(output+'/'+name,str(target/name))
sftp.close();client.close()
print(json.dumps({'status':'CPU_ANALYSIS_COMPLETE_TEXT_RECEIVED','exit_code':code,'output_dir':output,'gpu_runs':0}))
