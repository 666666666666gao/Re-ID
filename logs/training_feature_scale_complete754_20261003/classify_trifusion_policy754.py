"""Distinguish training cwd from shared Trifusion pretrained-weight references."""
from datetime import datetime
from pathlib import Path
import json
import paramiko

out = Path('C:/Users/gb/.codex_tmp/trifusion_compute_policy754')
assert not (out/'PROCESS_CLASSIFICATION.json').exists()
code = '''
from pathlib import Path
from datetime import datetime
import json, subprocess
text = subprocess.check_output(['ps','-u','gaob','-o','pid=,ppid=,args='], text=True)
rows = []
for line in text.splitlines():
    if 'trifusion' not in line.lower():
        continue
    pid, ppid, command = line.strip().split(None,2)
    result = subprocess.run(['readlink','/proc/'+pid+'/cwd'],capture_output=True,text=True)
    rows.append({'pid':int(pid),'ppid':int(ppid),'command':command,
                 'cwd_read_exit_code':result.returncode,'cwd':result.stdout.strip(),
                 'cwd_read_error':result.stderr})
assert all(r['cwd_read_exit_code']==0 for r in rows), rows
actual = [r for r in rows if r['cwd']=='/data2/gb/Re-ID/Trifusion'
          or r['cwd'].startswith('/data2/gb/Re-ID/Trifusion/')]
print(json.dumps({'observed_at':datetime.now().astimezone().isoformat(),'port':2025,
 'candidate_processes':rows,'actual_trifusion_processes':actual,
 'boundary':'Processes mentioning shared pretrained weights are classified by actual cwd; no process stopped or changed.'}))
'''
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2025,username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read()
status=stdout.channel.recv_exit_status()
client.close()
(out/'classification.stdout.txt').write_bytes(data)
(out/'classification.stderr.txt').write_bytes(error)
assert status==0,error.decode()
record=json.loads(data)
(out/'PROCESS_CLASSIFICATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'observed_at':record['observed_at'],
                  'candidate_processes':len(record['candidate_processes']),
                  'actual_trifusion_processes':len(record['actual_trifusion_processes']),
                  'other_project_cwds':sorted({r['cwd'] for r in record['candidate_processes']})}),flush=True)

statepath=Path('C:/Users/gb/.codex_tmp/trifusion_next_state_20261001.json')
state=json.loads(statepath.read_text(encoding='utf-8'))
policy=state['active_compute_policy']
assert policy['training_ports']==[2026] and policy['gpu_indices']==[0,1,2,3]
assert policy['max_simultaneous_single_gpu_jobs']==4 and not policy['new_2025_training_authorized']
policy['latest_user_confirmation']='现在不用25，只用26的四台显卡吧'
policy['latest_verification_receipt']=str(out/'COMPUTE_POLICY.json')
policy['process_classification_receipt']=str(out/'PROCESS_CLASSIFICATION.json')
policy['latest_verified_at']=record['observed_at']
state['next_candidate']['report_executed']=True
state['updated_at']=datetime.now().astimezone().isoformat()
statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-03.md').open('a',encoding='utf-8') as stream:
    stream.write('\n- '+record['observed_at']+' 用户再次确认TriFusion只用2026四卡0–3/max4，2025不新增训练；本地调度政策已一致，无须改动封存队列。06:06实查2026无计算进程，F2六端/唯一报告结束；2025命令中的Trifusion仅为共享预训练路径，实际cwd全部DeMo-DualAxis、该项目训练未停止。两次只读回执保留于.codex_tmp/trifusion_compute_policy754；未启动或重跑任何模型。\n')
