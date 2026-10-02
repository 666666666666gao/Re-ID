"""Read an existing 2026 F3 controller and its exact child artifacts once."""
from pathlib import Path
import argparse
import json
import paramiko

parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
args=parser.parse_args()
private=Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
launch=json.loads((private/'deploy/LAUNCH.json').read_bytes())
out=private/'observations'/args.label
assert not out.exists();out.mkdir(parents=True)
code='''
from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
campaign=Path(LAUNCH['campaign']);root=Path(LAUNCH['root'])
proc=Path('/proc')/str(LAUNCH['controller_pid'])
record={'observed_at':datetime.now().astimezone().isoformat(),'port':2026,
 'controller_pid':LAUNCH['controller_pid'],'controller_live':proc.is_dir(),
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True),
 'free_bytes':shutil.disk_usage(root).free,'files':{},'initialization_files':{},'children':[],
 'launcher_tail':Path(LAUNCH['log']).read_text()[-5000:],
 'boundary':'Read-only process/text observation. No model, optimizer, scoring, reporting, restart or retry.'}
if proc.is_dir():
 record['controller_cmdline']=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
for path in sorted(campaign.glob('*.json')):
 record['files'][path.name]=json.loads(path.read_text())
for path in sorted((campaign/'initialization').glob('*.json')):
 record['initialization_files'][path.name]=json.loads(path.read_text())
for phase in ('m0','full'):
 for variant in ('normalized','metric_raw'):
  for dataset in ('RGBNT201','RGBNT100','MSVR310'):
   child=campaign/(campaign.name+'_'+phase+'_'+variant+'_'+dataset)
   output=root/'trained-model'/(campaign.name+'_'+phase+'_'+variant+'_'+dataset)
   if child.exists():
    item={'phase':phase,'variant':variant,'dataset':dataset,'path':str(child),
     'files':{p.name:json.loads(p.read_text()) for p in sorted(child.glob('*.json'))},
     'logs':{p.name:p.read_text()[-3500:] for p in sorted(child.glob('*.log'))}}
    training=output/'training.json'
    if training.exists():item['training']=json.loads(training.read_text())
    steps=output/'training_steps.jsonl'
    if steps.exists():
     lines=steps.read_text().splitlines()
     item['step_rows']=len(lines)
     if lines:item['last_step']=json.loads(lines[-1])
    record['children'].append(item)
print(json.dumps(record))
'''
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write('LAUNCH='+repr(launch)+'\n'+code);stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status();client.close()
(out/'stdout.txt').write_bytes(data);(out/'stderr.txt').write_bytes(error)
(out/'exit.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
record=json.loads(data)
(out/'STATUS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state=record['files'].get('campaign.json',{})
print(json.dumps({'observed_at':record['observed_at'],'controller_live':record['controller_live'],
 'status':state.get('status'),'phase':state.get('phase'),'gpu':record['gpu'],
 'initialization_count':len(record['initialization_files']),
 'initial_forward_pairs':{name:value.get('status') for name,value in record['files'].items() if name.startswith('initial_forward_pair_')},
 'jobs':[{key:value for key,value in row.items() if key in ('phase','dataset','variant','status','gpu','pid','exit_code')} for row in state.get('jobs',[])],
 'steps':[{key:value for key,value in row.items() if key in ('phase','dataset','variant','step_rows','last_step')} for row in record['children']],
 'receipt':str(out/'STATUS.json'),'launcher_tail':record['launcher_tail'][-1800:]},indent=2))
