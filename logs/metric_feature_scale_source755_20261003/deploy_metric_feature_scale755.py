"""Launch the reviewed fresh F3 panel once, on server2026 only."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shlex
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
out = Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755/deploy')
assert not out.exists()
publication = json.loads((proof/'publication755_local.json').read_bytes())
synced = json.loads((proof/'five_copy755.json').read_bytes())
assert synced['status'] == 'FIVE_DOC_COPIES_AND_CUMULATIVE_OWNED_TEXT_PASS'
assert synced['head'] == json.loads((proof/'commit755.json').read_bytes())['head']
inputs = json.loads(Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755/SOURCE_REVIEW_INPUTS.json').read_bytes())
for name, digest in inputs['new_protected_inputs'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
old = json.loads(Path('C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/manifest.json').read_bytes())['source_sha256']
assert len(old) == 259
new_names = [Path(name).relative_to(repo).as_posix() for name in inputs['new_protected_inputs']]
new_names.append('refine-logs/metric_feature_scale_v1/EXPERIMENT_CODE_REVIEW.md')
expected = dict(old, **{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in new_names})
assert len(expected) == 265
out.mkdir()
root = '/data/gaob/Re-ID/Trifusion'
python = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
gate = f'''
from pathlib import Path
import hashlib,json,os,sys
root=Path({root!r});os.chdir(root);sys.path.insert(0,str(root))
assert os.environ['CUDA_VISIBLE_DEVICES']==''
from tools import queue_metric_feature_scale as panel
panel.configure()
assert panel.source_map()=={expected!r}
import torch,numpy,mamba_ssm,timm
assert torch.__version__=='2.5.1+cu121' and numpy.__version__=='1.24.4'
assert mamba_ssm.__version__=='2.2.6.post3' and timm.__version__=='1.0.15'
assert not torch.cuda.is_initialized()
print(json.dumps({{'status':'CPU_IMPORT_AND_SOURCE_BINDINGS_PASS','source_count':265,
 'source_sha256':panel.source_map(),'entry':str(Path(panel.__file__).resolve()),
 'torch':torch.__version__,'numpy':numpy.__version__,'mamba':mamba_ssm.__version__,
 'timm':timm.__version__,'cuda_initialized':torch.cuda.is_initialized(),
 'boundary':'CUDA-hidden import/source check only; no model, optimizer or score executed.'}}))
'''
code = f'''
from datetime import datetime
from pathlib import Path
import json,os,shutil,subprocess,sys
root=Path({root!r});os.chdir(root)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=={synced['head']!r}
gate=subprocess.run([{python!r},'-B','-c',{gate!r}],cwd=root,
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
gate_dir=root/'logs/metric_feature_scale_prelaunch755_20261003'
assert not gate_dir.exists();gate_dir.mkdir()
(gate_dir/'source.stdout.txt').write_bytes(gate.stdout)
(gate_dir/'source.stderr.txt').write_bytes(gate.stderr)
(gate_dir/'source.exit.json').write_text(json.dumps({{'exit_code':gate.returncode}})+'\\n')
assert gate.returncode==0,gate.stderr.decode()
source=json.loads(gate.stdout)
(gate_dir/'SOURCE_GATE.json').write_text(json.dumps(source,indent=2)+'\\n')
previous=json.loads((root/'logs/training_feature_scale_20261003_v2/campaign.json').read_bytes())
assert previous['status']=='COMPLETE' and previous['report_invocations']==1 and previous['report_exit_code']==0
assert not (Path('/proc')/str(previous['controller_pid'])).exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert [int(row.split(',')[0]) for row in gpu.splitlines()]==[0,1,2,3]
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
available=[int(row.split(',')[0]) for row in gpu.splitlines() if int(row.split(',')[3])<500]
assert available
initial_gpu=available[0]
initial_uuid=next(row.split(',')[1].strip() for row in gpu.splitlines() if int(row.split(',')[0])==initial_gpu)
assert not any(row.split(',')[0].strip()==initial_uuid for row in compute.splitlines())
free=shutil.disk_usage(root).free;assert free>=10*1024**3
campaign=root/'logs/metric_feature_scale_20261003_v1'
report=root/'results/metric_feature_scale_complete_20261003'
log=root/'logs/metric_feature_scale_20261003_v1_launcher.log'
assert not campaign.exists() and not report.exists() and not log.exists()
for phase in ('m0','full'):
 for variant in ('normalized','metric_raw'):
  for dataset in ('RGBNT201','RGBNT100','MSVR310'):
   assert not (root/'trained-model'/f'metric_feature_scale_20261003_v1_{{phase}}_{{variant}}_{{dataset}}').exists()
command=[{python!r},'-B',str(root/'tools/queue_metric_feature_scale.py'),'--campaign',str(campaign),
 '--report-dir',str(report),'--gpu',str(initial_gpu)]
with log.open('x') as stream:
 process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,
 stderr=subprocess.STDOUT,start_new_session=True)
assert process.poll() is None
cmdline=(Path('/proc')/str(process.pid)/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert str(campaign) in cmdline
receipt={{'status':'CONTROLLER_STARTED_INITIALIZATION_PENDING','observed_at':datetime.now().astimezone().isoformat(),
 'port':2026,'training_ports':[2026],'gpu_scope':[0,1,2,3],'max_parallel':4,
 'initial_gpu':initial_gpu,'root':str(root),'python':{python!r},'head':{synced['head']!r},
 'controller_pid':process.pid,'command':command,'cmdline':cmdline,'campaign':str(campaign),
 'report':str(report),'log':str(log),'source_sha256':source['source_sha256'],
 'gpu_before':gpu,'compute_before':compute,'free_bytes':free,'environment_rebuilt':False,
 'boundary':'One actual controller launch only. Initial pair witnesses and all6eight-update M0s must pass before fresh full50; no performance result yet.'}}
(gate_dir/'LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data,error = stdout.read(),stderr.read()
exit_code = stdout.channel.recv_exit_status()
(out/'stdout.txt').write_bytes(data);(out/'stderr.txt').write_bytes(error)
(out/'exit.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code == 0, error.decode()
receipt = json.loads(data)
(out/'LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
sftp = client.open_sftp()
for name in ('SOURCE_GATE.json','source.stdout.txt','source.stderr.txt','source.exit.json'):
    sftp.get(root+'/logs/metric_feature_scale_prelaunch755_20261003/'+name,str(out/name))
sftp.close();client.close()
print(json.dumps({key:receipt[key] for key in ('status','observed_at','port','controller_pid','campaign','initial_gpu','free_bytes')},indent=2))
