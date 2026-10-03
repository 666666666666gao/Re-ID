"""Deploy the reviewed first-evaluation finish on 2026 GPU1 once."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import paramiko

base=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766')
review_path=base/'reviewer_source/SOURCE_REVIEW.json'
review=json.loads(review_path.read_bytes())
assert review['verdict'] in ('PASS','WARN') and review['blocking_findings']==[]
assert review['review_independence']=='same-family' and review['acceptance_status']=='provisional'
bound={Path(name).resolve():digest for name,digest in review['reviewed_input_hashes'].items()}
for name in ('retire_closed_m0_766.py','finish_metric_feature_scale766.py'):
    path=Path('C:/Users/gb/.codex_tmp')/name
    assert bound[path.resolve()]==hashlib.sha256(path.read_bytes()).hexdigest()
spec=json.loads((base/'FINISH_SPEC.json').read_bytes())
retirement_path=base/'retirement/RETIREMENT.json'
retirement=json.loads(retirement_path.read_bytes())
assert retirement['status']=='RETIRED_EXACT_24_CLOSED_M0' and retirement['retired_bytes']==8444122668
assert spec['port']==2026 and spec['root']=='/data/gaob/Re-ID/Trifusion'
out=base/'deploy'
assert not out.exists();out.mkdir()
files={'finish_metric_feature_scale766.py':Path('C:/Users/gb/.codex_tmp/finish_metric_feature_scale766.py'),
       'FINISH_SPEC.json':base/'FINISH_SPEC.json','STORAGE_FINISH_PLAN.md':base/'STORAGE_FINISH_PLAN.md',
       'SOURCE_REVIEW.json':review_path,'SOURCE_REVIEW.md':base/'reviewer_source/SOURCE_REVIEW.md'}
root=spec['root'];asset=root+'/logs/metric_storage_finish_source766_20261003'
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
sftp=client.open_sftp();sftp.mkdir(asset)
hashes={}
for name,path in files.items():
    data=path.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
    with sftp.open(asset+'/'+name,'wb') as stream:stream.write(data)
    with sftp.open(asset+'/'+name,'rb') as stream:assert hashlib.sha256(stream.read()).hexdigest()==hashes[name]
sftp.close()
code=f'''
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path({root!r});asset=Path({asset!r});spec=json.loads((asset/'FINISH_SPEC.json').read_text())
assert all(hashlib.sha256((asset/name).read_bytes()).hexdigest()==expected for name,expected in {hashes!r}.items())
assert hashlib.sha256(Path(spec['retirement_receipt']).read_bytes()).hexdigest()=={hashlib.sha256(retirement_path.read_bytes()).hexdigest()!r}
assert hashlib.sha256((Path(spec['campaign'])/'campaign.json').read_bytes()).hexdigest()==spec['original_campaign_sha256']
assert not Path(spec['finish_directory']).exists()
assert shutil.disk_usage(root).free>=10*1024**3
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert dict((int(row.split(',')[0]),int(row.split(',')[1])) for row in gpu.splitlines())[1]<500
log=asset/'launcher.log';assert not log.exists()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(asset/'finish_metric_feature_scale766.py'),'--spec',str(asset/'FINISH_SPEC.json'),'--gpu','1']
with log.open('x') as stream:
 process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert process.poll() is None
cmdline=(Path('/proc')/str(process.pid)/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert str(asset/'finish_metric_feature_scale766.py') in cmdline
receipt={{'status':'STORAGE_FINISH_CONTROLLER_STARTED_EVALUATION_PENDING','observed_at':datetime.now().astimezone().isoformat(),
 'port':2026,'gpu':1,'gpu_scope':[0,1,2,3],'controller_pid':process.pid,'command':command,'cmdline':cmdline,
 'root':str(root),'asset_directory':str(asset),'finish_directory':spec['finish_directory'],'log':str(log),
 'source_asset_sha256':{hashes!r},'review_sha256':{hashlib.sha256(review_path.read_bytes()).hexdigest()!r},
 'retirement_receipt_sha256':{hashlib.sha256(retirement_path.read_bytes()).hexdigest()!r},
 'gpu_before':gpu,'free_bytes_before':shutil.disk_usage(root).free,
 'boundary':'One reviewed storage-only finish launch. Missing original strict evaluation first invocation and CPU report not yet accepted. No training restart, source/method/tolerance/checkpoint change.'}}
(asset/'LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
(out/'stdout.json').write_bytes(data);(out/'stderr.txt').write_bytes(error)
(out/'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
receipt=json.loads(data)
(out/'LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps({key:receipt[key] for key in ('status','observed_at','controller_pid','gpu','free_bytes_before')},indent=2))
