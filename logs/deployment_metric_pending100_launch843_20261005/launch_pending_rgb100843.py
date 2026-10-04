"""Launch only the registered never-started RGBNT100 pair once."""
from pathlib import Path
from datetime import datetime
import hashlib,json
import paramiko
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');private=Path('C:/Users/gb/.codex_tmp');base=private/'independent_evidence_draft'
proof=json.loads((private/'foundation_recipe_v1_20261002/four_copy842_2025_pending.json').read_bytes())
assert proof['status']=='FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING'
assert json.loads((base/'consumed_gallery_feature_retirement843/EXIT.json').read_bytes())['exit_code']==0
terminal=json.loads((base/'deployment_metric_original_terminal842b/stdout.json').read_bytes())
origin_sha=terminal['files']['logs/deployment_metric_role_v1_20261005_837/campaign.json']['sha256']
root='/data/gaob/Re-ID/Trifusion';relative='refine-logs/deployment_metric_role_v1/PENDING_RGBNT100_QUEUE.py'
coordinator_sha=hashlib.sha256((repo/relative).read_bytes()).hexdigest()
scope_sha=hashlib.sha256((repo/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_bytes()).hexdigest()
launch=root+'/logs/deployment_metric_pending100_launch_20261005_843'
campaign=root+'/logs/deployment_metric_role_pending100_20261005_842'
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',root+'/'+relative]
supervisor='\n'.join(['from pathlib import Path','from datetime import datetime','import json,os,subprocess',
 'launch=Path('+repr(launch)+')','command='+repr(command),"with (launch/'console.log').open('x') as log:",
 ' result=subprocess.run(command,cwd='+repr(root)+",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)",
 "(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')",'raise SystemExit(result.returncode)',''])
compile(supervisor,'pending100_supervisor843.py','exec')
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
root=Path({root!r});launch=Path({launch!r});campaign=Path({campaign!r})
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={proof['head']!r}
assert hashlib.sha256((root/{relative!r}).read_bytes()).hexdigest()=={coordinator_sha!r}
assert hashlib.sha256((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_bytes()).hexdigest()=={scope_sha!r}
scope=json.loads((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_text());assert len(scope['source_sha256'])==339
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in scope['source_sha256'].items())
origin=root/'logs/deployment_metric_role_v1_20261005_837/campaign.json'
assert hashlib.sha256(origin.read_bytes()).hexdigest()=={origin_sha!r}
assert json.loads(origin.read_text())['status']=='FAILED'
assert not Path('/proc/3606472').exists()
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>=3*384*1024**2+2*1024**3
assert not campaign.exists() and not launch.exists();launch.mkdir()
(launch/'supervisor.py').write_text({supervisor!r})
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
record=dict(status='REGISTERED_PENDING_RGBNT100_PAIR_LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=process.pid,
 start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),campaign=str(campaign),launch_dir=str(launch),command={command!r},
 source_commit={proof['head']!r},coordinator_sha256={coordinator_sha!r},source_files_verified=339,origin_terminal_sha256={origin_sha!r},expected_endpoints=2,
 gpu_memory_only=devices,disk_free_bytes=shutil.disk_usage(root).free,boundary='Only originally never-started100 pair,same model/training339source,no originalfailure retry or rewrite. Own8M0/fresh50/firststrict;2/2not6/6. Only26GPU0/1,no power/temperature action. Launch notM0/performance.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
compile(code,'remote_pending100_launch843.py','exec');packet=base/'pending_rgb100_launch843';assert not packet.exists();packet.mkdir()
(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close();(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n');assert rc==0,error.decode();print(data.decode())
