"""Launch the sealed six-endpoint campaign once on26 physicalGPU0/1."""
from datetime import datetime
from pathlib import Path
import hashlib,json
import paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');private=Path('C:/Users/gb/.codex_tmp')
base=private/'independent_evidence_draft'
proof=json.loads((private/'foundation_recipe_v1_20261002/four_copy837_2025_pending.json').read_bytes())
assert proof['status']=='FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING'
for receipt in ('deployment_metric_checks837/CONFIG_EXIT.json','deployment_metric_checks837/WITNESS_EXIT.json','deployment_metric_source_seal837/EXIT.json'):
    assert json.loads((base/receipt).read_bytes())['exit_code']==0
packet=base/'deployment_metric_launch837';assert not packet.exists()
scope_name='refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json'
scope_sha=hashlib.sha256((repo/scope_name).read_bytes()).hexdigest()
root='/data/gaob/Re-ID/Trifusion'
campaign=root+'/logs/deployment_metric_role_v1_20261005_837'
report=root+'/results/deployment_metric_role_v1_complete_20261005_837'
launch=root+'/logs/deployment_metric_role_launch_20261005_837'
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',root+'/tools/queue_deployment_metric_role.py','--campaign',campaign,'--report-dir',report]
supervisor='\n'.join([
 'from pathlib import Path','from datetime import datetime','import json,os,subprocess',
 'launch=Path('+repr(launch)+')','command='+repr(command),
 "with (launch/'console.log').open('x') as log:",
 ' result=subprocess.run(command,cwd='+repr(root)+",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)",
 "(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')",
 'raise SystemExit(result.returncode)',''])
compile(supervisor,'supervisor.py','exec')
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
root=Path({root!r})
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={proof['head']!r}
scope=root/{scope_name!r};assert hashlib.sha256(scope.read_bytes()).hexdigest()=={scope_sha!r}
bound=json.loads(scope.read_text());assert len(bound['source_sha256'])==339
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in bound['source_sha256'].items())
control=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert hashlib.sha256(control.read_bytes()).hexdigest()==bound['control_seal_sha256']
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in json.loads(control.read_text())['artifact_sha256'].items())
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>=7*384*1024**2+2*1024**3
campaign,report,launch=map(Path,{[campaign,report,launch]!r})
assert not campaign.exists() and not report.exists() and not launch.exists()
launch.mkdir()
(launch/'supervisor.py').write_text({supervisor!r})
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
record=dict(status='LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=process.pid,start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
 campaign=str(campaign),report_dir=str(report),launch_dir=str(launch),command={command!r},source_commit={proof['head']!r},source_files_verified=339,
 scope_sha256={scope_sha!r},gpu_memory_only=devices,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Only26GPU0/1,no power/temperature action. Own8M0 thenfresh50/firststrict perendpoint; onlyafterfull acceptance retire thatprobe withSHAjournal. Oldcontrols not replayed. Launch is notM0 or performance.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
compile(code,'remote_launch837.py','exec')
packet.mkdir();(packet/'remote_launch837.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
client.close();assert rc==0,error.decode()
print(json.dumps(json.loads(data)))
