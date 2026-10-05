"""Launch the registered six fresh endpoints after publication completes."""
from pathlib import Path
from datetime import datetime
import hashlib,json,paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
verified=json.loads((proof/'four_copy860_2025_pending.json').read_bytes())
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/row_transport_launch860')
assert not packet.exists();packet.mkdir()
supervisor='''from pathlib import Path
from datetime import datetime
import subprocess,json,os,sys
root=Path('/data/gaob/Re-ID/Trifusion')
launch=root/'logs/row_mass_role_transport_v2_launch_20261006_860'
command=[sys.executable,'-B',str(root/'tools/queue_row_mass_role_transport.py'),
 '--campaign',str(root/'logs/row_mass_role_transport_v2_20261006_860'),
 '--report-dir',str(root/'results/row_mass_role_transport_v2_complete_20261006_860')]
with (launch/'console.log').open('x') as log:
 child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)
 (launch/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\\n')
 code=child.wait()
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\\n')
sys.exit(code)
'''
code='expected_head='+repr(verified['head'])+'\nsupervisor='+repr(supervisor)+'\n'+'''from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess,shutil,os
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==expected_head
sources=json.loads((root/'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(sources)==366 and all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in sources.items())
assert json.loads((root/'logs/row_transport_component860_20261006/COMPONENT.json').read_text())['status']=='CPU_SYNTHETIC_COMPONENT_PASS'
free=shutil.disk_usage(root).free;assert free>=4966055936
gpu=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
rows={int(line.split(',')[0]):[int(v) for v in line.split(',')[1:]] for line in gpu.splitlines()}
assert set(rows)=={0,1} and all(v[0]<500 for v in rows.values())
campaign=root/'logs/row_mass_role_transport_v2_20261006_860'
report=root/'results/row_mass_role_transport_v2_complete_20261006_860'
launch=root/'logs/row_mass_role_transport_v2_launch_20261006_860'
assert not campaign.exists() and not report.exists() and not launch.exists()
launch.mkdir()
source=launch/'SUPERVISOR.py';source.write_text(supervisor)
with (launch/'supervisor.stdout.log').open('x') as out,(launch/'supervisor.stderr.log').open('x') as err:
 process=subprocess.Popen(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(source)],cwd=root,
  stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'))
ticks=int((Path('/proc')/str(process.pid)/'stat').read_text().split(') ')[1].split()[19])
record=dict(status='REGISTERED_SIX_ENDPOINT_QUEUE_LAUNCHED',at=datetime.now().astimezone().isoformat(),
 pid=process.pid,start_ticks=ticks,launch=str(launch),campaign=str(campaign),report_dir=str(report),
 head=expected_head,source_count=len(sources),source_sha256=sources,free_before=free,gpu_memory_only=rows,
 estimated_seconds=8*3600,poll_seconds=240,
 boundary='Only26physicalGPU0/1. Original RAW controls187 reused; each real8M0 precedes its own fresh50 and first strict. No performance conclusion, retry, 2025 or power/temperature action.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
(packet/'SUPERVISOR.py').write_text(supervisor,encoding='utf-8')
(packet/'LAUNCHER_SOURCE.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(180)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n')
c.close();assert status==0,error.decode()
r=json.loads(data)
handoff=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/row_transport_live_handoff860.json')
assert not handoff.exists()
handoff.write_text(json.dumps(dict(launch=r,last_observation=None,overall_goal='ACTIVE_UNMET',
 no_relaunch=True,previous_all_handles_consumed=True),indent=2)+'\n')
print(json.dumps({k:r[k] for k in ('status','at','pid','start_ticks','campaign','launch','report_dir','head','free_before','gpu_memory_only')},indent=2))
