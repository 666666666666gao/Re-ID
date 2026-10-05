expected_head='27b3932698f04b1cebdc8720687469e5cc73c69f'
supervisor="from pathlib import Path\nfrom datetime import datetime\nimport subprocess,json,os,sys\nroot=Path('/data/gaob/Re-ID/Trifusion')\nlaunch=root/'logs/row_mass_role_transport_v2_launch_20261006_860'\ncommand=[sys.executable,'-B',str(root/'tools/queue_row_mass_role_transport.py'),\n '--campaign',str(root/'logs/row_mass_role_transport_v2_20261006_860'),\n '--report-dir',str(root/'results/row_mass_role_transport_v2_complete_20261006_860')]\nwith (launch/'console.log').open('x') as log:\n child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)\n (launch/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\\n')\n code=child.wait()\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nsys.exit(code)\n"
from pathlib import Path
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
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
