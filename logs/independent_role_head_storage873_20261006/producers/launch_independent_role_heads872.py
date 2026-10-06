"""First launch of the registered three-endpoint head ownership study."""
from pathlib import Path
import json
import hashlib
import paramiko

proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
published=json.loads((proof/'four_copy872_2025_pending.json').read_bytes())
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_head_launch872')
assert not packet.exists(); packet.mkdir()
code=r'''from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_independent_role_heads as panel
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
assert head=='dc26b2be616fff109e13284580b218f9e0aff47d'
sources=panel.source_map();assert len(sources)==385
controls=panel.previous.require_controls();assert len(controls['artifact_sha256'])==187
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        assert '/data/gaob/Re-ID/Trifusion/tools/' not in cmd,cmd
memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used',
    '--format=csv,noheader,nounits'],text=True)
used={int(line.split(',')[0]):int(line.split(',')[1]) for line in memory.splitlines()}
assert used[0]<500 and used[1]<500
free=shutil.disk_usage(root).free;assert free>=panel.STORAGE_BYTES
journal=root/'logs/independent_role_heads_launch_20261006_872'
campaign=root/'logs/independent_role_heads_v1_20261006_872'
report=root/'results/independent_role_heads_v1_20261006_872'
assert not journal.exists() and not campaign.exists() and not report.exists()
journal.mkdir()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/queue_independent_role_heads.py'),
    '--campaign',str(campaign),'--report-dir',str(report)]
qualified=dict(at=datetime.now().astimezone().isoformat(),head=head,free_bytes=free,
    storage_required_bytes=panel.STORAGE_BYTES,gpu_memory_only=memory,source_sha256=sources,
    control_seal_sha256=panel.base.sha(panel.CONTROLS),command=command,
    boundary='First registered launch, no previous run restarted. Only26GPU0/1; no25/power/temperature. No source or Git mutation until producers terminal.')
(journal/'QUALIFIED.json').write_text(json.dumps(qualified,indent=2)+'\n')
supervisor=r"""from pathlib import Path
from datetime import datetime
import json,os,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
j=root/'logs/independent_role_heads_launch_20261006_872'
q=json.loads((j/'QUALIFIED.json').read_text())
with (j/'controller.log').open('x') as log:
    p=subprocess.Popen(q['command'],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (j/'CHILD.json').write_text(json.dumps(dict(pid=p.pid,start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]),
        at=datetime.now().astimezone().isoformat(),command=q['command']))+'\n')
    status=p.wait()
(j/'EXIT.json').write_text(json.dumps(dict(exit_code=status,completed_at=datetime.now().astimezone().isoformat()))+'\n')
"""
compile(supervisor,'role_head_supervisor872','exec')
(journal/'supervisor.py').write_text(supervisor)
with (journal/'supervisor.log').open('x') as log:
    p=subprocess.Popen(['/usr/bin/python3','-B',str(journal/'supervisor.py')],cwd=root,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(status='FIRST_REGISTERED_QUEUE_SUPERVISOR_LIVE',pid=p.pid,
    start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]),
    at=datetime.now().astimezone().isoformat(),head=head,physical_gpus=[0,1],
    campaign=str(campaign),report_dir=str(report),command=command,
    boundary='Verified supervisor handle at launch; formal initialization/M0/full status is not implied by this record.')
(journal/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(dict(launch=launch,qualified=qualified,
    files={str(p.relative_to(root)):panel.base.sha(p) for p in journal.iterdir() if p.is_file()})))
'''
compile(code,'launch_role_heads872','exec');(packet/'SOURCE.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(300)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n')
assert status==0,error.decode()
r=json.loads(data); assert r['launch']['head']==published['head']
s=c.open_sftp()
for n,d in r['files'].items():
    p=packet/'received'/n;p.parent.mkdir(parents=True,exist_ok=True)
    s.get('/data/gaob/Re-ID/Trifusion/'+n,str(p));assert hashlib.sha256(p.read_bytes()).hexdigest()==d
s.close();c.close()
print(json.dumps(r['launch'],indent=2))
