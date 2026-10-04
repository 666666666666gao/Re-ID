"""Deploy and start the registered administrative continuation exactly once."""
from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
base = private / 'independent_evidence_draft'
cleanup = json.loads((base / 'legacy_storage_retirement848/SUMMARY.json').read_bytes())
assert cleanup['status'] == 'LEGACY_STORAGE_41_FILES_RETIRED'
relative = 'refine-logs/deployment_metric_role_v1/STORAGE_CONTINUATION_848.py'
plan = 'refine-logs/deployment_metric_role_v1/STORAGE_CONTINUATION_PLAN_848.md'
ast.parse((repo / relative).read_text(encoding='utf-8'))
source_sha = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
plan_sha = hashlib.sha256((repo / plan).read_bytes()).hexdigest()
root = '/data/gaob/Re-ID/Trifusion'
launch = root + '/logs/deployment_metric_storage_launch_20261005_848'
campaign = root + '/logs/deployment_metric_storage_continuation_20261005_848'
command = ['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', root + '/' + relative]
supervisor = '\n'.join([
    'from pathlib import Path', 'from datetime import datetime', 'import json,os,subprocess',
    'launch=Path(' + repr(launch) + ')', 'command=' + repr(command),
    "with (launch/'console.log').open('x') as log:",
    ' result=subprocess.run(command,cwd=' + repr(root) + ",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)",
    "(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\n')",
    'raise SystemExit(result.returncode)', ''])
# Keep the newline in the generated Python string literal escaped.
supervisor = supervisor.replace("+'\n')", "+'\\n')")
compile(supervisor, 'storage_continuation_supervisor848.py', 'exec')
code = f'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path({root!r});launch=Path({launch!r});campaign=Path({campaign!r})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(root/{relative!r})=={source_sha!r}
assert sha(root/{plan!r})=={plan_sha!r}
scope=root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json'
assert sha(scope)=='76987d2ca7d6dc7a9aeac34000d37ebe5e7d1d8ead6d52b1832f0b1557523a3d'
sources=json.loads(scope.read_text())['source_sha256'];assert len(sources)==339
assert all(sha(root/n)==d for n,d in sources.items())
pending=root/'logs/deployment_metric_role_pending100_20261005_842/campaign.json'
assert sha(pending)=='4d5b9e00e2b3739a59c0ae411c89b28b4955701b5bb16780eaab84e54130223d'
old_exit=root/'logs/deployment_metric_pending100_launch_20261005_843/EXIT.json'
assert json.loads(old_exit.read_text())['exit_code']==1
assert not Path('/proc/3997841').exists() and not Path('/proc/3997842').exists()
assert not Path('/proc/4002738').exists()
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>=2*384*1024**2+2*1024**3
assert not launch.exists() and not campaign.exists();launch.mkdir()
(launch/'supervisor.py').write_text({supervisor!r})
with (launch/'supervisor.log').open('x') as log:
 p=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(p.pid)+'/stat').read_text()
record=dict(status='STORAGE_CONTINUATION_LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=p.pid,
 start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),campaign=str(campaign),launch_dir=str(launch),command={command!r},
 coordinator_sha256={source_sha!r},plan_sha256={plan_sha!r},scientific_sources=339,
 original_pending_campaign_sha256=sha(pending),original_exit_sha256=sha(old_exit),
 disk_free_bytes=shutil.disk_usage(root).free,gpu_memory_only=devices,
 boundary='First existingsemantic evaluation;original native never-started,same scientificsources. Original training,parentEXIT1,campaign snapshot unchanged. No semantic retraining/MSVRnative retry/report846/power-temperature action.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
compile(code, 'remote_storage_continuation_launch848.py', 'exec')
packet = base / 'storage_continuation_launch848'
assert not packet.exists()
packet.mkdir()
(packet / 'remote_source.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
sftp = client.open_sftp()
for name in (relative, plan):
    # A distinct administrative filename, never an active model/training file.
    assert name not in json.loads((repo / 'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_bytes())['source_sha256']
    sftp.put(str(repo / name), root + '/' + name)
sftp.close()
stdin, out, err = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(300)
data, error = out.read(), err.read()
rc = out.channel.recv_exit_status()
client.close()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=rc, at=datetime.now().astimezone().isoformat()))+'\n')
assert rc == 0, error.decode()
print(data.decode())
