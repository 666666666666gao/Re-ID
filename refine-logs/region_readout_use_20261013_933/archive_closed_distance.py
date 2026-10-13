"""Keep a SHA-verified closed diagnostic locally, retire only its remote duplicate."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

w=Path(__file__).resolve().parent
old=json.loads((w.parent/'readonly_results919/INTAKE_MANIFEST_919.json').read_text())
remote='/data/gaob/Re-ID/Trifusion/logs/region_use_fixed_best_diagnosis_20261013_918/RGBNT100_uniform/diagnostic_distances.pt'
pin=old['pinned_distances'][remote]
current=json.loads((w.parent/'gcf930/results/INPUT_SEAL.json').read_text())
assert remote not in current['artifact_sha256']
local=w/'archived_readonly_distances/RGBNT100_uniform_diagnostic_distances.pt'
assert not local.exists() and not (w/'DISTANCE_ARCHIVE.json').exists()
assert shutil.disk_usage(w).free>=pin['bytes']+1073741824
local.parent.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
opts=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
def ssh(code):
    run=subprocess.run(['ssh',*opts,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],
        input=code,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    return json.loads(run.stdout)
code='remote='+repr(remote)+'\npin='+repr(pin)+'\n'+r'''
from pathlib import Path
import hashlib,json
p=Path(remote)
assert p.is_file() and p.stat().st_size==pin['bytes']
assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
d=json.loads((p.parent/'DIAGNOSIS.json').read_text())
assert d['status']=='COMPLETE' and d['optimizer_updates']==0
assert d['distance_artifact']==pin
print(json.dumps({'status':'CLOSED_SOURCE_EXACT','pin':pin}))
'''
before=ssh(code)
copy=subprocess.run(['scp',*opts,'-P','2026','gaob@172.19.12.138:'+remote,str(local)],capture_output=True,text=True)
assert copy.returncode==0,copy.stderr
assert local.stat().st_size==pin['bytes'] and sha(local)==pin['sha256']
retire=code[:code.index('print(json.dumps')]+r'''
assert p.resolve().is_relative_to(Path('/data/gaob/Re-ID/Trifusion/logs/region_use_fixed_best_diagnosis_20261013_918/RGBNT100_uniform').resolve())
p.unlink()
assert not p.exists()
from datetime import datetime
print(json.dumps({'status':'REMOTE_DUPLICATE_RETIRED_LOCAL_CANONICAL','at':datetime.now().astimezone().isoformat(),
    'remote':remote,'bytes_retired':pin['bytes'],'sha256':pin['sha256'],'weights_deleted':0,
    'restore_before_old_array_consumer':True}))
'''
after=ssh(retire)
assert sha(local)==pin['sha256']
after.update(local_archive=str(local),original_closed_pin=pin,local_completed_at=datetime.now().astimezone().isoformat())
(w/'DISTANCE_ARCHIVE.json').write_text(json.dumps(after,indent=2)+'\n')
print(json.dumps(after))
