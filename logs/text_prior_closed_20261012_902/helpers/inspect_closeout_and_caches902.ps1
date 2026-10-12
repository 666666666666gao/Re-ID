$ErrorActionPreference='Stop'
$remoteSource=@'
import json
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/text_semantic_prior_v1_20261010_901'
state=json.loads((campaign/'campaign.json').read_text())
pids={state['controller_pid']}
pids.update(step['pid'] for job in state['jobs'] for step in job['steps'])
present=[]
for pid in sorted(pids):
 path=Path('/proc')/str(pid)/'cmdline'
 if path.exists(): present.append(dict(pid=pid,cmdline=path.read_bytes().replace(b'\0',b' ').decode(errors='replace')))
own=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 data=path.read_bytes().replace(b'\0',b' ').decode(errors='replace') if path.exists() else ''
 if ('text_semantic_prior_v1_20261010_901' in data or 'text_semantic_prior_v1_20261010_900' in data) and ('tools/' in data):
  own.append(dict(pid=int(path.parent.name),cmdline=data))
features=[]
locals=[]
for parent in (root/'logs',root/'trained-model'):
 for path in parent.rglob('*.pt'):
  name=path.name.lower()
  if 'feature' in name or 'evidence' in name or 'token' in name:
   row=dict(path=str(path),bytes=path.stat().st_size)
   features.append(row)
   if 'evidence' in name or 'token' in name: locals.append(row)
value=dict(observed_at=datetime.now().astimezone().isoformat(),campaign_status=state['status'],report_invocations=state['report_invocations'],
 report_exit_code=state['report_exit_code'],controller_exit_code=int((root/'logs/text_semantic_prior_resume901/CONTROLLER_EXIT_CODE.txt').read_text()),
 historical_pid_present=present,own_campaign_processes=own,feature_caches=features,local_named_caches=locals,
 boundary='CPU process/file inventory only. No GPU query, torch load, forward, training, deletion, evaluator or source modification.')
target=root/'logs/text_semantic_prior_resume901/CLOSEOUT_PROCESS_AND_CACHE_INVENTORY_902.json'
assert not target.exists()
target.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(observed_at=value['observed_at'],campaign_status=value['campaign_status'],historical_pid_present=present,own_campaign_processes=own,
 feature_cache_count=len(features),local_named_caches=locals,selected_feature_caches=[r for r in features if 'prepool' in r['path'] or 'global_task' in r['path'] or 'incremental' in r['path']])))
'@
$remoteSource | & ssh -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i 'C:/Users/gb/.ssh/id_ed25519' -p 2026 gaob@172.19.12.138 /usr/bin/python3 -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& scp -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i 'C:/Users/gb/.ssh/id_ed25519' -P 2026 gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/logs/text_semantic_prior_resume901/CLOSEOUT_PROCESS_AND_CACHE_INVENTORY_902.json 'C:/Users/gb/.codex_tmp/independent_evidence_draft/text_semantic_prior901/runtime/'
exit $LASTEXITCODE
