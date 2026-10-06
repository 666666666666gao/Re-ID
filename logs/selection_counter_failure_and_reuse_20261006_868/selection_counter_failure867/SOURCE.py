from pathlib import Path
from datetime import datetime
from collections import Counter
import json,shutil,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_signal_selection_reference as panel
campaign=root/'logs/signal_selection_reference_v1_20261006_867'
launch=root/'logs/signal_selection_reference_launch_20261006_867'
terminal=json.loads((launch/'EXIT.json').read_text());assert terminal['exit_code']==1
for record in ('LAUNCH.json','CHILD.json'):
    row=json.loads((launch/record).read_text());directory=Path('/proc')/str(row['pid'])
    assert not directory.exists() or int((directory/'stat').read_text().split()[21])!=row['start_ticks']
sources=panel.require_sources();panel.require_protected();assert len(sources)==373
state=json.loads((campaign/'campaign.json').read_text())
assert state['report_invocations']==0
job=next(j for j in state['jobs'] if (j['dataset'],j['selection'],j['phase'])==('MSVR310','global_only','full'))
assert job['status']=='RUNNING' and all(s['status']=='COMPLETE' and s['exit_code']==0 for s in job['steps'])
folder=panel.output_dir(campaign,'MSVR310','global_only','full')
training=json.loads((folder/'training.json').read_text())
formal=json.loads((folder/'official_metrics.json').read_text())
assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and formal['status']=='COMPLETE'
assert [r['epoch'] for r in training['history']]==list(range(1,51))
steps=[json.loads(line) for line in (folder/'training_steps.jsonl').read_text().splitlines()]
batches=[json.loads(line) for line in (folder/'training_batch_metadata.jsonl').read_text().splitlines()]
counts=dict(step_rows=len(steps),batch_rows=len(batches),history_updates=sum(r['steps'] for r in training['history']),
    steps_per_epoch=dict(Counter(r['epoch'] for r in steps)),
    batch_unique_id_counts=dict(Counter(len(set(r['labels'])) for r in batches)),
    batch_sizes=dict(Counter(len(r['labels']) for r in batches)),
    global_steps_consecutive=[r['global_step'] for r in batches]==list(range(1,len(batches)+1)),
    single_identity_batches=[dict(index=i,global_step=r['global_step'],size=len(r['labels']),labels=r['labels']) for i,r in enumerate(batches) if len(set(r['labels']))<2])
assert not (counts['step_rows']==counts['batch_rows']==counts['history_updates'])
weights={name:dict(sha256=panel.sha(folder/name),bytes=(folder/name).stat().st_size) for name in ('best_map.pth','official_distances.pt','best_epoch_distances.pt')}
assert weights['best_map.pth']['sha256']==formal['checkpoint_sha256']
assert weights['official_distances.pt']['sha256']==formal['distance_sha256']
assert weights['best_epoch_distances.pt']['sha256']==formal['training_best_distance_sha256']
m0=panel.verify_m0(campaign,'MSVR310','global_only')
files=[p for directory in (campaign,launch) for p in directory.rglob('*') if p.is_file()]
files.extend(p for p in folder.iterdir() if p.is_file())
files.extend(p for p in panel.output_dir(campaign,'MSVR310','global_only','m0').iterdir() if p.is_file())
files.extend(root/n for n in ('tools/run_signal_selection_reference.py','tools/run_foundation_recipe.py','tools/queue_signal_selection_reference.py','tools/continue_signal_selection_reference.py'))
files=sorted(set(p for p in files if p.suffix in ('.json','.jsonl','.log','.txt','.md','.py')))
result=dict(status='COUNTER_ACCEPTANCE_FAILURE_TERMINAL_INTAKE',at=datetime.now().astimezone().isoformat(),
    exit=terminal,campaign=state,failed_stage=job,training=training,first_strict=formal,m0=m0,counts=counts,
    weights=weights,source_sha256=sources,accepted_count=3,free_bytes=shutil.disk_usage(root).free,
    text_files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=panel.sha(p)) for p in files},
    boundary='MSVR50andfirststrictCLIcomplete, parentcounteracceptancefailed, notformalacceptedrow. ActualstateRUNNINGisstaleafteruncaughtassert; EXIT1andprocessabsenceauthoritative. No NN/eval/report rerun, nogatechange/delete/newscore/sourcechange.')
print(json.dumps(result))
