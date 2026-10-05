"""Finish the registered, never-started final full50 after recorded storage stop."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,str(ROOT))
from tools import queue_region_reconstruction as panel

original=ROOT/'logs/region_reconstruction_v1_20261005_858'
campaign=Path(__file__).resolve().parent
report=ROOT/'results/region_reconstruction_v1_complete_20261005_858'
plan=json.loads((campaign/'PLAN.json').read_text())
panel.configure()
assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
assert panel.base.sha(campaign/Path(__file__).name)==plan['coordinator_sha256']
assert all(panel.base.sha(Path(n))==d for n,d in plan['original_failure_sha256'].items())
assert not Path('/proc/1366428').exists() and not report.exists()
manifest=panel.base.require_sources(original)
assert len(manifest['source_sha256'])==354
panel.previous.require_controls()
old=json.loads((original/'campaign.json').read_text())
assert old['report_invocations']==0 and not old.get('active_command')
assert sum(j['phase']=='full' and j['status']=='COMPLETE' for j in old['jobs'])==5
assert old['jobs'][-1]['status']=='PENDING'
assert panel.verify_m0(original,'RGBNT100','native')['status']=='M0_PASS'
rows=[panel.previous.accepted_row(original,d,v) for d in panel.DATASETS for v in panel.VARIANTS
      if (d,v)!=('RGBNT100','native')]
run=panel.base.output_dir(original,'full','RGBNT100','native')
assert not run.exists()
manifest['administrative_completion']=dict(original_campaign=str(original),
    plan_sha256=panel.base.sha(campaign/'PLAN.json'),
    boundary='Original five accepted endpoints and all six M0s inherited. Only the previously unstarted final50 and first strict/report run here.')
panel.base.queue.write(campaign/'manifest.json',manifest)
shutil.copytree(original/'initialization',campaign/'initialization')
state=json.loads(json.dumps(old))
state.update(status='RUNNING',controller_pid=os.getpid(),started_at=panel.base.queue.stamp(),
    original_campaign=str(original),original_parent_exit_code=1,inherited_formal_endpoints=5,
    new_training_invocations=0,report_invocations=0)
panel.base.queue.write(campaign/'campaign.json',state)
full=state['jobs'][-1]
assert (full['phase'],full['dataset'],full['variant'])==('full','RGBNT100','native')
full['steps']=[]
for mode in ('train','evaluate'):
    step=dict(mode=mode,command=panel.command('RGBNT100','native',mode,original,run),
        origin='FIRST_INVOCATION_AFTER_ORIGINAL_PRELAUNCH_DISK_STOP')
    full['steps'].append(step)
    if mode=='train':
        state['new_training_invocations']+=1
        assert state['new_training_invocations']==1
    code=panel.research.run_logged(campaign,state,step,f'RGBNT100_native_first_{mode}.log')
    if code:
        full.update(status='FAILED',exit_code=code)
        panel.base.queue.write(campaign/'campaign.json',state)
        raise SystemExit(code)
assert all(panel.base.sha(Path(n))==d for n,d in plan['original_failure_sha256'].items())
row=panel.previous.accept_and_retire_probe(original,'RGBNT100','native')
full.update(status='COMPLETE',exit_code=0,completed_at=panel.base.queue.stamp(),result=row,
    completion_origin='FIRST_FRESH50_AND_FIRST_STRICT_ADMINISTRATIVE_COMPLETION')
shutil.copytree(original/'acceptance',campaign/'acceptance')
shutil.copyfile(original/'probe_retirement.jsonl',campaign/'probe_retirement.jsonl')
rows=[panel.previous.accepted_row(campaign,d,v) for d in panel.DATASETS for v in panel.VARIANTS]
panel.previous.require_controls()
panel.base.queue.write(campaign/'accepted_matrix.json',dict(schema=panel.SCHEMA,accepted=6,expected=6,rows=rows))
state.update(status='COMPLETE',completed_at=panel.base.queue.stamp(),report_invocations=1)
panel.base.queue.write(campaign/'campaign.json',state)
assert shutil.disk_usage(ROOT).free>=2*1024**3
with (campaign/'report.log').open('x') as output:
    result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/report_region_reconstruction.py'),
        '--campaign',str(campaign),'--output-dir',str(report)],cwd=ROOT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=output,stderr=subprocess.STDOUT)
state.update(report_exit_code=result.returncode,report_completed_at=panel.base.queue.stamp())
panel.base.queue.write(campaign/'campaign.json',state)
assert all(panel.base.sha(Path(n))==d for n,d in plan['original_failure_sha256'].items())
print(json.dumps(dict(status='FINAL_REGISTERED_FULL50_FIRST_STRICT_AND_REPORT_FINISHED',
    new_training_invocations=1,original_parent_exit_code=1,report_exit_code=result.returncode)),flush=True)
raise SystemExit(result.returncode)
