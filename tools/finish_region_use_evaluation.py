"""Only the unattempted first strict evaluation after remaining911 storage stop."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_region_use_probe as panel

def coordinate(campaign,seal_path):
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    expected=ROOT/'logs/region_use_remaining_strict_20261013_912/region_use_remaining_20261013_911'
    assert campaign==expected and not campaign.exists()
    seal=json.loads(seal_path.read_text())
    assert seal['schema']=='trifusion-region-use-first-unattempted-strict-v1'
    assert panel.base.sha(Path(__file__))==seal['controller_sha256']
    assert panel.source_map()==seal['source_sha256']
    assert all(panel.base.sha(Path(p))==h for p,h in seal['artifact_sha256'].items())
    original=ROOT/'logs/region_use_remaining_20261013_911'
    old=json.loads((original/'campaign.json').read_text())
    exit_record=json.loads((ROOT/'logs/region_use_remaining_admin911/EXIT.json').read_text())
    assert old['status']=='STOPPED_BEFORE_STAGE_INSUFFICIENT_STORAGE'
    assert old['stopped_before_mode']=='evaluate' and exit_record['exit_code']==1
    assert len(old['jobs'])==1
    job=copy.deepcopy(old['jobs'][0])
    assert (job['dataset'],job['comparison'],job['status'])==('RGBNT100','ambiguity','PENDING')
    assert [s['mode'] for s in job['stages']]==['prepare','m0','train']
    assert all(s['status']=='COMPLETE' and s['exit_code']==0 for s in job['stages'])
    run=panel.output_dir(original,'full','RGBNT100','ambiguity')
    assert panel.output_dir(campaign,'full','RGBNT100','ambiguity')==run
    training=json.loads((run/'training.json').read_text())
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert [h['epoch'] for h in training['history']]==list(range(1,51))
    assert not (run/'official_metrics.json').exists() and not (run/'official_distances.pt').exists()
    assert shutil.disk_usage(ROOT).free>=panel.RESERVE_BYTES+panel.NEXT_ARTIFACT_BYTES
    campaign.mkdir(parents=True)
    state=copy.deepcopy(old)
    state.update(status='RUNNING_FIRST_UNATTEMPTED_STRICT_EVALUATION',jobs=[job],
        controller_pid=os.getpid(),started_at=panel.base.queue.stamp(),
        original_remaining_campaign=str(original),original_remaining_exit_code=1,
        continuation_scope='Reuse accepted prepare/M0/full50; first strict only, zero training replay.',
        strict_continuation_seal_sha256=panel.base.sha(seal_path),
        strict_continuation_controller_sha256=panel.base.sha(Path(__file__)))
    state.pop('completed_at')
    state.pop('stopped_before_mode')
    panel.base.source_map=panel.source_map
    panel.base.queue.write(campaign/'manifest.json',dict(state))
    panel.base.queue.write(campaign/'campaign.json',state)
    stage=dict(mode='evaluate',command=panel.command(original,'RGBNT100','ambiguity','evaluate'))
    job['stages'].append(stage)
    code=panel.previous.run_logged(campaign,state,stage,'RGBNT100_ambiguity_evaluate.log')
    state.update(status='FAILED' if code else 'STRICT_EXIT0_VALIDATION_PENDING',
        completed_at=panel.base.queue.stamp())
    panel.base.queue.write(campaign/'campaign.json',state)
    if code:
        job['status']='FAILED'
        panel.base.queue.write(campaign/'campaign.json',state)
        return code
    result=json.loads((run/'official_metrics.json').read_text())
    assert result['schema']==panel.SCHEMA and result['status']=='COMPLETE'
    assert result['training_epochs']==50 and result['seed']==42
    assert result['condition']['region_use_comparison']=='ambiguity'
    assert result['checkpoint_sha256']==seal['artifact_sha256'][str(run/'best_map.pth')]
    assert all(panel.base.sha(Path(p))==h for p,h in seal['artifact_sha256'].items())
    job.update(status='COMPLETE',result=result,completed_at=panel.base.queue.stamp())
    state.update(status='TRAINING_AND_STRICT_EVALUATION_COMPLETE',completed_at=panel.base.queue.stamp())
    panel.base.queue.write(campaign/'campaign.json',state)
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--seal',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(coordinate(args.campaign.resolve(),args.seal.resolve()))
