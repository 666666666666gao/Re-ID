"""Run only the two never-started RGBNT100 endpoints, without altering origin."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,str(ROOT))
from tools import queue_deployment_metric_role as original

ORIGIN=ROOT/'logs/deployment_metric_role_v1_20261005_837'
CAMPAIGN=ROOT/'logs/deployment_metric_role_pending100_20261005_842'


def main():
    original.configure()
    base=original.base
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    origin=json.loads((ORIGIN/'campaign.json').read_text())
    assert origin['status']=='FAILED' and 'active_command' not in origin
    assert sum(j['phase']=='full' and j['status']=='COMPLETE' for j in origin['jobs'])==3
    failed=next(j for j in origin['jobs'] if (j['dataset'],j['variant'],j['phase'])==('MSVR310','native','m0'))
    assert failed['status']=='FAILED' and failed['exit_code']==1
    assert all(j['status']=='PENDING' for j in origin['jobs'] if j['dataset']=='RGBNT100')
    assert not any(r['dataset']=='RGBNT100' for r in origin['preparation'])
    origin_sha=base.sha(ORIGIN/'campaign.json')
    assert not CAMPAIGN.exists()
    assert shutil.disk_usage(ROOT).free>=3*384*1024**2+2*1024**3
    controls,sources=original.require_controls(),original.source_map()
    jobs=[dict(dataset='RGBNT100',variant=v,phase=p,status='PENDING') for v in ('semantic','native') for p in ('m0','full')]
    CAMPAIGN.mkdir()
    manifest=dict(schema=original.SCHEMA,seed=42,epochs=50,source_sha256=sources,initialization_sha256={},jobs=jobs,
        physical_gpus=[0,1],max_parallel_jobs=1,poll_seconds=240,control_seal_sha256=base.sha(original.CONTROLS),
        role_metric_policy=original.METRIC_POLICY,origin_campaign=str(ORIGIN),origin_terminal_sha256=origin_sha,
        continuation_source_sha256=base.sha(Path(__file__)),
        boundary='Only never-started RGBNT100 pair. Original MSVR native M0 failure retained,not retried. Same sealed339 model/training sources,real8M0/fresh50/firststrict. No original six-endpoint report or power-temperature action.')
    state=dict(status='RUNNING',controller_pid=os.getpid(),started_at=base.queue.stamp(),jobs=jobs,preparation=[],
               report_invocations=0,origin_campaign=str(ORIGIN),origin_terminal_sha256=origin_sha)
    base.queue.write(CAMPAIGN/'manifest.json',manifest);base.queue.write(CAMPAIGN/'campaign.json',state)
    for variant in ('semantic','native'):
        prep=dict(dataset='RGBNT100',variant=variant,mode='prepare',
            command=original.command('RGBNT100',variant,'prepare',CAMPAIGN,ROOT/'trained-model/deployment_metric_pending100_prepare_unused'))
        state['preparation'].append(prep)
        if original.previous.previous.run_logged(CAMPAIGN,state,prep,f'prepare_RGBNT100_{variant}.log'):
            return 1
        witness=CAMPAIGN/'initialization'/f'RGBNT100_{variant}.json'
        binding=base.expected_binding(CAMPAIGN,'RGBNT100',variant)
        control=next(r['initializer'] for r in controls['rows'] if (r['dataset'],r['variant'])==('RGBNT100',variant))
        excluded=('architecture','entry_sha256','scope','role_metric_policy')
        assert {k:v for k,v in binding.items() if k not in excluded}=={k:v for k,v in control.items() if k not in excluded}
        assert binding['role_metric_policy']==original.METRIC_POLICY
        manifest['initialization_sha256'][str(witness)]=base.sha(witness);base.queue.write(CAMPAIGN/'manifest.json',manifest)
        for phase in ('m0','full'):
            if phase=='full':original.previous.previous.verify_m0(CAMPAIGN,'RGBNT100',variant)
            job=next(j for j in jobs if (j['variant'],j['phase'])==(variant,phase));job['steps']=[]
            for mode in (('m0',) if phase=='m0' else ('train','evaluate')):
                step=dict(mode=mode,command=original.command('RGBNT100',variant,mode,CAMPAIGN,base.output_dir(CAMPAIGN,phase,'RGBNT100',variant)))
                job['steps'].append(step)
                if original.previous.previous.run_logged(CAMPAIGN,state,step,f'RGBNT100_{variant}_{mode}.log'):
                    job.update(status='FAILED',exit_code=step['exit_code']);base.queue.write(CAMPAIGN/'campaign.json',state)
                    return 1
            row=original.previous.previous.verify_m0(CAMPAIGN,'RGBNT100',variant) if phase=='m0' else original.accept_and_retire_probe(CAMPAIGN,'RGBNT100',variant)
            job.update(status='COMPLETE',exit_code=0,completed_at=base.queue.stamp(),result=row);base.queue.write(CAMPAIGN/'campaign.json',state)
    rows=[original.accepted_row(CAMPAIGN,'RGBNT100',v) for v in ('semantic','native')]
    assert base.sha(ORIGIN/'campaign.json')==origin_sha
    original.require_controls()
    base.queue.write(CAMPAIGN/'accepted_matrix.json',dict(schema=original.SCHEMA,accepted=2,expected=2,rows=rows,origin_terminal_sha256=origin_sha))
    state.update(status='COMPLETE',completed_at=base.queue.stamp(),report_invocations=0)
    base.queue.write(CAMPAIGN/'campaign.json',state)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
