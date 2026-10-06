"""Three independent fused-head interventions; reuse sealed raw controls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_deployment_metric_role as previous

SCHEMA='trifusion-independent-role-heads-v1'
DATASETS=previous.DATASETS
base=previous.base
CONTROLS=previous.CONTROLS
SOURCE_SCOPE=ROOT/'refine-logs/independent_role_heads_v1/SOURCE_SCOPE.json'
STORAGE_BYTES=4*384*1024**2+2*1024**3


def source_map():
    sources=json.loads(SOURCE_SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT/name)==digest for name,digest in sources.items())
    return sources


def command(dataset,variant,mode,campaign,output):
    assert variant=='semantic'
    result=previous.previous.command(dataset,variant,mode,campaign,output)
    result[2]=str(ROOT/'tools/run_independent_role_heads.py')
    return result


def configure():
    previous.previous.configure()
    base.SCHEMA,base.RECIPES=SCHEMA,('semantic',)
    base.source_map,base.command=source_map,command


def coordinate(args):
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion'
    assert not args.campaign.exists() and not args.report_dir.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    assert shutil.disk_usage(ROOT).free>=STORAGE_BYTES
    controls,sources=previous.require_controls(),source_map()
    args.campaign.mkdir(parents=True)
    jobs=[dict(phase=phase,dataset=d,variant='semantic',source='independent_role_heads',status='PENDING')
          for d in DATASETS for phase in ('m0','full')]
    manifest=dict(schema=SCHEMA,seed=42,epochs=50,source_sha256=sources,initialization_sha256={},
        jobs=jobs,physical_gpus=[0,1],max_parallel_jobs=1,poll_seconds=240,
        control_seal_sha256=base.sha(CONTROLS),role_head_policy='independent_raw_fused_heads',storage_required_bytes=STORAGE_BYTES,
        storage_policy='strict_accept_each_full_then_hash_seal_and_retire_its_m0_probe',
        boundary='Three fresh semantic evidence runs with independent fused heads. Original raw objectives, role evidence, sampler, recipe and L2_1536 output preserved. Extra trainable head capacity disclosed. No power/temperature action or parity repair.')
    state=dict(status='RUNNING',controller_pid=os.getpid(),started_at=base.queue.stamp(),
               jobs=jobs,preparation=[],report_invocations=0)
    base.queue.write(args.campaign/'manifest.json',manifest)
    base.queue.write(args.campaign/'campaign.json',state)
    for dataset in DATASETS:
        prep=dict(dataset=dataset,variant='semantic',mode='prepare',command=command(dataset,'semantic','prepare',
                  args.campaign,ROOT/'trained-model/independent_role_heads_prepare_unused'))
        state['preparation'].append(prep)
        if previous.previous.previous.run_logged(args.campaign,state,prep,f'prepare_{dataset}.log'):
            return 1
        witness=args.campaign/'initialization'/f'{dataset}_semantic.json'
        binding=base.expected_binding(args.campaign,dataset,'semantic')
        old=next(r['initializer'] for r in controls['rows'] if (r['dataset'],r['variant'])==(dataset,'semantic'))
        assert binding['objective_gradient_policy']=='global_author_heads_to_global_task_separate_fused_heads_to_role_task'
        assert binding['role_head_initialization']=='deepcopy_original_global_heads_no_rng_consumption'
        assert binding['trainable_parameters']==old['trainable_parameters']+binding['role_head_parameters']
        assert binding['trainable_parameter_tensors']==old['trainable_parameter_tensors']+binding['role_head_tensors']
        for key in ('public_clip_sha256','protocol_sha256','author_source_commit','visual_initial_sha256',
                    'camera_initial_sha256','shared_initializer_sha256','batch_size','num_instances','cfg_yaml',
                    'head_names','role_input_gradient_policy','visual_placement'):
            assert binding[key]==old[key],key
        manifest['initialization_sha256'][str(witness)]=base.sha(witness)
        base.queue.write(args.campaign/'manifest.json',manifest)
        paired=args.campaign/f'initial_forward_pair_{dataset}.json'
        prep=dict(dataset=dataset,mode='initial_forward_pair',command=[sys.executable,'-B',
             str(ROOT/'tools/check_independent_role_heads.py'),'--campaign',str(args.campaign),
             '--dataset',dataset,'--output',str(paired)])
        state['preparation'].append(prep)
        if previous.previous.previous.run_logged(args.campaign,state,prep,f'initial_forward_pair_{dataset}.log'):
            return 1
        manifest['initialization_sha256'][str(paired)]=base.sha(paired)
        base.queue.write(args.campaign/'manifest.json',manifest)
        for phase in ('m0','full'):
            if phase=='full':
                previous.previous.previous.verify_m0(args.campaign,dataset,'semantic')
            job=next(r for r in jobs if (r['dataset'],r['phase'])==(dataset,phase))
            job['steps']=[]
            for mode in (('m0',) if phase=='m0' else ('train','evaluate')):
                step=dict(mode=mode,command=command(dataset,'semantic',mode,args.campaign,
                          base.output_dir(args.campaign,phase,dataset,'semantic')))
                job['steps'].append(step)
                if previous.previous.previous.run_logged(args.campaign,state,step,f'{dataset}_{mode}.log'):
                    job.update(status='FAILED',exit_code=step['exit_code'])
                    base.queue.write(args.campaign/'campaign.json',state)
                    return 1
            row=(previous.previous.previous.verify_m0(args.campaign,dataset,'semantic') if phase=='m0'
                 else previous.accept_and_retire_probe(args.campaign,dataset,'semantic'))
            job.update(status='COMPLETE',exit_code=0,completed_at=base.queue.stamp(),result=row)
            base.queue.write(args.campaign/'campaign.json',state)
    previous.require_controls()
    rows=[previous.accepted_row(args.campaign,d,'semantic') for d in DATASETS]
    base.queue.write(args.campaign/'accepted_matrix.json',dict(schema=SCHEMA,accepted=3,expected=3,rows=rows))
    state.update(status='COMPLETE',completed_at=base.queue.stamp(),report_invocations=1)
    base.queue.write(args.campaign/'campaign.json',state)
    with (args.campaign/'report.log').open('x') as log:
        result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/report_independent_role_heads.py'),
            '--campaign',str(args.campaign),'--output-dir',str(args.report_dir)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode,report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign/'campaign.json',state)
    return result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    args.campaign,args.report_dir=args.campaign.resolve(),args.report_dir.resolve()
    configure()
    return coordinate(args)


if __name__=='__main__':
    raise SystemExit(main())
