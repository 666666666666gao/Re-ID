"""Nine fresh source-reference endpoints on one physical GPU0/1 pair."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DATASETS=('RGBNT201','MSVR310','RGBNT100')
SELECTIONS=('global_only','masked','all_patch')
SCHEMA='trifusion-signal-selection-reference-v1'
SCOPE=ROOT/'refine-logs/signal_selection_reference_v1/SOURCE_SCOPE.json'
PROTECTED=ROOT/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
STORAGE_BYTES=10*360*1024**2+2*1024**3

def stamp():return datetime.now().astimezone().isoformat()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def require_sources():
    sources=json.loads(SCOPE.read_text())['source_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in sources.items())
    return sources

def require_protected():
    protected=json.loads(PROTECTED.read_text())['artifact_sha256']
    assert len(protected)==187 and all(sha(name)==digest for name,digest in protected.items())

def output_dir(campaign,dataset,selection,phase):
    return ROOT/'trained-model'/f'{campaign.name}_{phase}_{selection}_{dataset}'

def command(campaign,dataset,selection,mode,output):
    return [sys.executable,'-B',str(ROOT/'tools/run_signal_selection_reference.py'),
        '--dataset',dataset,'--selection',selection,'--mode',mode,
        '--protocol',str(ROOT/f'logs/training_feature_scale_protocols_20261002/{dataset}.json'),
        '--signal-source',str(ROOT/'comparators/Signal-cd1b0a6'),
        '--clip-weight',str(ROOT/'pertrained-model/ViT-B-16.pt'),
        '--initialization',str(campaign/f'initialization/{dataset}_{selection}.json'),
        '--output-dir',str(output)]

def run(campaign,state,step,log_name):
    assert shutil.disk_usage(ROOT).free>=2*1024**3
    require_sources()
    step.update(status='RUNNING',started_at=stamp())
    with (campaign/log_name).open('x') as log:
        process=subprocess.Popen(step['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        step['pid']=process.pid
        step['start_ticks']=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21])
        write(campaign/'campaign.json',state)
        process.wait()
        step.update(status='COMPLETE' if process.returncode==0 else 'FAILED',
                    exit_code=process.returncode,completed_at=stamp())
    write(campaign/'campaign.json',state)
    return process.returncode

def binding(campaign,dataset,selection):
    witness=json.loads((campaign/f'initialization/{dataset}_{selection}.json').read_text())
    assert witness['schema']==SCHEMA and witness['status']=='INITIALIZATION_VERIFIED'
    return witness['binding']

def verify_m0(campaign,dataset,selection):
    folder=output_dir(campaign,dataset,selection,'m0')
    result=json.loads((folder/'training.json').read_text())
    assert result['schema']==SCHEMA and result['status']=='M0_PASS'
    assert result['initializer']==binding(campaign,dataset,selection)
    assert len(result['history'])==1 and result['history'][0]['steps']==8
    assert result['m0']['nonzero_gradient_parameters']==result['m0']['trainable_parameters']
    assert result['m0']['reload_max_abs_difference']<=1e-5
    assert all(count==8 for count in result['selection_reference']['m0_bn_counts'].values())
    assert sha(folder/'m0_reload_probe.pth')==result['m0']['reload_probe_sha256']
    assert len((folder/'training_batch_metadata.jsonl').read_text().splitlines())==8
    return result

def accepted_row(campaign,dataset,selection,*,batch_metadata_path=None):
    folder=output_dir(campaign,dataset,selection,'full')
    receipt=json.loads((folder/'official_metrics.json').read_text())
    training=json.loads((folder/'training.json').read_text())
    assert receipt['schema']==SCHEMA and receipt['status']=='COMPLETE'
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert training['initializer']==binding(campaign,dataset,selection)
    assert [row['epoch'] for row in training['history']]==list(range(1,51))
    best=max(training['history'],key=lambda row:(row['official_fused']['mAP'],row['epoch']))
    assert best['epoch']==training['best_epoch']==receipt['selected_epoch']
    assert all(abs(best['official_fused'][k]-v)<1e-5 for k,v in receipt['metrics'].items())
    assert sha(folder/'best_map.pth')==receipt['checkpoint_sha256']
    assert sha(folder/'official_distances.pt')==receipt['distance_sha256']
    assert sha(folder/'best_epoch_distances.pt')==receipt['training_best_distance_sha256']
    steps=[json.loads(line) for line in (folder/'training_steps.jsonl').read_text().splitlines()]
    batches=[json.loads(line) for line in (folder/'training_batch_metadata.jsonl' if batch_metadata_path is None else batch_metadata_path).read_text().splitlines()]
    assert len(steps)==len(batches)==sum(row['steps'] for row in training['history'])
    assert [row['global_step'] for row in batches]==list(range(1,len(batches)+1))
    return dict(status='VERIFIED_COMPLETE',dataset=dataset,variant=selection,run_dir=str(folder),
        best_epoch=best['epoch'],metrics=receipt['metrics'],checkpoint_sha256=receipt['checkpoint_sha256'],
        distance_sha256=receipt['distance_sha256'],receipt_sha256=sha(folder/'official_metrics.json'),
        initializer=training['initializer'],formal_steps=len(steps))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    campaign,report=args.campaign.resolve(),args.report_dir.resolve()
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    assert not campaign.exists() and not report.exists()
    assert shutil.disk_usage(ROOT).free>=STORAGE_BYTES
    require_protected()
    sources=require_sources()
    campaign.mkdir(parents=True)
    jobs=[dict(dataset=d,selection=s,phase=p,status='PENDING') for d in DATASETS for s in SELECTIONS for p in ('m0','full')]
    state=dict(schema=SCHEMA,status='RUNNING',controller_pid=os.getpid(),started_at=stamp(),
        jobs=jobs,preparation=[],report_invocations=0)
    manifest=dict(schema=SCHEMA,source_sha256=sources,physical_gpus=[0,1],max_parallel_jobs=1,
        storage_required_bytes=STORAGE_BYTES,initialization_sha256={},seed=42,epochs=50,
        protected_seal_sha256=sha(PROTECTED),feature_widths=dict(global_only=1536,masked=3072,all_patch=3072),
        boundary='Pinned author reference, not new method. Joint global/var author RAW losses. Fresh process per arm, same topk/head/3072 in SIM pair; full-to-global mixes head/capacity/width. No2025/power/temperature action.')
    write(campaign/'manifest.json',manifest);write(campaign/'campaign.json',state)
    for dataset in DATASETS:
        for selection in SELECTIONS:
            step=dict(dataset=dataset,selection=selection,mode='prepare',command=command(campaign,dataset,selection,'prepare',ROOT/'trained-model/selection_prepare_unused'))
            state['preparation'].append(step)
            if run(campaign,state,step,f'prepare_{dataset}_{selection}.log'):
                state.update(status='FAILED',failed_preparation=step)
                write(campaign/'campaign.json',state);return 1
            witness=campaign/f'initialization/{dataset}_{selection}.json'
            manifest['initialization_sha256'][str(witness)]=sha(witness)
            write(campaign/'manifest.json',manifest)
        bindings={s:binding(campaign,dataset,s) for s in SELECTIONS}
        a,b=bindings['masked'],bindings['all_patch']
        for key in ('initial_model_state_sha256','trainable_parameters','trainable_parameter_tensors','cfg_yaml','selection_topk','feature_width'):
            assert a[key]==b[key],key
        for key in ('plain_foundation_binding','visual_initial_sha256','camera_initial_sha256','protocol_sha256','batch_size','num_instances'):
            assert all(bindings[s][key]==bindings['global_only'][key] for s in SELECTIONS),key
        write(campaign/f'paired_initialization_{dataset}.json',dict(status='MATCHED_STATE_VERIFIED',dataset=dataset,sim_initial_state_sha256=a['initial_model_state_sha256'],plain_state_sha256=a['plain_foundation_binding']['initial_model_state_sha256']))
        for selection in SELECTIONS:
            for phase in ('m0','full'):
                job=next(r for r in jobs if (r['dataset'],r['selection'],r['phase'])==(dataset,selection,phase))
                job['steps']=[]
                for mode in (('m0',) if phase=='m0' else ('train','evaluate')):
                    step=dict(mode=mode,command=command(campaign,dataset,selection,mode,output_dir(campaign,dataset,selection,phase)))
                    job['steps'].append(step)
                    if run(campaign,state,step,f'{dataset}_{selection}_{mode}.log'):
                        job.update(status='FAILED',exit_code=step['exit_code']);state.update(status='FAILED',failed_job=job)
                        write(campaign/'campaign.json',state);return 1
                if phase=='m0':
                    verify_m0(campaign,dataset,selection)
                else:
                    row=accepted_row(campaign,dataset,selection)
                    m0=verify_m0(campaign,dataset,selection)
                    probe=output_dir(campaign,dataset,selection,'m0')/'m0_reload_probe.pth'
                    retirement=dict(path=str(probe),sha256=sha(probe),bytes=probe.stat().st_size,at=stamp(),formal_receipt_sha256=row['receipt_sha256'])
                    assert retirement['sha256']==m0['m0']['reload_probe_sha256']
                    with (campaign/'retired_m0.jsonl').open('a') as stream:stream.write(json.dumps(retirement)+'\n')
                    probe.unlink()
                    job['result']=row
                job.update(status='COMPLETE',exit_code=0,completed_at=stamp())
                write(campaign/'campaign.json',state)
    require_sources();require_protected()
    rows=[accepted_row(campaign,d,s) for d in DATASETS for s in SELECTIONS]
    write(campaign/'accepted_matrix.json',dict(schema=SCHEMA,accepted=9,expected=9,rows=rows))
    state.update(status='COMPLETE',completed_at=stamp(),report_invocations=1)
    step=dict(command=[sys.executable,'-B',str(ROOT/'tools/report_signal_selection_reference.py'),'--campaign',str(campaign),'--output-dir',str(report)])
    state['report_step']=step
    with (campaign/'report.log').open('x') as log:
        result=subprocess.run(step['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode,report_completed_at=stamp())
    write(campaign/'campaign.json',state)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
