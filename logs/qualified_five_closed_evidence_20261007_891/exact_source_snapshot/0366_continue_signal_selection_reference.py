"""Evaluation-width revision: reuse accepted RGBNT201 work, run eight missing full endpoints."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_signal_selection_reference as panel

ORIGINAL=ROOT/'logs/signal_selection_reference_v1_20261006_866'

def origin(campaign,dataset,selection,phase):
    if dataset=='RGBNT201' and (selection=='global_only' or selection=='masked' and phase=='m0'):
        return ORIGINAL
    return campaign

def accepted_row(campaign,dataset,selection):
    return panel.accepted_row(origin(campaign,dataset,selection,'full'),dataset,selection)

def original_m0(selection):
    folder=panel.output_dir(ORIGINAL,'RGBNT201',selection,'m0')
    result=json.loads((folder/'training.json').read_text())
    assert result['status']=='M0_PASS' and result['initializer']==panel.binding(ORIGINAL,'RGBNT201',selection)
    assert result['history'][0]['steps']==8 and len(result['history'])==1
    assert result['m0']['nonzero_gradient_parameters']==result['m0']['trainable_parameters']
    assert result['m0']['reload_max_abs_difference']<=1e-5
    assert all(v==8 for v in result['selection_reference']['m0_bn_counts'].values())
    assert len((folder/'training_batch_metadata.jsonl').read_text().splitlines())==8
    probe=folder/'m0_reload_probe.pth'
    if selection=='global_only':
        rows=[json.loads(line) for line in (ORIGINAL/'retired_m0.jsonl').read_text().splitlines()]
        retired=next(row for row in rows if row['path']==str(probe))
        assert not probe.exists() and retired['sha256']==result['m0']['reload_probe_sha256']
        assert retired['formal_receipt_sha256']==panel.sha(panel.output_dir(ORIGINAL,'RGBNT201',selection,'full')/'official_metrics.json')
    else:
        assert panel.verify_m0(ORIGINAL,'RGBNT201',selection)==result
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    campaign,report=args.campaign.resolve(),args.report_dir.resolve()
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    assert not campaign.exists() and not report.exists()
    assert shutil.disk_usage(ROOT).free>=panel.STORAGE_BYTES
    sources=panel.require_sources();panel.require_protected()
    old=json.loads((ORIGINAL/'campaign.json').read_text())
    assert old['status']=='FAILED' and old['report_invocations']==0
    assert old['failed_job']['dataset']=='RGBNT201' and old['failed_job']['selection']=='masked' and old['failed_job']['phase']=='full'
    assert json.loads((ROOT/'logs/signal_selection_reference_launch_20261006_866/EXIT.json').read_text())['exit_code']==1
    retained=panel.accepted_row(ORIGINAL,'RGBNT201','global_only')
    for selection in ('global_only','masked'):original_m0(selection)
    campaign.mkdir(parents=True)
    jobs=[dict(dataset=d,selection=s,phase=p,status='PENDING') for d in panel.DATASETS for s in panel.SELECTIONS for p in ('m0','full')]
    state=dict(schema=panel.SCHEMA,status='RUNNING',controller_pid=os.getpid(),started_at=panel.stamp(),
               jobs=jobs,preparation=[],report_invocations=0,original_campaign=str(ORIGINAL),reused_formal=1,new_formal_expected=8)
    origins={f'{d}:{s}':str(origin(campaign,d,s,'full')) for d in panel.DATASETS for s in panel.SELECTIONS}
    panel.write(campaign/'endpoint_origins.json',origins)
    manifest=dict(schema=panel.SCHEMA,source_sha256=sources,physical_gpus=[0,1],max_parallel_jobs=1,
        storage_required_bytes=panel.STORAGE_BYTES,initialization_sha256={},seed=42,epochs=50,
        protected_seal_sha256=panel.sha(panel.PROTECTED),feature_widths=dict(global_only=1536,masked=3072,all_patch=3072),
        original_campaign_sha256=panel.sha(ORIGINAL/'campaign.json'),original_launch_exit_sha256=panel.sha(ROOT/'logs/signal_selection_reference_launch_20261006_866/EXIT.json'),
        retained_global=retained,endpoint_origins=origins,
        boundary='Evaluation-only1536/3072 fix. Reuse original accepted global50 and two accepted M0s after exact fresh binding comparison except entrySHA. Failed masked53updates kept, fresh50 not optimizer resume. No recipe/model/loss/seed/topk/precision change.')
    panel.write(campaign/'manifest.json',manifest);panel.write(campaign/'campaign.json',state)
    for dataset in panel.DATASETS:
        for selection in panel.SELECTIONS:
            step=dict(dataset=dataset,selection=selection,mode='prepare',command=panel.command(campaign,dataset,selection,'prepare',ROOT/'trained-model/selection_prepare_unused'))
            state['preparation'].append(step)
            if panel.run(campaign,state,step,f'prepare_{dataset}_{selection}.log'):
                state.update(status='FAILED',failed_preparation=step);panel.write(campaign/'campaign.json',state);return 1
            witness=campaign/f'initialization/{dataset}_{selection}.json'
            manifest['initialization_sha256'][str(witness)]=panel.sha(witness)
            panel.write(campaign/'manifest.json',manifest)
        bindings={s:panel.binding(campaign,dataset,s) for s in panel.SELECTIONS}
        a,b=bindings['masked'],bindings['all_patch']
        for key in ('initial_model_state_sha256','trainable_parameters','trainable_parameter_tensors','cfg_yaml','selection_topk','feature_width'):
            assert a[key]==b[key],key
        for key in ('plain_foundation_binding','visual_initial_sha256','camera_initial_sha256','protocol_sha256','batch_size','num_instances'):
            assert all(bindings[s][key]==bindings['global_only'][key] for s in panel.SELECTIONS),key
        if dataset=='RGBNT201':
            for selection in panel.SELECTIONS:
                before=panel.binding(ORIGINAL,dataset,selection)
                assert {k:v for k,v in before.items() if k!='entry_sha256'}=={k:v for k,v in bindings[selection].items() if k!='entry_sha256'}
            panel.write(campaign/'initializer_reuse_proof.json',dict(status='EXACT_ALL_BINDING_FIELDS_EXCEPT_ENTRY_SHA',old={s:panel.binding(ORIGINAL,dataset,s) for s in panel.SELECTIONS},new=bindings,accepted_global=retained))
        panel.write(campaign/f'paired_initialization_{dataset}.json',dict(status='MATCHED_STATE_VERIFIED',dataset=dataset,sim_initial_state_sha256=a['initial_model_state_sha256'],plain_state_sha256=a['plain_foundation_binding']['initial_model_state_sha256']))
        for selection in panel.SELECTIONS:
            for phase in ('m0','full'):
                job=next(r for r in jobs if (r['dataset'],r['selection'],r['phase'])==(dataset,selection,phase))
                source=origin(campaign,dataset,selection,phase)
                if source==ORIGINAL:
                    original=next(r for r in old['jobs'] if (r['dataset'],r['selection'],r['phase'])==(dataset,selection,phase))
                    assert original['status']=='COMPLETE' and original['exit_code']==0
                    job.update(copy.deepcopy(original),source_campaign=str(ORIGINAL),reuse_no_execution=True,reused_at=panel.stamp())
                    panel.write(campaign/'campaign.json',state);continue
                job['steps']=[]
                for mode in (('m0',) if phase=='m0' else ('train','evaluate')):
                    step=dict(mode=mode,command=panel.command(campaign,dataset,selection,mode,panel.output_dir(campaign,dataset,selection,phase)))
                    job['steps'].append(step)
                    if panel.run(campaign,state,step,f'{dataset}_{selection}_{mode}.log'):
                        job.update(status='FAILED',exit_code=step['exit_code']);state.update(status='FAILED',failed_job=job)
                        panel.write(campaign/'campaign.json',state);return 1
                if phase=='m0':panel.verify_m0(campaign,dataset,selection)
                else:
                    row=accepted_row(campaign,dataset,selection)
                    m0_origin=origin(campaign,dataset,selection,'m0')
                    m0=panel.verify_m0(m0_origin,dataset,selection)
                    probe=panel.output_dir(m0_origin,dataset,selection,'m0')/'m0_reload_probe.pth'
                    retirement=dict(path=str(probe),sha256=panel.sha(probe),bytes=probe.stat().st_size,at=panel.stamp(),formal_receipt_sha256=row['receipt_sha256'],m0_origin_campaign=str(m0_origin))
                    assert retirement['sha256']==m0['m0']['reload_probe_sha256']
                    with (campaign/'retired_m0.jsonl').open('a') as stream:stream.write(json.dumps(retirement)+'\n')
                    probe.unlink();job['result']=row
                    if dataset=='RGBNT201' and selection=='masked':
                        old_batches=(panel.output_dir(ORIGINAL,dataset,selection,'full')/'training_batch_metadata.jsonl').read_bytes()
                        new_batches=(Path(row['run_dir'])/'training_batch_metadata.jsonl').read_bytes().splitlines(keepends=True)
                        assert old_batches==b''.join(new_batches[:53])
                        panel.write(campaign/'failed_first_epoch_batch_reuse.json',dict(status='FIRST53_METADATA_EQUAL',old_failed_steps=53,not_optimizer_resume=True,boundary='Labels/camera/view/RGB basename metadata, not augmentation-byte equality.'))
                job.update(status='COMPLETE',exit_code=0,completed_at=panel.stamp())
                panel.write(campaign/'campaign.json',state)
    panel.require_sources();panel.require_protected()
    rows=[accepted_row(campaign,d,s) for d in panel.DATASETS for s in panel.SELECTIONS]
    panel.write(campaign/'accepted_matrix.json',dict(schema=panel.SCHEMA,accepted=9,expected=9,rows=rows))
    state.update(status='COMPLETE',completed_at=panel.stamp(),report_invocations=1)
    step=dict(command=[sys.executable,'-B',str(ROOT/'tools/report_signal_selection_reference.py'),'--campaign',str(campaign),'--origin-campaigns',str(campaign/'endpoint_origins.json'),'--output-dir',str(report)])
    state['report_step']=step
    with (campaign/'report.log').open('x') as log:
        result=subprocess.run(step['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode,report_completed_at=panel.stamp())
    panel.write(campaign/'campaign.json',state)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
