"""Train five missing endpoints after the recorded metadata-only counter repair."""
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
PREVIOUS=ROOT/'logs/signal_selection_reference_v1_20261006_867'
NORMALIZATION=ROOT/'logs/selection_counter_record_normalization_20261006_868'
STORAGE_BYTES=6*360*1024**2+2*1024**3

def origin(campaign,dataset,selection,phase):
    if dataset=='RGBNT201':
        if selection=='global_only' or selection=='masked' and phase=='m0':return ORIGINAL
        return PREVIOUS
    if dataset=='MSVR310' and selection=='global_only':return PREVIOUS
    return campaign

def metadata_path(campaign,dataset,selection):
    if dataset=='MSVR310' and selection=='global_only':return NORMALIZATION/'MSVR310_optimizer_batch_metadata.jsonl'
    return panel.output_dir(origin(campaign,dataset,selection,'full'),dataset,selection,'full')/'training_batch_metadata.jsonl'

def accepted_row(campaign,dataset,selection):
    return panel.accepted_row(origin(campaign,dataset,selection,'full'),dataset,selection,batch_metadata_path=metadata_path(campaign,dataset,selection))

def reused_m0(dataset,selection):
    source=origin(None,dataset,selection,'m0')
    folder=panel.output_dir(source,dataset,selection,'m0')
    result=json.loads((folder/'training.json').read_text())
    assert result['schema']==panel.SCHEMA and result['status']=='M0_PASS'
    assert result['initializer']==panel.binding(source,dataset,selection)
    assert len(result['history'])==1 and result['history'][0]['steps']==8
    assert result['m0']['nonzero_gradient_parameters']==result['m0']['trainable_parameters']
    assert result['m0']['reload_max_abs_difference']<=1e-5
    assert all(v==8 for v in result['selection_reference']['m0_bn_counts'].values())
    assert len((folder/'training_batch_metadata.jsonl').read_text().splitlines())==8
    probe=folder/'m0_reload_probe.pth'
    if dataset=='MSVR310':
        assert panel.verify_m0(source,dataset,selection)==result
    else:
        ledger=ORIGINAL if selection=='global_only' else PREVIOUS
        retired=next(r for r in map(json.loads,(ledger/'retired_m0.jsonl').read_text().splitlines()) if r['path']==str(probe))
        assert not probe.exists() and retired['sha256']==result['m0']['reload_probe_sha256']
        full_origin=origin(None,dataset,selection,'full')
        assert retired['formal_receipt_sha256']==panel.sha(panel.output_dir(full_origin,dataset,selection,'full')/'official_metrics.json')
    return result

def retire_probe(campaign,dataset,selection,m0_source,row,m0):
    probe=panel.output_dir(m0_source,dataset,selection,'m0')/'m0_reload_probe.pth'
    retirement=dict(path=str(probe),sha256=panel.sha(probe),bytes=probe.stat().st_size,at=panel.stamp(),formal_receipt_sha256=row['receipt_sha256'],m0_origin_campaign=str(m0_source))
    assert retirement['sha256']==m0['m0']['reload_probe_sha256']
    with (campaign/'retired_m0.jsonl').open('a') as stream:stream.write(json.dumps(retirement)+'\n')
    probe.unlink()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args();campaign,report=args.campaign.resolve(),args.report_dir.resolve()
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    assert not campaign.exists() and not report.exists()
    sources=panel.require_sources();panel.require_protected()
    assert json.loads((ROOT/'logs/signal_selection_reference_launch_20261006_867/EXIT.json').read_text())['exit_code']==1
    old=json.loads((PREVIOUS/'campaign.json').read_text())
    assert old['report_invocations']==0
    failed=next(r for r in old['jobs'] if (r['dataset'],r['selection'],r['phase'])==('MSVR310','global_only','full'))
    assert failed['status']=='PENDING' and len(failed['steps'])==2 and all(r['exit_code']==0 for r in failed['steps'])
    normalization=json.loads((NORMALIZATION/'PROVENANCE.json').read_text())
    assert normalization['status']=='TRAIN_EVALUATION_PHASES_VERIFIED' and normalization['training_rows']==706 and normalization['evaluation_rows']==1350
    for path,digest in normalization['artifact_sha256'].items():assert panel.sha(path)==digest
    reused=[(d,s) for d in panel.DATASETS for s in panel.SELECTIONS if origin(campaign,d,s,'full')!=campaign]
    assert len(reused)==4
    old_m0={(d,s):reused_m0(d,s) for d,s in reused}
    retained={(d,s):accepted_row(campaign,d,s) for d,s in reused}
    campaign.mkdir(parents=True)
    jobs=[dict(dataset=d,selection=s,phase=p,status='PENDING') for d in panel.DATASETS for s in panel.SELECTIONS for p in ('m0','full')]
    state=dict(schema=panel.SCHEMA,status='RUNNING',controller_pid=os.getpid(),started_at=panel.stamp(),jobs=jobs,preparation=[],report_invocations=0,reused_formal=4,new_formal_expected=5)
    origins={f'{d}:{s}':str(origin(campaign,d,s,'full')) for d in panel.DATASETS for s in panel.SELECTIONS}
    batch_paths={f'{d}:{s}':str(metadata_path(campaign,d,s)) for d in panel.DATASETS for s in panel.SELECTIONS}
    panel.write(campaign/'endpoint_origins.json',origins);panel.write(campaign/'batch_metadata_paths.json',batch_paths)
    manifest=dict(schema=panel.SCHEMA,source_sha256=sources,physical_gpus=[0,1],max_parallel_jobs=1,storage_required_bytes=STORAGE_BYTES,
        initialization_sha256={},seed=42,epochs=50,protected_seal_sha256=panel.sha(panel.PROTECTED),endpoint_origins=origins,batch_metadata_paths=batch_paths,
        original_failed_campaign_sha256=panel.sha(PREVIOUS/'campaign.json'),original_failed_exit_sha256=panel.sha(ROOT/'logs/signal_selection_reference_launch_20261006_867/EXIT.json'),
        normalization_provenance_sha256=panel.sha(NORMALIZATION/'PROVENANCE.json'),
        boundary='Only training-phase metadata logger fixed. Reuse four full50/firststrict and four M0; fresh bindings must agree except logging entrySHA. Explicit canonical706 MSVR metadata; original2056/stale status/EXIT1 remain unchanged. Five new8M0/full50. No model/recipe/loss/precision/seed/topk/filter/tolerance change.')
    panel.write(campaign/'manifest.json',manifest);panel.write(campaign/'campaign.json',state)
    for dataset in panel.DATASETS:
        for selection in panel.SELECTIONS:
            step=dict(dataset=dataset,selection=selection,mode='prepare',command=panel.command(campaign,dataset,selection,'prepare',ROOT/'trained-model/selection_prepare_unused'))
            state['preparation'].append(step)
            if panel.run(campaign,state,step,f'prepare_{dataset}_{selection}.log'):
                state.update(status='FAILED',failed_preparation=step);panel.write(campaign/'campaign.json',state);return 1
            witness=campaign/f'initialization/{dataset}_{selection}.json'
            manifest['initialization_sha256'][str(witness)]=panel.sha(witness);panel.write(campaign/'manifest.json',manifest)
        bindings={s:panel.binding(campaign,dataset,s) for s in panel.SELECTIONS}
        a,b=bindings['masked'],bindings['all_patch']
        for key in ('initial_model_state_sha256','trainable_parameters','trainable_parameter_tensors','cfg_yaml','selection_topk','feature_width'):assert a[key]==b[key],key
        for key in ('plain_foundation_binding','visual_initial_sha256','camera_initial_sha256','protocol_sha256','batch_size','num_instances'):
            assert all(bindings[s][key]==bindings['global_only'][key] for s in panel.SELECTIONS),key
        if dataset in ('RGBNT201','MSVR310'):
            comparisons={}
            for selection in panel.SELECTIONS:
                source=origin(campaign,dataset,selection,'full') if (dataset,selection) in retained else PREVIOUS
                before=panel.binding(source,dataset,selection)
                assert {k:v for k,v in before.items() if k!='entry_sha256'}=={k:v for k,v in bindings[selection].items() if k!='entry_sha256'}
                comparisons[selection]=dict(old=before,new=bindings[selection],source_campaign=str(source))
            panel.write(campaign/f'initializer_reuse_proof_{dataset}.json',dict(status='EXACT_ALL_BINDING_FIELDS_EXCEPT_LOGGING_ENTRY_SHA',comparisons=comparisons))
        panel.write(campaign/f'paired_initialization_{dataset}.json',dict(status='MATCHED_STATE_VERIFIED',dataset=dataset,sim_initial_state_sha256=a['initial_model_state_sha256'],plain_state_sha256=a['plain_foundation_binding']['initial_model_state_sha256']))
    for dataset,selection in reused:
        for phase in ('m0','full'):
            job=next(r for r in jobs if (r['dataset'],r['selection'],r['phase'])==(dataset,selection,phase))
            source=origin(campaign,dataset,selection,phase)
            old_state=json.loads((source/'campaign.json').read_text())
            previous=next(r for r in old_state['jobs'] if (r['dataset'],r['selection'],r['phase'])==(dataset,selection,phase))
            job.update(copy.deepcopy(previous),status='COMPLETE',exit_code=0,source_campaign=str(source),reuse_no_execution=True,reused_at=panel.stamp())
            if phase=='full':job['result']=retained[(dataset,selection)]
            panel.write(campaign/'campaign.json',state)
    panel.write(campaign/'administrative_reuse_acceptance.json',dict(status='VERIFIED_FOUR_EXISTING_FULL50',rows=list(retained.values()),normalization_provenance_sha256=manifest['normalization_provenance_sha256'],no_new_training_or_evaluation=True))
    retire_probe(campaign,'MSVR310','global_only',PREVIOUS,retained[('MSVR310','global_only')],old_m0[('MSVR310','global_only')])
    assert shutil.disk_usage(ROOT).free>=STORAGE_BYTES
    panel.write(campaign/'storage_after_reuse.json',dict(required_bytes=STORAGE_BYTES,free_bytes=shutil.disk_usage(ROOT).free,retired_old_probe_only_after_reuse=True))
    for dataset in panel.DATASETS:
        for selection in panel.SELECTIONS:
            if (dataset,selection) in retained:continue
            for phase in ('m0','full'):
                job=next(r for r in jobs if (r['dataset'],r['selection'],r['phase'])==(dataset,selection,phase));job['steps']=[]
                for mode in (('m0',) if phase=='m0' else ('train','evaluate')):
                    step=dict(mode=mode,command=panel.command(campaign,dataset,selection,mode,panel.output_dir(campaign,dataset,selection,phase)));job['steps'].append(step)
                    if panel.run(campaign,state,step,f'{dataset}_{selection}_{mode}.log'):
                        job.update(status='FAILED',exit_code=step['exit_code']);state.update(status='FAILED',failed_job=job);panel.write(campaign/'campaign.json',state);return 1
                if phase=='m0':panel.verify_m0(campaign,dataset,selection)
                else:
                    row=accepted_row(campaign,dataset,selection);m0=panel.verify_m0(campaign,dataset,selection)
                    retire_probe(campaign,dataset,selection,campaign,row,m0);job['result']=row
                job.update(status='COMPLETE',exit_code=0,completed_at=panel.stamp());panel.write(campaign/'campaign.json',state)
    panel.require_sources();panel.require_protected()
    rows=[accepted_row(campaign,d,s) for d in panel.DATASETS for s in panel.SELECTIONS]
    panel.write(campaign/'accepted_matrix.json',dict(schema=panel.SCHEMA,accepted=9,expected=9,rows=rows))
    state.update(status='COMPLETE',completed_at=panel.stamp(),report_invocations=1)
    step=dict(command=[sys.executable,'-B',str(ROOT/'tools/report_signal_selection_reference.py'),'--campaign',str(campaign),'--origin-campaigns',str(campaign/'endpoint_origins.json'),'--batch-metadata-paths',str(campaign/'batch_metadata_paths.json'),'--output-dir',str(report)])
    state['report_step']=step
    with (campaign/'report.log').open('x') as log:
        result=subprocess.run(step['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode,report_completed_at=panel.stamp());panel.write(campaign/'campaign.json',state)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
