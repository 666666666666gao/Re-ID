"""Reuse seven accepted references and train only two disk-blocked endpoints."""
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
from tools.resume_signal_selection_reference import retire_probe

ORIGINAL=ROOT/'logs/signal_selection_reference_v1_20261006_866'
PREVIOUS=ROOT/'logs/signal_selection_reference_v1_20261006_867'
STOPPED=ROOT/'logs/signal_selection_reference_v1_20261006_868'
STOPPED_LAUNCH=ROOT/'logs/signal_selection_reference_launch_20261006_868'
OVERLAY=ROOT/'refine-logs/signal_selection_disk_continuation_v1/SOURCE_SCOPE.json'
STORAGE_BYTES=2*1024**3+3*360*1024**2+6*59_074_461

def origin(campaign,dataset,selection,phase):
    if dataset=='RGBNT201':
        if selection=='global_only' or selection=='masked' and phase=='m0':return ORIGINAL
        return PREVIOUS
    if dataset=='MSVR310' and selection=='global_only':return PREVIOUS
    if dataset=='RGBNT100' and (selection=='all_patch' or selection=='masked' and phase=='full'):return campaign
    return STOPPED

def metadata_path(campaign,dataset,selection):
    if origin(campaign,dataset,selection,'full')==campaign:
        return panel.output_dir(campaign,dataset,selection,'full')/'training_batch_metadata.jsonl'
    return Path(json.loads((STOPPED/'batch_metadata_paths.json').read_text())[f'{dataset}:{selection}'])

def accepted_row(campaign,dataset,selection):
    return panel.accepted_row(origin(campaign,dataset,selection,'full'),dataset,selection,
                              batch_metadata_path=metadata_path(campaign,dataset,selection))

def reuse_m0(campaign,dataset,selection):
    source=origin(campaign,dataset,selection,'m0')
    folder=panel.output_dir(source,dataset,selection,'m0')
    result=json.loads((folder/'training.json').read_text())
    assert result['schema']==panel.SCHEMA and result['status']=='M0_PASS'
    assert result['initializer']==panel.binding(source,dataset,selection)
    assert {k:v for k,v in result['initializer'].items() if k!='entry_sha256'}=={
        k:v for k,v in panel.binding(campaign,dataset,selection).items() if k!='entry_sha256'}
    assert len(result['history'])==1 and result['history'][0]['steps']==8
    assert result['m0']['nonzero_gradient_parameters']==result['m0']['trainable_parameters']
    assert result['m0']['reload_max_abs_difference']<=1e-5
    assert all(v==8 for v in result['selection_reference']['m0_bn_counts'].values())
    assert len((folder/'training_batch_metadata.jsonl').read_text().splitlines())==8
    probe=folder/'m0_reload_probe.pth'
    if dataset=='RGBNT100' and selection=='masked':
        assert panel.verify_m0(source,dataset,selection)==result
    else:
        retired=[r for ledger in (ORIGINAL,PREVIOUS,STOPPED)
                 for r in map(json.loads,(ledger/'retired_m0.jsonl').read_text().splitlines())
                 if r['path']==str(probe)]
        assert len(retired)==1 and not probe.exists()
        assert retired[0]['sha256']==result['m0']['reload_probe_sha256']
        assert retired[0]['formal_receipt_sha256']==accepted_row(campaign,dataset,selection)['receipt_sha256']
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args();campaign,report=args.campaign.resolve(),args.report_dir.resolve()
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    assert not campaign.exists() and not report.exists()
    assert shutil.disk_usage(ROOT).free>=STORAGE_BYTES
    sources=panel.require_sources();panel.require_protected()
    overlay=json.loads(OVERLAY.read_text())['source_sha256']
    assert all(panel.sha(ROOT/n)==d for n,d in overlay.items())
    stopped=json.loads((STOPPED/'campaign.json').read_text())
    assert json.loads((STOPPED_LAUNCH/'EXIT.json').read_text())['exit_code']==1
    assert stopped['report_invocations']==0
    for name in ('LAUNCH.json','CHILD.json'):
        r=json.loads((STOPPED_LAUNCH/name).read_text());p=Path('/proc')/str(r['pid'])
        assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
    assert sum(j['phase']=='full' and j['status']=='COMPLETE' for j in stopped['jobs'])==7
    assert sum(j['phase']=='m0' and j['status']=='COMPLETE' for j in stopped['jobs'])==8
    for selection in ('masked','all_patch'):
        j=next(j for j in stopped['jobs'] if (j['dataset'],j['selection'],j['phase'])==('RGBNT100',selection,'full'))
        assert j['status']=='PENDING' and not j['steps']
    old_manifest=json.loads((STOPPED/'manifest.json').read_text())
    campaign.mkdir(parents=True);(campaign/'initialization').mkdir()
    initialization_sha={}
    for dataset in panel.DATASETS:
        for selection in panel.SELECTIONS:
            before=STOPPED/f'initialization/{dataset}_{selection}.json'
            assert panel.sha(before)==old_manifest['initialization_sha256'][str(before)]
            after=campaign/f'initialization/{dataset}_{selection}.json'
            shutil.copyfile(before,after);assert panel.sha(after)==panel.sha(before)
            initialization_sha[str(after)]=panel.sha(after)
    jobs=[];m0={};retained={}
    for dataset in panel.DATASETS:
        for selection in panel.SELECTIONS:
            for phase in ('m0','full'):
                source=origin(campaign,dataset,selection,phase)
                job=dict(dataset=dataset,selection=selection,phase=phase,status='PENDING',steps=[])
                if source!=campaign:
                    old=json.loads((source/'campaign.json').read_text())
                    prior=next(j for j in old['jobs'] if (j['dataset'],j['selection'],j['phase'])==(dataset,selection,phase))
                    job.update(copy.deepcopy(prior),status='COMPLETE',exit_code=0,source_campaign=str(source),reuse_no_execution=True,reused_at=panel.stamp())
                    if phase=='m0':m0[(dataset,selection)]=reuse_m0(campaign,dataset,selection)
                    else:
                        retained[(dataset,selection)]=accepted_row(campaign,dataset,selection)
                        job['result']=retained[(dataset,selection)]
                jobs.append(job)
    assert len(retained)==7 and len(m0)==8
    origins={f'{d}:{s}':str(origin(campaign,d,s,'full')) for d in panel.DATASETS for s in panel.SELECTIONS}
    batch_paths={f'{d}:{s}':str(metadata_path(campaign,d,s)) for d in panel.DATASETS for s in panel.SELECTIONS}
    panel.write(campaign/'endpoint_origins.json',origins);panel.write(campaign/'batch_metadata_paths.json',batch_paths)
    panel.write(campaign/'manifest.json',dict(schema=panel.SCHEMA,source_sha256=sources,continuation_source_sha256=overlay,
        physical_gpus=[0,1],max_parallel_jobs=1,storage_required_bytes=STORAGE_BYTES,initialization_sha256=initialization_sha,
        seed=42,epochs=50,protected_seal_sha256=panel.sha(panel.PROTECTED),endpoint_origins=origins,batch_metadata_paths=batch_paths,
        stopped_exit_sha256=panel.sha(STOPPED_LAUNCH/'EXIT.json'),stopped_manifest_sha256=panel.sha(STOPPED/'manifest.json'),
        boundary='Disk-only continuation: reuse seven full50/firststrict, eight M0 and nine actual initialization witnesses. Only masked/all_patch RGBNT100 fresh50 and one all_patch M0; no probe used as initializer. Original868 disk EXIT1/stale flags and all former failures unchanged. Original374 scope and metadata-path supplement unchanged. No training/model/recipe/precision/seed/filter/tolerance change. Storage budget includes three360MiB save/probe slots, six known-size RGBNT100 distance arrays and unchanged2GiB reserve; conditional on other filesystem usage.'))
    panel.write(campaign/'reuse_acceptance.json',dict(status='VERIFIED_SEVEN_FULL_EIGHT_M0',rows=list(retained.values()),
        m0_origins={f'{d}:{s}':str(origin(campaign,d,s,'m0')) for d,s in m0},no_new_training_or_evaluation=True))
    state=dict(schema=panel.SCHEMA,status='RUNNING',controller_pid=os.getpid(),started_at=panel.stamp(),jobs=jobs,
        preparation=[],report_invocations=0,reused_formal=7,new_formal_expected=2,reused_m0=8,new_m0_expected=1)
    panel.write(campaign/'campaign.json',state)
    for selection in ('masked','all_patch'):
        for phase in ('m0','full'):
            job=next(j for j in jobs if (j['dataset'],j['selection'],j['phase'])==('RGBNT100',selection,phase))
            if job['status']=='COMPLETE':continue
            for mode in (('m0',) if phase=='m0' else ('train','evaluate')):
                step=dict(mode=mode,command=panel.command(campaign,'RGBNT100',selection,mode,
                          panel.output_dir(campaign,'RGBNT100',selection,phase)))
                job['steps'].append(step)
                if panel.run(campaign,state,step,f'RGBNT100_{selection}_{mode}.log'):
                    job.update(status='FAILED',exit_code=step['exit_code']);state.update(status='FAILED',failed_job=job)
                    panel.write(campaign/'campaign.json',state);return 1
            if phase=='m0':m0[('RGBNT100',selection)]=panel.verify_m0(campaign,'RGBNT100',selection)
            else:
                row=accepted_row(campaign,'RGBNT100',selection)
                retire_probe(campaign,'RGBNT100',selection,origin(campaign,'RGBNT100',selection,'m0'),row,m0[('RGBNT100',selection)])
                job['result']=row
            job.update(status='COMPLETE',exit_code=0,completed_at=panel.stamp());panel.write(campaign/'campaign.json',state)
    panel.require_sources();panel.require_protected()
    assert all(panel.sha(ROOT/n)==d for n,d in overlay.items())
    rows=[accepted_row(campaign,d,s) for d in panel.DATASETS for s in panel.SELECTIONS]
    panel.write(campaign/'accepted_matrix.json',dict(schema=panel.SCHEMA,accepted=9,expected=9,rows=rows))
    state.update(status='COMPLETE',completed_at=panel.stamp(),report_invocations=1)
    step=dict(command=[sys.executable,'-B',str(ROOT/'tools/report_signal_selection_reference.py'),
        '--campaign',str(campaign),'--origin-campaigns',str(campaign/'endpoint_origins.json'),
        '--batch-metadata-paths',str(campaign/'batch_metadata_paths.json'),'--output-dir',str(report)])
    state['report_step']=step
    with (campaign/'report.log').open('x') as log:
        result=subprocess.run(step['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode,report_completed_at=panel.stamp());panel.write(campaign/'campaign.json',state)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
