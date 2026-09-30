"""Read-only provenance, terminal stdout and diagnostic checks for integrity audit."""
import json,hashlib,math
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/data/gaob/Re-ID/Trifusion')
OUT=ROOT/'.codex_tmp/integrity678_supplement_v2.json'
assert not OUT.exists()
report={'checks':[],'hashes':{},'original_attempts':[],'slots':[],'trajectories':[],'started_at':datetime.now(timezone.utc).isoformat()}
def sha(p):
    p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();report['hashes'][str(p)]={'sha256':h,'bytes':p.stat().st_size};return h
def read(p):sha(p);return json.loads(Path(p).read_text())
def ck(n,b,detail=None):report['checks'].append({'name':n,'status':'PASS' if b else 'FAIL','details':detail})
def events(p):
    sha(p);return [json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
matrix=read(ROOT/'logs/patch_memory_roles_recovery_20261001/accepted_matrix.json')
independent=read(ROOT/'.codex_tmp/integrity678_cpu_evidence.json')
for row in matrix['rows']:
    tag=row['dataset']+'/'+row['variant'];run=Path(row['run_dir']); m0=Path(row['m0_run_dir']); child=Path(row['campaign_dir'])
    if row['recovery_action']=='evaluate_existing':
        train_child=ROOT/'logs/patch_memory_roles_20260930'/m0.name.removesuffix('_seed42_m0')
    else:train_child=child
    tr=read(run/'training.json');ev=read(run/'official_metrics.json')
    training_events=[e for e in events(train_child/'train.log') if e.get('event')=='epoch']
    ck(tag+'/stdout_training_history',len(training_events)==50 and all(all(e[k]==h[k] for k in h) for e,h in zip(training_events,tr['history'])))
    m0_events=[e for e in events(train_child/'m0.log') if e.get('event')=='epoch']
    ck(tag+'/stdout_M0',len(m0_events)==1 and m0_events[0]['steps']==8)
    ev_events=events(child/'evaluate.log')
    # The entry prints before its final architecture/memory fields are appended.
    ck(tag+'/stdout_evaluation',any(all(e.get(k)==v for k,v in ev.items() if k not in ('architecture','memory_mode')) for e in ev_events))
for folder in ('patch_memory_complete_intake_20261001','patch_memory_recovery_intake_20261001_0038','patch_memory_recovery_intake_20261001_0111','patch_memory_100_diagnosis_intake_20261001'):
    base=ROOT/'logs'/folder;intake=read(base/'INTAKE.json')
    refs=intake['files'] if 'files' in intake else intake['file_sha256']
    for rel,meta in refs.items():
        expected=meta['sha256'] if isinstance(meta,dict) else meta
        ck(folder+'/'+rel,sha(base/rel)==expected)
        if isinstance(meta,dict) and meta.get('source','').startswith('/data/'):
            ck(folder+'/'+rel+'/source_equal',sha(Path(meta['source']))==expected)
snap=read(ROOT/'logs/patch_memory_training_complete_20260930/SNAPSHOT.json')
for ds,members in snap['text_sha256'].items():
    folder={'MSVR310':'MSVR310_local','RGBNT201':'RGBNT201_full'}[ds]
    for rel,digest in members.items():
        ck('training_complete_snapshot/'+folder+'/'+rel,sha(ROOT/'logs/patch_memory_training_complete_20260930'/folder/rel)==digest)
for p in sorted((ROOT/'logs/patch_memory_roles_20260930').glob('*/campaign.json')):
    j=read(p);job=j['jobs'][-1];run=Path(job['output_dir']);t=read(run/'training.json');sha(run/'training_steps.jsonl');sha(p.parent/'train.log')
    steps=[json.loads(l) for l in (run/'training_steps.jsonl').read_text().splitlines()]
    report['original_attempts'].append({'dataset':j['dataset'],'variant':j['variant'],'campaign_status':j['status'],'last_job_status':job['status'],'last_job_exit_code':job.get('exit_code'),
       'training_status':t['status'],'completed_epoch_receipts':len(t['history']),'logged_steps':len(steps),'source':str(p),
       'M0_status':j['jobs'][0]['status'],'M0_exit_code':j['jobs'][0].get('exit_code')})
for ds in ('RGBNT201','RGBNT100','MSVR310'):
    pairpath=ROOT/(f'logs/patch_memory_pair_diagnosis_20261001/{ds}.json' if ds!='RGBNT100' else 'logs/patch_memory_100_diagnosis_20261001/cpu/RGBNT100.json')
    pair=read(pairpath);fresh=independent['pairwise'][ds]
    ck(ds+'/pair_ranking_independent',abs(pair['delta_metrics']['mAP']-fresh['delta_mAP_pp'])<1e-8 and abs(pair['delta_metrics']['Rank-1']-fresh['delta_R1_pp'])<1e-8
       and pair['rank1_repairs']==fresh['repairs'] and pair['rank1_new_errors']==fresh['new_errors'] and pair['query_ap_improved']==fresh['AP_improved'] and pair['query_ap_worsened']==fresh['AP_worsened'])
    slotdir=ROOT/(f'logs/patch_memory_slot_diagnosis_20261001/{ds}' if ds!='RGBNT100' else 'logs/patch_memory_100_diagnosis_20261001/slots')
    complete=read(slotdir/'COMPLETE.json')
    for row in [x for x in matrix['rows'] if x['dataset']==ds]:
        v=row['variant'];path=slotdir/(v+'.json');s=read(path)
        ck(ds+'/'+v+'/slot_report_hash',sha(path)==complete['endpoint_reports'][v])
        ck(ds+'/'+v+'/slot_source_checkpoint_protocol',s['source_sha256']==sha(ROOT/'tools/diagnose_patch_memory_slots.py') and s['checkpoint_sha256']==row['checkpoint_sha256'] and s['best_epoch']==row['best_epoch'] and s['protocol_sha256']==sha(ROOT/f'logs/official_three_dataset_protocols_20260923/{ds}.json'))
        ck(ds+'/'+v+'/slot_coverage_receipts',all(s['statistics'][split]['rows']==independent['datasets'][ds]['counts'][split] for split in ('query','gallery')) and s['model_state_unchanged'] and max(s['first_batch_instrumentation_max_difference'].values())<=1e-5 and max(s['full_gallery_metric_difference_pp'].values())<1e-5)
        names=s['statistic_names'];values=s['statistics']['query']['role_modality_mean']
        mean={name:sum(values[r][m][i] for r in range(3) for m in range(3))/9 for i,name in enumerate(names)}
        report['slots'].append({'dataset':ds,'variant':v,'query_mean':mean,'source':str(path),'status':'RECEIPT_AND_ARITHMETIC_VERIFIED_NOT_NEURAL_REPLAY'})
    if ds=='RGBNT100':
        campaign=read(ROOT/'logs/patch_memory_100_diagnosis_20261001/campaign.json')
        ck('RGBNT100/diagnostic_wait_exits',campaign['status']=='COMPLETE' and all(j['exit_code']==0 and j['status']=='COMPLETE' for j in campaign['jobs']))
        for rel,digest in campaign['source_sha256'].items():ck('RGBNT100/diagnostic_source/'+rel,sha(ROOT/rel)==digest)
        ck('RGBNT100/diagnostic_launcher',sha(ROOT/'.codex_tmp/launch_patch_memory_100_diagnosis_20261001.py')==campaign['launcher_sha256'])
        for rel in ('cpu.log','slots.log'):sha(ROOT/'logs/patch_memory_100_diagnosis_20261001'/rel)
    else:
        sha(ROOT/f'logs/patch_memory_slot_diagnosis_20261001/{ds}.log');sha(ROOT/f'logs/patch_memory_pair_diagnosis_20261001/{ds}.log')
for rel in ('logs/patch_memory_pair_diagnosis_20261001/FAILED_SOURCE_BINDING.json','logs/patch_memory_pair_diagnosis_20261001/COMPLETE.json','logs/patch_memory_pair_diagnosis_20261001/LAUNCH.json','logs/patch_memory_slot_diagnosis_20261001/LAUNCH.json','logs/patch_memory_100_diagnosis_launch_20261001.json','.codex_tmp/start_patch_memory_100_diagnosis_20261001.py'):
    sha(ROOT/rel)
traj=read(ROOT/'results/PATCH_MEMORY_ARCHIVED_TRAJECTORY_2026-10-01.json')
for row in traj['rows']:
    path=ROOT/row['training_receipt_path'];stepspath=ROOT/row['training_steps_path'];t=read(path);sha(stepspath)
    steps=[json.loads(l) for l in stepspath.read_text().splitlines()]
    best=t['history'][t['best_epoch']-1];last=t['history'][-1];bsteps=[s for s in steps if s['epoch']==best['epoch']];lsteps=[s for s in steps if s['epoch']==50]
    computed={'best_epoch':best['epoch'],'best_metrics':best['official_fused'],'final_epoch':50,'final_metrics':last['official_fused'],
        'final_minus_best_map_pp':last['official_fused']['mAP']-best['official_fused']['mAP'],'final_minus_best_rank1_pp':last['official_fused']['Rank-1']-best['official_fused']['Rank-1'],
        'best_mean_loss':best['mean_loss'],'final_mean_loss':last['mean_loss'],'best_mean_triplet':math.fsum(s['triplet'] for s in bsteps)/len(bsteps),'final_mean_triplet':math.fsum(s['triplet'] for s in lsteps)/len(lsteps),
        'total_steps':len(steps),'positive_triplet_steps':sum(s['triplet']>0 for s in steps),'after_best_steps':sum(s['epoch']>best['epoch'] for s in steps),'after_best_positive_triplet_steps':sum(s['epoch']>best['epoch'] and s['triplet']>0 for s in steps),'last_positive_triplet_epoch':max(s['epoch'] for s in steps if s['triplet']>0)}
    valid=sha(path)==row['training_receipt_sha256'] and sha(stepspath)==row['training_steps_sha256']
    valid &= all(abs(v-row[k])<1e-10 if isinstance(v,float) else v==row[k] for k,v in computed.items())
    ck(row['dataset']+'/'+row['variant']+'/trajectory',valid)
    report['trajectories'].append({'dataset':row['dataset'],'variant':row['variant'],**computed})
summary=read(ROOT/'results/patch_memory_complete_20261001/SUMMARY.json')
ck('summary/source_hash',sha(ROOT/'tools/report_patch_memory_complete.py')==summary['source_sha256'])
for rel,digest in summary['source_artifacts_sha256'].items():ck('summary/binding/'+rel,sha(ROOT/rel)==digest)
for rel,digest in summary['figure_sha256'].items():ck('summary/figure/'+rel,sha(ROOT/'results/patch_memory_complete_20261001'/rel)==digest)
for row in summary['rows']:
    source=next(r for r in matrix['rows'] if (r['dataset'],r['variant'])==(row['dataset'],row['variant']))
    t=read(Path(row['run_dir'])/'training.json');steps=[json.loads(l) for l in (Path(row['run_dir'])/'training_steps.jsonl').read_text().splitlines()]
    best=t['history'][row['best_epoch']-1];last=t['history'][-1]
    valid=all(row[k]==v for k,v in source.items()) and row['final_metrics']==last['official_fused']
    valid &= abs(row['final_minus_best_map_pp']-(last['official_fused']['mAP']-best['official_fused']['mAP']))<1e-10
    valid &= abs(row['final_minus_best_rank1_pp']-(last['official_fused']['Rank-1']-best['official_fused']['Rank-1']))<1e-10
    valid &= abs(row['best_mean_loss']-best['mean_loss'])<1e-10 and abs(row['final_mean_loss']-last['mean_loss'])<1e-10
    valid &= row['after_best_steps']==sum(s['epoch']>row['best_epoch'] for s in steps) and row['after_best_positive_triplet_steps']==sum(s['epoch']>row['best_epoch'] and s['triplet']>0 for s in steps)
    ck('summary/row/'+row['dataset']+'/'+row['variant'],valid)
for pair in summary['pairs']:
    ds=pair['dataset'];fresh=independent['pairwise'][ds]
    ck('summary/pair/'+ds,abs(pair['delta_metrics']['mAP']-fresh['delta_mAP_pp'])<1e-8 and abs(pair['delta_metrics']['Rank-1']-fresh['delta_R1_pp'])<1e-8 and pair['registered_gate_passed']==(fresh['delta_mAP_pp']>0 and fresh['delta_mAP_pp']>=pair['required_positive_map_pp'] and fresh['delta_R1_pp']>=0))
for slot in summary['slot_statistics']:
    checked=next(s for s in report['slots'] if (s['dataset'],s['variant'])==(slot['dataset'],slot['mode']))
    ck('summary/slot/'+slot['dataset']+'/'+slot['mode'],all(abs(v-checked['query_mean'][k])<1e-10 for k,v in slot['query_role_modality_mean'].items()))
ck('summary/final_gate',summary['registered_advancement_gate']=='FAIL' and independent['gate_pass'] is False)
report['completed_at']=datetime.now(timezone.utc).isoformat();report['status']='COMPLETE' if all(c['status']=='PASS' for c in report['checks']) else 'COMPLETED_WITH_FAILURES'
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(report['checks']),'failures':[c for c in report['checks'] if c['status']!='PASS'],'original_attempts':report['original_attempts']}))
