"""CPU-only full50/first-strict diagnosis for one new formation endpoint."""
import argparse,json,math,sys
from datetime import datetime
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository',type=Path,required=True)
parser.add_argument('--exposure',choices=('gradient','random'),required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
root=args.repository.resolve()
sys.path.insert(0,str(root))
import numpy as np
import torch
from tools.report_prepool_dense_correspondence import paired,descriptor
from tools.queue_region_use_probe import base

campaign=root/'logs/global_conditioned_evidence_full_20261013_926'
inputs=json.loads((campaign/'INPUTS.json').read_text())
assert all(base.sha(root/n)==h for n,h in inputs['runtime_source_sha256'].items())
assert all(base.sha(root/n)==row['sha256'] for n,row in inputs['m0_file_sha256'].items())
state=json.loads((campaign/'campaign.json').read_text())
accepted=next(row for row in state['accepted_endpoints'] if row['exposure']==args.exposure)
stages=[s for s in state['stages'] if s['exposure']==args.exposure and 'mode' in s]
assert [s['mode'] for s in stages]==['train','evaluate'] and all(s['exit_code']==0 for s in stages)
run=Path(accepted['output_dir'])
receipt=json.loads((run/'official_metrics.json').read_text())
training=json.loads((run/'training.json').read_text())
assert base.sha(run/'official_metrics.json')==accepted['official_metrics_sha256']
assert receipt['status']=='COMPLETE' and receipt['schema']=='trifusion-global-conditioned-evidence-v1'
assert receipt['seed']==42 and receipt['training_epochs']==50 and receipt['metrics']==accepted['metrics']
assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
assert [r['epoch'] for r in training['history']]==list(range(1,51))
selected=max(training['history'],key=lambda r:(r['official_fused']['mAP'],r['epoch']))
assert receipt['selected_epoch']==training['best_epoch']==selected['epoch']
assert all(abs(receipt['metrics'][k]-selected['official_fused'][k])<1e-5 for k in receipt['metrics'])
assert base.sha(run/'best_map.pth')==receipt['checkpoint_sha256']
assert base.sha(run/'official_distances.pt')==receipt['distance_sha256']
seal=root/'refine-logs/region_use_fixed_best_diagnosis918/INPUT_SEAL.json'
assert base.sha(seal)=='8e5c37e57c088a8ab53344266195bbc7183ccd27a94d0f0910ff9c20998b4b41'
pins=json.loads(seal.read_text())['artifact_sha256']
control_seal=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert base.sha(control_seal)==pins[str(control_seal)]
controls=json.loads(control_seal.read_text())
raw=next(row for row in controls['rows'] if row['dataset']=='RGBNT201' and row['variant']=='semantic')
global_only=next(row for row in controls['rows'] if row['dataset']=='RGBNT201' and row['variant']=='global_only')
steps=[json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
batches=[json.loads(line) for line in (run/'training_batch_order.jsonl').read_text().splitlines()]
assert len(steps)==len(batches)==sum(r['steps'] for r in training['history'])
assert [(s['epoch'],s['batch']) for s in steps]==[(b['epoch'],b['batch']) for b in batches]
old_order=Path(raw['run_dir'])/'training_batch_order.jsonl'
assert base.sha(old_order)==pins[str(old_order)]
assert (run/'training_batch_order.jsonl').read_bytes()==old_order.read_bytes()
traces=[]
for epoch in range(1,51):
    rows=[s for s in steps if s['epoch']==epoch]
    activities=[s['evidence_exposure'] for s in rows]
    assert len(rows)==training['history'][epoch-1]['steps']
    assert all(a['excluded_patches']==32*a['active_sample_modalities'] and a['available_sample_modalities']==192 for a in activities)
    assert all(math.isfinite(a[k]) for a in activities for k in ('gradient_norm_min','gradient_norm_max','gradient_norm_mean','target'))
    traces.append({'epoch':epoch,'steps':len(rows),
        'supported_active_steps':sum(a['support']['supported_relations']>0 and a['active_sample_modalities']>0 for a in activities),
        'zero_support_steps':sum(a['support']['supported_relations']==0 for a in activities),
        'legal_relations':sum(a['support']['legal_relations'] for a in activities),
        'supported_relations':sum(a['support']['supported_relations'] for a in activities),
        'active_sample_modalities':sum(a['active_sample_modalities'] for a in activities),
        'available_sample_modalities':sum(a['available_sample_modalities'] for a in activities),
        'excluded_patches':sum(a['excluded_patches'] for a in activities),
        'gradient_norm_mean':float(np.mean([a['gradient_norm_mean'] for a in activities])),
        'global_loss_mean':float(np.mean([s['global_loss'] for s in rows])),
        'fused_role_loss_mean':float(np.mean([s['fused_role_loss'] for s in rows]))})
torch.set_num_threads(1)
candidate={'run_dir':str(run),'metrics':receipt['metrics'],'distance_sha256':receipt['distance_sha256']}
pairs=[]
for control,name in ((raw,'raw_semantic'),(global_only,'independent_global')):
    result=paired('RGBNT201',descriptor(control,name),descriptor(candidate,args.exposure))
    changes=result['query_changes']
    groups={'repair':[],'new_error':[],'kept_correct':[],'kept_wrong':[]}
    for q in changes:
        first,second=q['control_first_rank']==1,q['candidate_first_rank']==1
        group=('kept_correct' if second else 'new_error') if first else ('repair' if second else 'kept_wrong')
        groups[group].append((q['candidate_ap']-q['control_ap'])*100)
    result['rank1_group_ap_contributions']={n:{'queries':len(v),'contribution_to_total_map_points':math.fsum(v)/len(changes)} for n,v in groups.items()}
    assert abs(sum(v['contribution_to_total_map_points'] for v in result['rank1_group_ap_contributions'].values())-result['delta_metrics']['mAP'])<1e-5
    delta=np.sort([(q['candidate_ap']-q['control_ap'])*100 for q in changes])
    result['worst_five_percent_query_ap_delta_mean_points']=float(delta[:math.ceil(.05*len(delta))].mean())
    pairs.append(result)
global_history_path=Path(global_only['run_dir'])/'training.json'
assert base.sha(global_history_path)==pins[str(global_history_path)]
global_history=json.loads(global_history_path.read_text())
same_epoch=next(r['official_fused'] for r in global_history['history'] if r['epoch']==receipt['selected_epoch'])
value={'status':'ONE_NEW_FORMATION_ENDPOINT_CPU_DIAGNOSIS_COMPLETE','at':datetime.now().astimezone().isoformat(),
    'dataset':'RGBNT201','exposure':args.exposure,'metrics':receipt['metrics'],'selected_epoch':receipt['selected_epoch'],
    'history':training['history'],'activity_traces':traces,'formal_recorded_steps':len(steps),
    'raw_semantic_training_order_equal':True,'pairs':pairs,
    'independent_global_same_epoch_metrics':same_epoch,
    'fused_minus_independent_same_epoch':{k:receipt['metrics'][k]-same_epoch[k] for k in receipt['metrics']},
    'same_model_global_decomposition_available':False,
    'best_to_last_map_drop':receipt['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
    'producer_train_wall_seconds':(datetime.fromisoformat(stages[0]['completed_at'])-datetime.fromisoformat(stages[0]['started_at'])).total_seconds(),
    'training_loop_seconds':sum(r['seconds'] for r in training['history']),
    'partition_peak_memory':training['partition_peak_memory'],'neural_calls':0,'optimizer_updates':0,
    'inputs':{str(p):base.sha(p) for p in (Path(__file__),seal,control_seal,run/'training.json',run/'official_metrics.json',run/'training_steps.jsonl',run/'training_batch_order.jsonl')},
    'boundary':'One original seed42 full50/first-strict endpoint, saved full legal camera-filtered ranking. Activity is model-derived sensitivity/support, not ground-truth region importance. Same-epoch independent history is a text metric comparison, not same-model global or feature equality. Final gradient/random comparison requires both endpoints. No model inference, test-time adaptation, causal proof, seed uncertainty or Signal superiority attested here.'}
assert not args.output_dir.exists()
args.output_dir.mkdir(parents=True)
(args.output_dir/'SUMMARY.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':value['status'],'exposure':args.exposure,'metrics':receipt['metrics'],
    'formal_steps':len(steps),'supported_active_steps':sum(r['supported_active_steps'] for r in traces)}),flush=True)
