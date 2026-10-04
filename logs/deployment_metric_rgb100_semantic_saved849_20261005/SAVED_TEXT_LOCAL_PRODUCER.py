"""Analyze the newly accepted RGBNT100 semantic pair from saved texts only."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'deployment_metric_RGBNT100_semantic_full848'
output=base/'deployment_metric_rgb100_semantic_saved849'
assert not output.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal_path=repo/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
seal=json.loads(seal_path.read_bytes())
assert sha(seal_path)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
summary_path=packet/'SUMMARY.json'
summary=json.loads(summary_path.read_bytes())
assert summary['status']=='ORIGINAL_FRESH50_FIRST_STRICT_AND_PROBE_RETIREMENT_VERIFIED'
row=summary['row']
assert (row['dataset'],row['variant'],row['best_epoch'])==('RGBNT100','semantic',5)
run_relative=row['run_dir'].removeprefix('/data/gaob/Re-ID/Trifusion/')
training_path=packet/'received'/run_relative/'training.json'
acceptance_path=packet/'received/logs/deployment_metric_role_pending100_20261005_842/acceptance/RGBNT100_semantic.json'
acceptance=json.loads(acceptance_path.read_bytes())
assert sha(acceptance_path)==summary['full_acceptance_sha256']
assert sha(training_path)==acceptance['artifact_sha256'][row['run_dir']+'/training.json']
training=json.loads(training_path.read_bytes())
control=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==('RGBNT100','semantic'))
transport_path=base/'global_task_role_rgb100_semantic_full829/stdout.json'
transport=json.loads(transport_path.read_bytes())
control_name=control['run_dir']+'/training.json'
old_raw=transport['files'][control_name.removeprefix('/data/gaob/Re-ID/Trifusion/')].encode('utf-8')
assert hashlib.sha256(old_raw).hexdigest()==seal['artifact_sha256'][control_name]
old_training=json.loads(old_raw)
old_tasks=transport['summary']['training_task_and_norm_trajectory']
new_tasks=summary['training_task_and_norm_trajectory']
for values in (training['history'],old_training['history'],old_tasks,new_tasks):
    assert [r['epoch'] for r in values]==list(range(1,51))
assert training['best_epoch']==old_training['best_epoch']==control['best_epoch']==5
records=[]
for nh,oh,n,o in zip(training['history'],old_training['history'],new_tasks,old_tasks):
    assert nh['steps']==oh['steps']==n['steps']==o['steps']
    r=dict(epoch=n['epoch'],steps=n['steps'],new_global_loss=n['global_loss'],old_global_loss=o['mean_global_loss'],
        new_global_norm=n['shared_global_norm_mean'],old_global_norm=o['shared_global_norm_mean'],
        new_role_loss=n['fused_role_loss'],old_role_loss=o['mean_fused_role_loss'],
        new_correction_norm=n['correction_norm_mean'],old_correction_norm=o['correction_norm_mean'],
        new_scaled_correction_norm=n['actual_scaled_correction_norm_mean'],old_scaled_correction_norm=o['actual_scaled_correction_norm_mean'],
        new_correction_global_ratio=n['actual_scaled_correction_global_ratio_mean'],old_correction_global_ratio=o['actual_scaled_correction_global_ratio_mean'],
        new_role_metric_norm=n['role_metric_norm_mean'])
    for metric in ('mAP','Rank-1','Rank-5','Rank-10'):
        r['new_'+metric]=nh['official_fused'][metric]
        r['old_'+metric]=oh['official_fused'][metric]
        r['delta_'+metric]=r['new_'+metric]-r['old_'+metric]
    assert all(math.isfinite(v) for v in r.values())
    records.append(r)
assert sum(r['steps'] for r in records)==3129
counts={metric:dict(positive=sum(r['delta_'+metric]>0 for r in records),
                   equal=sum(r['delta_'+metric]==0 for r in records),negative=sum(r['delta_'+metric]<0 for r in records))
        for metric in ('mAP','Rank-1','Rank-5','Rank-10')}
global_loss_diff=max(abs(r['new_global_loss']-r['old_global_loss']) for r in records)
global_norm_diff=max(abs(r['new_global_norm']-r['old_global_norm']) for r in records)
boundary=('One accepted RGBNT100 semantic full50/3129-step pair, original selected E5 and first strict unchanged. '
    'Saved text only: no remote access,PyTorch,model/inference/optimizer,post-hoc best/seed selection or current native query. '
    'Epochs are dependent observations,not training seeds. Recorded global loss/norm equality is scalar aggregate equality, '
    'not proof of all parameters/global query features being identical. Correction ratios are means over training batches, '
    'not ratios on query features and not causal proof. Raw/L2 role losses change metric geometry and cannot be directly '
    'ranked as relative fitting quality. Current native is incomplete;original MSVR native M0 failure is preserved. '
    'No full five-endpoint/12-pair saved-array report or claim of general effectiveness.')
analysis=dict(status='RGBNT100_SEMANTIC_ACCEPTED_SAVED_PAIR_ANALYSIS_COMPLETE',at=datetime.now().astimezone().isoformat(),
    producer_sha256=sha(Path(__file__)),dataset='RGBNT100',variant='semantic',epochs=50,formal_steps=3129,selected_epoch=5,
    selected_metrics=row['metrics'],registered_progress=summary['deltas']['semantic']['phase_progress'],
    global_loss_max_abs_difference=global_loss_diff,global_norm_max_abs_difference=global_norm_diff,
    role_metric_norm_range=[min(r['new_role_metric_norm'] for r in records),max(r['new_role_metric_norm'] for r in records)],
    selected_epoch_record=records[4],last_epoch_record=records[-1],epoch_delta_counts=counts,
    epoch_map_positive_rank1_nonnegative=sum(r['delta_mAP']>0 and r['delta_Rank-1']>=0 for r in records),
    new_best_to_last_map_drop=records[4]['new_mAP']-records[-1]['new_mAP'],
    old_best_to_last_map_drop=records[4]['old_mAP']-records[-1]['old_mAP'],
    descriptive_fixed_epochs=[records[i-1] for i in (1,5,10,20,30,40,50)],
    inputs={str(p):sha(p) for p in (summary_path,training_path,acceptance_path,transport_path,seal_path)},
    matched_control_training_sha256=hashlib.sha256(old_raw).hexdigest(),boundary=boundary)
output.mkdir()
with (output/'TRAJECTORY.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]))
    writer.writeheader();writer.writerows(records)
(output/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
lines=['# RGBNT100 semantic: accepted saved full50 pair','',boundary,'',
    '| Epoch | New mAP | Raw control mAP | New R1 | Raw control R1 | New c/g (%) | Raw c/g (%) |',
    '|---:|---:|---:|---:|---:|---:|---:|']
for r in analysis['descriptive_fixed_epochs']:
    lines.append(f"| {r['epoch']} | {r['new_mAP']:.6f} | {r['old_mAP']:.6f} | {r['new_Rank-1']:.6f} | {r['old_Rank-1']:.6f} | {100*r['new_correction_global_ratio']:.6f} | {100*r['old_correction_global_ratio']:.6f} |")
lines += ['',f'Recorded global objective maximum absolute difference: {global_loss_diff}; global norm: {global_norm_diff}.',
    f"Best-to-last mAP drop: new {analysis['new_best_to_last_map_drop']:.6f}; raw control {analysis['old_best_to_last_map_drop']:.6f}.",
    '', 'Observation: correction behavior and retrieval trajectories diverge under matched recorded global training aggregates.',
    'Interpretation limit: this does not establish that correction magnitude causes errors,or that global states are bitwise identical.',
    'Next action: finish the original native full50/firststrict and the explicitly registered saved-array report before choosing another intervention.']
(output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:analysis[k] for k in ('status','at','global_loss_max_abs_difference','global_norm_max_abs_difference',
    'epoch_delta_counts','epoch_map_positive_rank1_nonnegative','new_best_to_last_map_drop','old_best_to_last_map_drop')}))
