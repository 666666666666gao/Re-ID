"""Render the already accepted report and histories; no retrieval/model recomputation."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path,PurePosixPath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
intake=base/'semantic_capacity_complete_intake856'
output=base/'semantic_capacity_text_analysis857'
assert not output.exists()
record=json.loads((intake/'stdout.json').read_bytes())
assert record['status']=='CAPACITY_COMPLETE_WITH_FIRST_EVAL_CONTINUATION_VERIFIED'
assert record['formal_completed']==3 and record['formal_epochs']==150 and record['formal_steps']==6484
assert record['report_invocations']==1 and record['completion_launch_exit']['exit_code']==0
assert record['original_launch_exit']['exit_code']==1 and record['new_training_invocations']==0
for name,info in record['files'].items():
    body=(intake/'received'/name).read_bytes()
    assert len(body)==info['bytes'] and hashlib.sha256(body).hexdigest()==info['sha256']
report=json.loads((intake/'received/results/semantic_capacity_control_v1_complete_20261005_856/SUMMARY.json').read_bytes())
controls=json.loads((intake/'received/refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_bytes())
assert len(report['rows'])==3 and len(report['pairs'])==9 and len(controls['rows'])==9
output.mkdir()

def write_csv(name,rows):
    assert rows
    with (output/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

datasets=('RGBNT201','MSVR310','RGBNT100')
curves,selected,query_rows,identity_rows=[],[],[],[]
histories={}
sources=[('raw_'+r['variant'],r) for r in controls['rows']]+[('semantic_capacity',r) for r in report['rows']]
for variant,row in sources:
    relative=PurePosixPath(row['run_dir']).relative_to('/data/gaob/Re-ID/Trifusion')
    training=json.loads((intake/'received'/relative/'training.json').read_bytes())
    assert len(training['history'])==50
    histories[row['dataset'],variant]=training['history']
    for epoch in training['history']:
        curves.append(dict(dataset=row['dataset'],variant=variant,epoch=epoch['epoch'],
                           mAP=epoch['official_fused']['mAP'],Rank1=epoch['official_fused']['Rank-1'],
                           Rank5=epoch['official_fused']['Rank-5'],Rank10=epoch['official_fused']['Rank-10']))
    selected.append(dict(dataset=row['dataset'],variant=variant,best_epoch=row['best_epoch'],
                         mAP=row['metrics']['mAP'],Rank1=row['metrics']['Rank-1'],
                         Rank5=row['metrics']['Rank-5'],Rank10=row['metrics']['Rank-10'],
                         best_to_last_mAP_drop=row['metrics']['mAP']-training['history'][-1]['official_fused']['mAP']))
for pair in report['pairs']:
    diag=pair['paired_diagnosis']
    for item in diag['query_changes']:
        query_rows.append(dict(dataset=pair['dataset'],control=pair['control'],query_index=item['query_index'],
            identity=item['identity'],control_AP=item['control_ap'],candidate_AP=item['candidate_ap'],
            delta_AP_points=100*(item['candidate_ap']-item['control_ap']),
            control_first_rank=item['control_first_rank'],candidate_first_rank=item['candidate_first_rank']))
    for item in diag['identity_changes']:
        identity_rows.append(dict(dataset=pair['dataset'],control=pair['control'],**item))
write_csv('selected_checkpoints.csv',selected)
write_csv('complete_50epoch_curves.csv',curves)
write_csv('all_query_differences.csv',query_rows)
write_csv('all_identity_differences.csv',identity_rows)
write_csv('nine_paired_comparisons.csv',[dict(dataset=p['dataset'],control=p['control'],
    delta_mAP=p['delta']['mAP'],delta_Rank1=p['delta']['Rank-1'],
    delta_Rank5=p['delta']['Rank-5'],delta_Rank10=p['delta']['Rank-10'],
    rank1_repairs=p['rank1_repairs'],rank1_new_errors=p['rank1_new_errors'],
    identity_macro_delta_AP_points=p['identity_macro_mean_delta_ap_points'],phase_progress=p['phase_progress'])
    for p in record['pairs']])

colors={'raw_global_only':'#505050','raw_semantic':'#2379b6','raw_native':'#dc7d20','semantic_capacity':'#269c67'}
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,layout='constrained')
for column,dataset in enumerate(datasets):
    for variant,color in colors.items():
        history=histories[dataset,variant]
        for axis,key in zip(axes[:,column],('mAP','Rank-1')):
            axis.plot([r['epoch'] for r in history],[r['official_fused'][key] for r in history],
                      label=variant,color=color,linewidth=1.25)
            axis.set_ylabel(key+' (%)');axis.grid(alpha=.2)
    axes[0,column].set_title(dataset);axes[1,column].set_xlabel('Epoch')
axes[0,0].legend(fontsize=8)
fig.savefig(output/'complete_50epoch_curves.svg')
fig.savefig(output/'complete_50epoch_curves.png',dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
for axis,dataset in zip(axes,datasets):
    pairs=[p for p in record['pairs'] if p['dataset']==dataset]
    axis.bar([p['control'].removeprefix('raw_') for p in pairs],[p['delta']['mAP'] for p in pairs],color='#269c67')
    axis.axhline(0,color='#555555',linewidth=.8)
    axis.axhline(.5,color='#888888',linestyle='--',linewidth=.8)
    axis.set_title(dataset);axis.set_ylabel('Capacity minus control mAP (pp)')
    axis.grid(axis='y',alpha=.2)
fig.savefig(output/'nine_mAP_differences.svg')
fig.savefig(output/'nine_mAP_differences.png',dpi=180)
plt.close(fig)

summary=dict(status='ACCEPTED_REPORT_TEXT_VISUALIZATION_COMPLETE',created_at=datetime.now().astimezone().isoformat(),
    original_report_sha256=hashlib.sha256((intake/'received/results/semantic_capacity_control_v1_complete_20261005_856/SUMMARY.json').read_bytes()).hexdigest(),
    curve_rows=len(curves),selected_rows=len(selected),query_rows=len(query_rows),identity_rows=len(identity_rows),
    phase_progress_count=sum(p['phase_progress'] for p in record['pairs']),
    boundary='Displays original accepted single-seed histories and once-only CPU report; no new inference, retrieval scoring, bootstrap, checkpoint selection or causal/SOTA claim. Epochs are dependent observations, not training replicates. Dashed .5 line is only the mAP part of the project gate; Rank1 must also not fall.')
summary['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in output.iterdir()}
(output/'SUMMARY.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
print(json.dumps(summary))
