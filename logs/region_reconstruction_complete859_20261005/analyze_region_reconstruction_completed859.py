"""Render accepted saved histories and the original fifteen-pair report only."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
intake = base/'region_reconstruction_completed_admin_intake858'
output = base/'region_reconstruction_complete_analysis859'
record = json.loads((intake/'stdout.json').read_bytes())
assert record['status'] == 'SIX_REGION_RECONSTRUCTION_ENDPOINTS_ADMINISTRATIVE_COMPLETION_VERIFIED'
assert record['accepted'] == 6 and record['formal_epochs'] == 300 and record['formal_steps'] == 12968
assert record['pairs'] == 15 and record['report_invocations'] == 1
assert record['original_parent_exit_code'] == 1 and record['completion_launch_exit']['exit_code'] == 0
assert record['new_training_invocations'] == 1 and record['inherited_formal_endpoints'] == 5
assert not output.exists()
for name, digest in record['text_sha256'].items():
    assert hashlib.sha256((intake/'received'/name).read_bytes()).hexdigest() == digest, name
report_path = intake/'received/results/region_reconstruction_v1_complete_20261005_858/SUMMARY.json'
report = json.loads(report_path.read_bytes())
controls = json.loads((intake/'received/refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_bytes())
assert len(report['rows']) == 6 and len(report['pairs']) == 15 and len(controls['rows']) == 9
output.mkdir()

def write_csv(name, rows):
    with (output/name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

datasets = ('RGBNT201', 'MSVR310', 'RGBNT100')
curves, selected, query_rows, identity_rows, paired = [], [], [], [], []
histories = {}
sources = [('raw_'+r['variant'], r) for r in controls['rows']]
sources += [(r['query_mode'], r) for r in report['rows']]
for variant, row in sources:
    relative = PurePosixPath(row['run_dir']).relative_to('/data/gaob/Re-ID/Trifusion')
    training = json.loads((intake/'received'/relative/'training.json').read_bytes())
    assert [r['epoch'] for r in training['history']] == list(range(1, 51))
    histories[row['dataset'], variant] = training['history']
    for epoch in training['history']:
        metrics = epoch['official_fused']
        curves.append(dict(dataset=row['dataset'], variant=variant, epoch=epoch['epoch'],
                           mAP=metrics['mAP'], Rank1=metrics['Rank-1'],
                           Rank5=metrics['Rank-5'], Rank10=metrics['Rank-10']))
    selected.append(dict(dataset=row['dataset'], variant=variant, best_epoch=row['best_epoch'],
        mAP=row['metrics']['mAP'], Rank1=row['metrics']['Rank-1'],
        Rank5=row['metrics']['Rank-5'], Rank10=row['metrics']['Rank-10'],
        last_mAP=training['history'][-1]['official_fused']['mAP'],
        best_to_last_mAP_drop=row['metrics']['mAP']-training['history'][-1]['official_fused']['mAP']))
for pair in report['pairs']:
    diag = pair['paired_diagnosis']
    delta = diag['delta_metrics']
    paired.append(dict(dataset=pair['dataset'], candidate=pair['candidate'], control=pair['control'],
        delta_mAP=delta['mAP'], delta_Rank1=delta['Rank-1'],
        delta_Rank5=delta['Rank-5'], delta_Rank10=delta['Rank-10'],
        rank1_repairs=diag['rank1_repairs'], rank1_new_errors=diag['rank1_new_errors'],
        identity_macro_delta_AP_points=diag['identity_macro_mean_delta_ap_points'],
        phase_progress=pair['phase_progress']))
    for item in diag['query_changes']:
        query_rows.append(dict(dataset=pair['dataset'], candidate=pair['candidate'], control=pair['control'],
            **item, delta_AP_points=100*(item['candidate_ap']-item['control_ap'])))
    for item in diag['identity_changes']:
        identity_rows.append(dict(dataset=pair['dataset'], candidate=pair['candidate'], control=pair['control'], **item))
assert len(curves) == 750 and len(selected) == 15 and len(paired) == 15
write_csv('selected_checkpoints.csv', selected)
write_csv('complete_50epoch_curves.csv', curves)
write_csv('fifteen_paired_comparisons.csv', paired)
write_csv('all_query_differences.csv', query_rows)
write_csv('all_identity_differences.csv', identity_rows)

colors = {'raw_global_only':'#505050', 'raw_semantic':'#2379b6', 'raw_native':'#dc7d20',
          'patch':'#269c67', 'mean':'#a04ba4'}
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout='constrained')
for column, dataset in enumerate(datasets):
    for variant, color in colors.items():
        history = histories[dataset, variant]
        for axis, key in zip(axes[:, column], ('mAP', 'Rank-1')):
            axis.plot([r['epoch'] for r in history], [r['official_fused'][key] for r in history],
                      label=variant, color=color, linewidth=1.25)
            axis.set_ylabel(key+' (%)')
            axis.grid(alpha=.2)
    axes[0, column].set_title(dataset)
    axes[1, column].set_xlabel('Epoch')
axes[0, 0].legend(fontsize=8)
fig.savefig(output/'complete_50epoch_curves.svg')
fig.savefig(output/'complete_50epoch_curves.png', dpi=180)
plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
for axis, dataset in zip(axes, datasets):
    pairs = [p for p in paired if p['dataset'] == dataset]
    labels = [p['candidate']+' - '+p['control'].removeprefix('raw_') for p in pairs]
    axis.barh(labels, [p['delta_mAP'] for p in pairs], color=['#269c67' if p['candidate']=='patch' else '#a04ba4' for p in pairs])
    axis.axvline(0, color='#555555', linewidth=.8)
    axis.axvline(.5, color='#888888', linestyle='--', linewidth=.8)
    axis.set_title(dataset)
    axis.set_xlabel('mAP difference (pp)')
    axis.tick_params(axis='y', labelsize=8)
    axis.grid(axis='x', alpha=.2)
fig.savefig(output/'fifteen_mAP_differences.svg')
fig.savefig(output/'fifteen_mAP_differences.png', dpi=180)
plt.close(fig)
summary = dict(status='ACCEPTED_REGION_REPORT_TEXT_VISUALIZATION_COMPLETE',
    created_at=datetime.now().astimezone().isoformat(), original_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
    curve_rows=len(curves), selected_rows=len(selected), pair_rows=len(paired),
    query_rows=len(query_rows), identity_rows=len(identity_rows),
    primary_patch_vs_mean_progress_count=sum(p['phase_progress'] for p in paired if p['control']=='mean'),
    all_pair_progress_count=sum(p['phase_progress'] for p in paired),
    boundary='Original accepted single-seed histories and once-only CPU report only. No inference, retrieval, bootstrap or checkpoint reselection rerun. Epochs are dependent observations. .5pp line is only mAP component of project gate; Rank1 must also not decline. Original parent EXIT1 preserved, separate administrative last-training completion EXIT0.')
summary['artifacts'] = {p.name:dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in output.iterdir()}
(output/'SUMMARY.json').write_bytes((json.dumps(summary, indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in summary.items() if k!='artifacts'}, indent=2))
