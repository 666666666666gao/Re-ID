"""One CPU report after all six N1 endpoints; use saved full-gallery GT distances."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_native_detail as native
from tools.analyze_correspondence_distances import compare, sha
from tools.report_clean_clip_joint_complete import elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    native.install()
    campaign = args.campaign.resolve()
    manifest = native.panel.require_sources(campaign)
    state = json.loads((campaign/'campaign.json').read_text())
    matrix = json.loads((campaign/'accepted_matrix.json').read_text())
    assert matrix['schema'] == native.SCHEMA and matrix['accepted'] == matrix['expected'] == len(matrix['rows']) == 6
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 6
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
    assert {(row['dataset'],row['variant']) for row in matrix['rows']} == {
        (dataset,variant) for dataset in native.DATASETS for variant in native.CONDITIONS}
    assert sha(native.SUMMARY) == native.SUMMARY_SHA
    old_report = json.loads(native.SUMMARY.read_text())
    old_matrix_path = native.PREDECESSOR/'accepted_matrix.json'
    old_matrix = json.loads(old_matrix_path.read_text())
    assert old_matrix['schema'] == 'trifusion-clean-public-clip-joint-v1'
    assert old_matrix['accepted'] == old_matrix['expected'] == len(old_matrix['rows']) == 6
    for old in old_matrix['rows']:
        saved = next(row for row in old_report['rows']
                     if (row['dataset'],row['variant']) == (old['dataset'],old['variant']))
        assert old == {key:saved[key] for key in old}
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    files = [campaign/name for name in ('manifest.json','campaign.json','accepted_matrix.json')]
    files += [Path(manifest['preflight_path']), *map(Path,manifest['initialization_sha256']),
              native.SUMMARY, old_matrix_path, Path(__file__).resolve(), ROOT/'tools/analyze_correspondence_distances.py']
    sources = {str(path):sha(path) for path in files}
    rows, histories, outputs = [], {}, set()
    for row in matrix['rows']:
        run = Path(row['run_dir'])
        child_dir = native.queue.child_campaign(campaign,'clean_clip',row['dataset'],row['variant'])
        child = json.loads((child_dir/'campaign.json').read_text())
        assert child['status'] == 'COMPLETE' and child['verification'] == row
        assert [stage['mode'] for stage in child['jobs']] == ['m0','train','evaluate']
        assert all(stage['status'] == 'COMPLETE' and stage['exit_code'] == 0 for stage in child['jobs'])
        actual = native.panel.verify(run,Path(row['m0_dir']),row['dataset'],row['variant'],
                                     Path(manifest['preflight_path']).parent)
        assert actual == row
        training = json.loads((run/'training.json').read_text())
        history = training['history']
        steps = sum(1 for _ in (run/'training_steps.jsonl').open())
        assert steps == sum(item['steps'] for item in history)
        best = history[row['best_epoch']-1]
        for path in (child_dir/'campaign.json',run/'training.json',run/'training_steps.jsonl',
                     run/'official_metrics.json',Path(row['m0_dir'])/'training.json',Path(row['m0_dir'])/'training_steps.jsonl'):
            sources[str(path)] = sha(path)
        outputs.update((run,Path(row['m0_dir'])))
        histories[row['dataset'],row['variant']] = history
        rows.append({**row,'formal_steps':steps,'trainable_parameters':row['initializer']['trainable_parameters'],
            'best_mean_loss':best['mean_loss'],'final_mean_loss':history[-1]['mean_loss'],
            'final_minus_best_metrics':{name:history[-1]['official_fused'][name]-row['metrics'][name] for name in row['metrics']}})
    # Original independent controls remain separate records; no old training/report rerun.
    combined = {'rows':old_matrix['rows']+matrix['rows']}
    pairs = []
    for dataset in native.DATASETS:
        for control, gate in (('roles','N1-A'),('low','N1-B'),('global_only',None)):
            diagnosis = compare(combined,dataset,control,'high')
            if control == 'low':
                diagnosis['boundary'] = (
                    'Official post-selection diagnosis, not new inference or selection. '
                    'Matched parameters and initialization; input computation differs. '
                    'Low upsampling cannot restore removed detail. '
                    'Bootstrap resamples fixed-model identities, not training seeds.')
            pair = {'dataset':dataset,'gate':gate,'paired_diagnosis':diagnosis}
            if gate is not None:
                delta = diagnosis['delta_metrics']
                floor = 0.5 if gate == 'N1-A' and dataset in ('RGBNT201','MSVR310') else 0.0
                pair.update(minimum_map_points=floor,
                            gate_passed=delta['mAP'] > 0 and delta['mAP'] >= floor and delta['Rank-1'] >= 0)
            pairs.append(pair)
    for row in combined['rows']:
        for name in ('official_metrics.json','official_distances.pt'):
            path = Path(row['run_dir'])/name
            sources[str(path)] = sha(path)
    assert native.panel.require_sources(campaign) == manifest
    assert all(sha(Path(path)) == digest for path,digest in sources.items())
    gates = {gate:'PASS' if all(p['gate_passed'] for p in pairs if p['gate'] == gate) else 'FAIL'
             for gate in ('N1-A','N1-B')}
    boundaries = [
        'N1 only: image-native CNN values, existing CLIP semantic keys; N2/N3 not tested.',
        'All high/low initial states matched; all nonstem tensors matched original clean roles. New stem adds93,248parameters.',
        'High/low share parameters,512candidates and1536output; low upsampling cannot restore removed detail. Input computation differs.',
        'Full50 per endpoint, one official fused-mAP-best with later tied epoch, strict full-state reload, real GT/full gallery/original filtering/no rerank.',
        'Official benchmark used in development and epoch selection; single seed42 and fixed-model identity bootstrap do not prove unbiased or multi-seed stability.',
        'Prior clean global-only/roles records are fixed matched-recipe controls, not newly trained arms in this campaign.',
        'N1-A/B are intermediate contribution gates, not three-dataset baseline/SOTA achievement or individual-role necessity.',
        'Reported training/epoch-eval seconds exclude construction and prior controls; cost differences remain explicit.',
    ]
    report = {'schema':'trifusion-native-detail-complete-report-v1','status':'COMPLETE',
        'created_at':datetime.now().astimezone().isoformat(),'campaign':str(campaign),'accepted':6,
        'formal_epochs':300,'formal_steps':sum(row['formal_steps'] for row in rows),
        'registered_gates':gates,'registered_both_gates':'PASS' if all(v == 'PASS' for v in gates.values()) else 'FAIL',
        'rows':rows,'prior_control_rows':old_report['rows'],'pairs':pairs,'report_inputs_sha256':sources,
        'campaign_parent_observed_wall_seconds':elapsed(state['started_at'],state['completed_at']),
        'sum_training_and_epoch_eval_seconds':sum(row['training_and_epoch_eval_seconds'] for row in rows),
        'logical_m0_and_full_output_bytes':sum(p.stat().st_size for directory in outputs for p in directory.rglob('*') if p.is_file()),
        'disk_free_bytes_at_report':shutil.disk_usage(ROOT).free,'boundaries':boundaries}
    args.output_dir.mkdir(parents=True)
    figure, axes = plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
    for index,dataset in enumerate(native.DATASETS):
        for variant in native.CONDITIONS:
            history = histories[dataset,variant]
            epochs = [item['epoch'] for item in history]
            axes[index,0].plot(epochs,[item['mean_loss'] for item in history],label=variant)
            axes[index,1].plot(epochs,[item['official_fused']['mAP'] for item in history],label=variant)
        for column,label in enumerate(('Mean training loss','Official fused mAP (%)')):
            axes[index,column].set(title=dataset,xlabel='Epoch',ylabel=label)
            axes[index,column].grid(alpha=.2)
            axes[index,column].legend()
    for suffix in ('png','svg'):
        figure.savefig(args.output_dir/f'TRAINING_CURVES.{suffix}',dpi=160)
    plt.close(figure)
    report['figure_sha256'] = {name:sha(args.output_dir/name) for name in ('TRAINING_CURVES.png','TRAINING_CURVES.svg')}
    (args.output_dir/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# N1 full-six CPU report','','| Dataset | Variant | Best epoch | mAP | R1 | R5 | R10 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:(native.DATASETS.index(r['dataset']),native.CONDITIONS.index(r['variant']))):
        m = row['metrics']
        tail = [f"{m[name]:.4f}" if row['dataset'] == 'RGBNT201' else '—' for name in ('Rank-5','Rank-10')]
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['best_epoch']} | {m['mAP']:.4f} | {m['Rank-1']:.4f} | {' | '.join(tail)} |")
    lines += ['', '| Dataset | Comparison | ΔmAP | ΔR1 | Repairs/new errors | Gate |', '|---|---|---:|---:|---|---|']
    for p in pairs:
        d = p['paired_diagnosis']
        verdict = f"{p['gate']}: {'PASS' if p['gate_passed'] else 'FAIL'}" if p['gate'] is not None else 'Descriptive only'
        lines.append(f"| {p['dataset']} | high minus {d['control']} | {d['delta_metrics']['mAP']:+.4f} | {d['delta_metrics']['Rank-1']:+.4f} | {d['rank1_repairs']}/{d['rank1_new_errors']} | {verdict} |")
    lines += ['', 'Registered gates: '+json.dumps(gates)+'.','',*['- '+line for line in boundaries]]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'COMPLETE','accepted':6,'registered_gates':gates,'output_dir':str(args.output_dir)}),flush=True)


if __name__ == '__main__':
    main()
