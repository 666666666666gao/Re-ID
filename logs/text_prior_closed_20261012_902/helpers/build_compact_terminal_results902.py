import json
from pathlib import Path

work = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/text_semantic_prior901')
root = work / 'runtime/TERMINAL_901'
summary = json.loads((root/'logs/text_semantic_prior_v1_20261010_901_report/SUMMARY.json').read_text(encoding='utf-8'))
compact = {key: summary[key] for key in ('schema','status','created_at','accepted','expected',
             'formal_epochs','formal_steps','missing','primary_advance','primary_pairs_available','boundary')}
compact['pairs'] = [{key: value for key, value in row.items() if key not in ('query_changes','identity_changes')}
                    for row in summary['pairs']]
matrix = json.loads((root/'logs/text_semantic_prior_v1_20261010_901/accepted_matrix.json').read_text(encoding='utf-8'))
compact['rows'] = []
for row in matrix['rows']:
    training = json.loads((root/Path(row['run_dir']).relative_to('/data/gaob/Re-ID/Trifusion')/'training.json').read_text(encoding='utf-8'))
    last = training['history'][-1]['official_fused']
    compact['rows'].append(dict(dataset=row['dataset'],package=row['package'],best_epoch=row['best_epoch'],
        metrics=row['metrics'],own_global_metrics=row['own_global_metrics'],last_epoch_metrics=last,
        best_to_last_mAP_drop=row['metrics']['mAP']-last['mAP'],
        checkpoint_sha256=row['checkpoint_sha256'],run_dir=row['run_dir']))
compact['intake_helper_failure'] = dict(tool_chunk='e43c78', exception="KeyError: 'metrics'",
    cause="Compact helper used nonexistent history.metrics; production history uses official_fused.",
    audit_trace_closure_already_written=True, neural_rerun=False, original_results_changed=False)
target=work/'runtime/COMPACT_TERMINAL_RESULTS_901.json'
assert not target.exists()
target.write_text(json.dumps(compact,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(accepted=compact['accepted'],formal_epochs=compact['formal_epochs'],formal_steps=compact['formal_steps'],
    primary_pairs=[r for r in compact['pairs'] if r.get('primary_pair')]),ensure_ascii=False))
