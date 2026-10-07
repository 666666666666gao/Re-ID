from pathlib import Path
import hashlib
import json
import math
import statistics

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/fixed_five_complete893')
data = json.loads((packet / 'REMOTE.json').read_bytes())
assert data['original_exit']['exit_code'] == 0 and len(data['rows']) == 5
mapping = json.loads((packet / 'FILE_MAP.json').read_bytes())
assert all(hashlib.sha256(Path(r['local_file']).read_bytes()).hexdigest() == r['sha256'] for r in mapping.values())
rows = []
for row in data['rows']:
    expected = 836 if row['dataset'] == 'RGBNT201' else (591 if row['dataset'] == 'MSVR310' else 1715)
    for name, score in row['scores'].items():
        ap, rank = score['average_precision'], score['first_match_rank']
        assert len(ap) == len(rank) == expected and all(math.isfinite(x) and 0 <= x <= 1 for x in ap)
        calculated = {'mAP': statistics.mean(ap) * 100, **{'Rank-' + str(k): sum(r <= k for r in rank) * 100 / expected for k in (1, 5, 10)}}
        assert all(abs(v - score['metrics'][k]) < 1e-5 for k, v in calculated.items()), (row['dataset'], name)
    pairs = {'same_model_global_to_fused': ('global', 'fused'),
             'independent_global_only_to_same_model_global': ('independent_global_only', 'global'),
             'independent_global_only_to_fused': ('independent_global_only', 'fused')}
    for name, (a, b) in pairs.items():
        pair = row['comparisons'][name]
        first, second = row['scores'][a], row['scores'][b]
        repairs = sum(x != 1 and y == 1 for x, y in zip(first['first_match_rank'], second['first_match_rank']))
        errors = sum(x == 1 and y != 1 for x, y in zip(first['first_match_rank'], second['first_match_rank']))
        assert repairs == pair['rank1_repairs'] and errors == pair['rank1_new_errors']
        assert len(pair['query_changes']) == expected
        for i, change in enumerate(pair['query_changes']):
            assert change['query_index'] == i
            assert change['control_ap'] == first['average_precision'][i] and change['candidate_ap'] == second['average_precision'][i]
            assert change['control_first_rank'] == first['first_match_rank'][i] and change['candidate_first_rank'] == second['first_match_rank'][i]
        for k, v in pair['delta_metrics'].items():
            assert abs(v - (second['metrics'][k] - first['metrics'][k])) < 1e-10
        identity_delta = []
        for identity in pair['identity_changes']:
            selected = [q for q in pair['query_changes'] if q['identity'] == identity['identity']]
            assert len(selected) == identity['queries']
            delta = statistics.mean(q['candidate_ap'] - q['control_ap'] for q in selected) * 100
            assert abs(delta - identity['mean_delta_ap_points']) < 1e-10
            identity_delta.append(delta)
        assert abs(statistics.mean(identity_delta) - pair['identity_macro_delta_ap_points']) < 1e-10
    own = row['comparisons']['same_model_global_to_fused']
    ap_up = sum(q['candidate_ap'] > q['control_ap'] for q in own['query_changes'])
    ap_down = sum(q['candidate_ap'] < q['control_ap'] for q in own['query_changes'])
    rows.append(dict(dataset=row['dataset'], objective=row['objective'], selected_epoch=row['selected_epoch'],
        metrics={n:v['metrics'] for n,v in row['scores'].items()},
        same_model_delta=own['delta_metrics'], repairs=own['rank1_repairs'], new_errors=own['rank1_new_errors'],
        query_ap_improved=ap_up, query_ap_worsened=ap_down, query_ap_unchanged=expected-ap_up-ap_down,
        identity_macro_delta=own['identity_macro_delta_ap_points'],
        independent_to_own_global_delta=row['comparisons']['independent_global_only_to_same_model_global']['delta_metrics'],
        diagnostics=row['diagnostic'], original_distance_max_difference=row['fused_distance_max_absolute_difference_from_original']))
output = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/fixed_five_component_analysis893')
assert not output.exists()
output.mkdir()
summary = dict(status='ALL_FIVE_COMPONENT_ARRAYS_INDEPENDENTLY_TEXT_CHECKED', rows=rows,
    source_count=424, sealed_artifacts=75, optimizer_updates=0,
    boundary='Standard-library CPU recheck of every saved score/query/identity row. No tensors loaded locally, no NN, fitting, checkpoint selection, new threshold or source modification. Fixed-best consumed-protocol diagnosis, not independent ablation/causal attribution/training-seed uncertainty.')
(output / 'SUMMARY.json').write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
table = '\n'.join('| {dataset} | {objective} | {g:.4f} | {c:.4f} | {f:.4f} | {dm:+.4f} / {dr:+.4f} | {repairs} / {new_errors} | {query_ap_improved} / {query_ap_worsened} |'.format(
    **r, g=r['metrics']['global']['mAP'], c=r['metrics']['correction']['mAP'], f=r['metrics']['fused']['mAP'],
    dm=r['same_model_delta']['mAP'], dr=r['same_model_delta']['Rank-1']) for r in rows)
report = '''# 固定五份best的global / 修正 / 融合分解

| 数据集 | 目标 | own-g mAP | c单独mAP | f mAP | f−own-g ΔmAP / ΔR1 | 首位修复 / 新增 | query AP改善 / 恶化 |
|---|---|---:|---:|---:|---:|---:|---:|
''' + table + '\n\n' + summary['boundary'] + '\n'
(output / 'REPORT.md').write_bytes(report.encode('utf-8'))
print(json.dumps(dict(status=summary['status'],rows=[{k:v for k,v in r.items() if k not in ('diagnostics','metrics')} for r in rows]),ensure_ascii=False))
