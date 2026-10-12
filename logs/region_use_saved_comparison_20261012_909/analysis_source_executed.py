"""Compare two saved query tables; no model, tensor, or production invocation."""
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


WORK = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903')
INTAKE = WORK / 'results908'
OUT = WORK / 'comparison909'


def stats(rows):
    count = len(rows)
    delta = [100 * (r['ambiguity_ap'] - r['uniform_ap']) for r in rows]
    identities = defaultdict(list)
    groups = defaultdict(list)
    for r, change in zip(rows, delta):
        identities[r['identity']].append(change)
        old, new = r['uniform_first_rank'] == 1, r['ambiguity_first_rank'] == 1
        group = ('still_correct' if new else 'new_error') if old else ('repair' if new else 'still_wrong')
        groups[group].append(change)
    tail_count = math.ceil(count * .05)
    return {
        'query_count': count,
        'delta_mAP_points': sum(delta) / count,
        'delta_Rank1_points': 100 * sum((r['ambiguity_first_rank'] == 1) - (r['uniform_first_rank'] == 1) for r in rows) / count,
        'query_AP_improved': sum(x > 0 for x in delta),
        'query_AP_worsened': sum(x < 0 for x in delta),
        'query_AP_equal': sum(x == 0 for x in delta),
        'identity_macro_delta_AP_points': sum(sum(x) / len(x) for x in identities.values()) / len(identities),
        'identities': [{'identity': identity, 'query_count': len(changes), 'mean_delta_AP_points': sum(changes) / len(changes)} for identity, changes in sorted(identities.items())],
        'rank1_transition_groups': {name: {'query_count': len(changes), 'contribution_to_all_query_delta_mAP_points': sum(changes) / count} for name, changes in groups.items()},
        'worst_ceil5percent_query_count': tail_count,
        'worst_ceil5percent_mean_delta_AP_points': sum(sorted(delta)[:tail_count]) / tail_count,
    }


manifest = json.loads((INTAKE / 'INTAKE_MANIFEST_908.json').read_text())['files']
input_hashes = {}
results = []
for dataset in ('RGBNT201', 'MSVR310'):
    source = {}
    for condition in ('uniform', 'ambiguity'):
        relative = f'logs/region_use_endpoint_diagnosis_20261012_907/{dataset}_{condition}/SUMMARY.json'
        path = INTAKE / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest[relative]
        input_hashes[str(path)] = digest
        summary = json.loads(path.read_text())
        assert summary['status'] == 'ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE'
        assert summary['dataset'] == dataset and summary['comparison'] == condition
        pair = [p for p in summary['pairs'] if p['control'] == 'independent_global']
        assert len(pair) == 1
        source[condition] = (summary, pair[0])
    u, up = source['uniform']
    a, ap = source['ambiguity']
    assert up['metrics']['independent_global'] == ap['metrics']['independent_global']
    assert len(up['query_changes']) == len(ap['query_changes'])
    rows = []
    for i, (ur, ar) in enumerate(zip(up['query_changes'], ap['query_changes'])):
        assert ur['query_index'] == ar['query_index'] == i
        assert ur['identity'] == ar['identity']
        assert ur['control_ap'] == ar['control_ap'] and ur['control_first_rank'] == ar['control_first_rank']
        rows.append({
            'query_index': i, 'identity': ur['identity'],
            'global_ap': ur['control_ap'], 'global_first_rank': ur['control_first_rank'],
            'uniform_ap': ur['candidate_ap'], 'uniform_first_rank': ur['candidate_first_rank'],
            'ambiguity_ap': ar['candidate_ap'], 'ambiguity_first_rank': ar['candidate_first_rank'],
        })
    for name, s in [('uniform', u), ('ambiguity', a)]:
        assert abs(sum(r[name + '_ap'] for r in rows) * 100 / len(rows) - s['metrics']['mAP']) < 1e-5
        for rank in (1, 5, 10):
            value = sum(r[name + '_first_rank'] <= rank for r in rows) * 100 / len(rows)
            assert abs(value - s['metrics'][f'Rank-{rank}']) < 1e-5
    full = stats(rows)
    assert abs(sum(g['contribution_to_all_query_delta_mAP_points'] for g in full['rank1_transition_groups'].values()) - full['delta_mAP_points']) < 1e-10
    partitions = defaultdict(list)
    for r in rows:
        partitions['global_rank1_correct' if r['global_first_rank'] == 1 else 'global_rank1_wrong'].append(r)
    strata = []
    for name, subset in partitions.items():
        item = stats(subset)
        item['stratum'] = name
        item['contribution_to_all_query_delta_mAP_points'] = item['delta_mAP_points'] * len(subset) / len(rows)
        strata.append(item)
    assert abs(sum(s['contribution_to_all_query_delta_mAP_points'] for s in strata) - full['delta_mAP_points']) < 1e-10
    results.append({'dataset': dataset, 'selected_epochs': {'uniform': u['candidate']['best_epoch'], 'ambiguity': a['candidate']['best_epoch']}, 'full': full, 'global_rank1_strata': strata, 'query_rows': rows})

OUT.mkdir()
value = {
    'status': 'TWO_DATASET_SAVED_QUERY_COMPARISON_COMPLETE',
    'created_at': datetime.now().astimezone().isoformat(),
    'inputs': input_hashes,
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'delta_direction': 'ambiguity minus uniform',
    'scope': 'Four completed seed42 full50 mAP-best endpoints, two datasets. Official benchmark consumed in development. Saved GT-scored query rows only, not independent binary/image replay. No NN/optimizer/GPU/SSH/report rerun or new training. Global rank1 strata describe benchmark difficulty; they are not the training ambiguity weights or a causal partition. No claims about RGBNT100, full cohort, statistical significance, teacher, inference comparator, or unique failure cause.',
    'results': results,
}
(OUT / 'SUMMARY.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
lines = ['# Saved query comparison: ambiguity minus uniform', '', value['scope'], '', '| Dataset | Best U / A | Delta mAP | Delta R1 | AP improved / worsened | Worst 5% AP delta |', '|---|---:|---:|---:|---:|---:|']
for result in results:
    f = result['full']
    e = result['selected_epochs']
    lines.append(f'| {result["dataset"]} | {e["uniform"]} / {e["ambiguity"]} | {f["delta_mAP_points"]:+.8f} | {f["delta_Rank1_points"]:+.8f} | {f["query_AP_improved"]} / {f["query_AP_worsened"]} | {f["worst_ceil5percent_mean_delta_AP_points"]:+.8f} |')
    lines.extend(['', f'## {result["dataset"]}', '', '| Global stratum | Queries | Within-stratum delta AP | Contribution to overall delta mAP |', '|---|---:|---:|---:|'])
    for s in result['global_rank1_strata']:
        lines.append(f'| {s["stratum"]} | {s["query_count"]} | {s["delta_mAP_points"]:+.8f} | {s["contribution_to_all_query_delta_mAP_points"]:+.8f} |')
lines.extend(['', 'This descriptive comparison does not establish why weighting changed the outcomes. Any final decision awaits the original RGBNT100 endpoints; no rescue tuning or additional seeds are triggered.'])
(OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'status': value['status'], 'datasets': [{'dataset': r['dataset'], 'full': {k: v for k, v in r['full'].items() if k != 'identities'}, 'strata': [{k: s[k] for k in ('stratum', 'query_count', 'delta_mAP_points', 'contribution_to_all_query_delta_mAP_points')} for s in r['global_rank1_strata']]} for r in results]}))
