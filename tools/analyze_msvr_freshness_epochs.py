from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['preflight', 'source'], required=True)
parser.add_argument('--input-dir', type=Path, required=True)
args = parser.parse_args()
root = args.input_dir
complete = json.loads((root / 'local_complete_reaggregation.json').read_bytes())
cpu_path = root / args.mode / 'cpu_verification.json'
cpu = json.loads(cpu_path.read_bytes())
summary = json.loads((root / args.mode / 'summary.json').read_bytes())
assert complete['status'] == 'PASS_COMPLETE_FRESHNESS_TEXT_REAGGREGATION'
assert cpu['status'] == 'PASS_COMPLETE_FRESHNESS_' + args.mode.upper()
assert len(cpu['all_rows']) == complete['total_steps'] == (72 if args.mode == 'preflight' else 1560)
rows = cpu['all_rows']

def stats(values):
    if not values:
        return dict(count=0)
    v = np.asarray(values, dtype=np.float64)
    assert np.isfinite(v).all()
    return dict(count=len(v), mean=float(v.mean()), minimum=float(v.min()), median=float(np.median(v)),
                p05=float(np.quantile(v, .05)), p95=float(np.quantile(v, .95)), maximum=float(v.max()))

epochs = []
ages = []
role_rows = []
all_count = 0
for endpoint in summary['endpoints']:
    fold, end = endpoint['fold'], endpoint['endpoint']
    audits = [json.loads(x) for x in (root / args.mode / f'fold_{fold}_{end}' / 'audit.jsonl').read_text().splitlines()]
    for history in endpoint['training']['history']:
        epoch = history['epoch']
        part = [x for x in rows if (x['fold'], x['endpoint'], x['epoch']) == (fold, end, epoch)]
        audit_part = [x for x in audits if x['epoch'] == epoch]
        assert len(part) == len(audit_part) == history['updates']
        all_count += len(part)
        memory = [x for x in part if x['memory_records']]
        opportunity = Counter(age for x in memory for age in x['historical_ages'])
        candidates = sum(x['memory_records'] for x in memory)
        assert sum(opportunity.values()) == candidates
        loss_diff = stats([x['fresh_loss'] - x['stale_loss'] for x in memory])
        epoch_row = dict(fold=fold, endpoint=end, epoch=epoch, updates=len(part), history_steps=len(memory),
                         anchor_exposures=64 * len(memory), candidate_exposures=candidates,
                         distance_error_pair_weighted_mean=sum(x['distance_error_mean'] * x['memory_records'] for x in memory) / candidates if candidates else None,
                         distance_error_max=max(x['distance_error_max'] for x in memory) if memory else None,
                         fresh_minus_stale_loss=loss_diff,
                         original_trajectory_loss_difference=stats([x['original_trajectory_loss_difference'] for x in audit_part if x['original_trajectory_loss_difference'] is not None]),
                         elapsed_seconds=history['elapsed_seconds'], learning_rate=history['learning_rate'])
        for name in ['positive_winner_changes', 'negative_winner_changes', 'stale_wrong_fresh_correct', 'stale_correct_fresh_wrong',
                     'stale_positive_ties', 'stale_negative_ties', 'fresh_positive_ties', 'fresh_negative_ties']:
            epoch_row[name] = sum(x[name] for x in part)
        epochs.append(epoch_row)
        for age in range(1, 9):
            drift = [d for x in memory for a, d in zip(x['historical_ages'], x['drift'], strict=True) if a == age]
            assert len(drift) == opportunity[age]
            ages.append(dict(fold=fold, endpoint=end, epoch=epoch, age=age,
                             candidate_exposures=len(drift), candidate_share=len(drift) / candidates if candidates else None,
                             drift=stats(drift)))
        for role in ['cnn', 'transformer', 'mamba']:
            g = [x['parameter_gradients'][role] for x in memory]
            role_rows.append(dict(fold=fold, endpoint=end, epoch=epoch, role=role, history_steps=len(memory),
                                  fresh_cosine=stats([x['fresh']['cosine'] for x in g if x['fresh']['cosine'] is not None]),
                                  undefined_cosine=sum(x['fresh']['cosine'] is None for x in g),
                                  fresh_difference_norm=stats([x['fresh']['difference_norm'] for x in g]),
                                  duplicate_difference_norm=stats([x['duplicate_noise']['difference_norm'] for x in g]),
                                  exceeds_duplicate_count=sum(x['fresh']['difference_norm'] > x['duplicate_noise']['difference_norm'] for x in g)))
assert all_count == complete['total_steps']
assert len(epochs) == (6 if args.mode == 'preflight' else 120)
assert sum(x['candidate_exposures'] for x in ages) == sum(x['candidate_exposures'] for x in epochs)
report = dict(status='PASS_ALL_REGISTERED_EPOCH_AGE_TEXT_REAGGREGATION', mode=args.mode,
              total_steps=all_count, cpu_sha256=hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
              epochs=epochs, all_age_opportunities=ages, all_role_epochs=role_rows,
              limitations='All registered epochs, including no-history warmup, are retained. Age shares describe available candidates, not winner probabilities. Runtime gradient witnesses are not independent gradient recomputation. No new model or retrieval execution.')
(root / 'local_epoch_age_reaggregation.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())

def flatten(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            out.update({k + '_' + kk: vv for kk, vv in v.items()})
        else:
            out[k] = v
    return out

for name, values in [('epochs', epochs), ('ages', ages), ('role_epochs', role_rows)]:
    flat = [flatten(x) for x in values]
    columns = list(dict.fromkeys(k for x in flat for k in x))
    with (root / ('all_' + name + '.csv')).open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
print(json.dumps(dict(status=report['status'], mode=args.mode, steps=all_count, epoch_rows=len(epochs), age_rows=len(ages), role_rows=len(role_rows))))
