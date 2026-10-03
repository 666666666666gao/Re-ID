"""List nine exact M0 probes of the already accepted shared/private panel."""
from pathlib import Path
import ast
import hashlib
import json

base = Path('C:/Users/gb/.codex_tmp')
out = base / 'independent_evidence_draft/storage770'
assert not out.exists()
out.mkdir()
inventory = json.loads((base / 'metric_feature_scale_failed_closeout766/storage_inventory/INVENTORY.json').read_bytes())
summary_path = base / 'shared_private_complete714_intake_20261001/results/shared_private_evidence_complete_20261001/SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
assert summary['accepted'] == len(summary['rows']) == 9
assert all(row['status'] == 'VERIFIED_COMPLETE' for row in summary['rows'])
root = '/data/gaob/Re-ID/Trifusion'
closed = {'local': str(summary_path), 'remote': root + '/results/shared_private_evidence_complete_20261001/SUMMARY.json',
    'sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    'campaign': root + '/logs/shared_private_evidence_20261001_v2', 'accepted': 9}
candidates, bests = [], []
for row in summary['rows']:
    if row['dataset'] == 'RGBNT201':
        directory = 'shared_private_preflight_20261001_v2_RGBNT201_' + row['variant'] + '_m0'
    else:
        directory = Path(row['run_dir']).name.removesuffix('_full') + '_m0'
    path = root + '/trained-model/' + directory + '/m0_reload_probe.pth'
    item = next(v for v in inventory['weights'] if v['path'] == path)
    assert item['training_status'] == 'M0_PASS'
    candidates.append({'path': path, 'bytes': item['bytes'], 'sha256': item['recorded_probe_sha256'], 'original_summary': closed['remote']})
    bests.append({'path': row['run_dir'] + '/best_map.pth', 'sha256': row['checkpoint_sha256']})
assert len({row['path'] for row in candidates}) == 9
f3 = json.loads((base / 'metric_feature_scale_complete_v1/INTAKE.json').read_bytes())
previous = json.loads((base / 'metric_feature_scale_failed_closeout766/retirement/SPEC.json').read_bytes())
protected_f3 = previous['protected_f3']
assert not {row['path'] for row in candidates} & {row['path'] for row in protected_f3}
spec = {'root': root, 'port': 2026, 'candidates': candidates, 'closed_summaries': [closed],
    'protected_best': bests, 'protected_f3': protected_f3, 'f3_source_sha256': f3['source_sha256'],
    'public_clip': previous['public_clip'], 'remote_receipt': root + '/logs/closed_shared_private_m0_retirement770_20261003/RETIREMENT.json',
    'retired_bytes': sum(row['bytes'] for row in candidates),
    'boundary': 'Only nine closed accepted shared/private engineering probes. Exact formal bests, all current F3 binaries, public/author weights and all source/text/distances retained. No new plan references these M0 weights. Historical probe replay is retired; failure history is not overwritten.'}
(out / 'SPEC.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
source = (base / 'retire_closed_m0_766.py').read_text(encoding='utf-8')
source = source.replace("metric_feature_scale_failed_closeout766/retirement", "independent_evidence_draft/storage770")
source = source.replace("len(spec['candidates'])==24", "len(spec['candidates'])==9")
source = source.replace('RETIRED_EXACT_24_CLOSED_M0', 'RETIRED_EXACT_9_CLOSED_SHARED_PRIVATE_M0')
source = source.replace("'queue_clean_clip_joint.py','run_clean_clip_joint.py'", "'queue_clean_clip_joint.py','run_clean_clip_joint.py','queue_shared_private_evidence.py','run_shared_private_evidence.py','queue_independent_native_evidence.py','run_independent_native_evidence.py'")
script = base / 'retire_closed_shared_private_m0_770.py'
assert not script.exists()
script.write_text(source, encoding='utf-8')
ast.parse(source)
print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'candidates': 9, 'bytes': spec['retired_bytes'], 'spec': str(out / 'SPEC.json')}, indent=2))
