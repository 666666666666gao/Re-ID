"""Count static training-environment relation capacity, without sampler replay."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

TASK = Path('C:/Users/gb')
ROOT = TASK / '.trifusion_github_publish_22c3bee'
SOURCE = TASK / '.codex_tmp/clean_clip_audit_source721_20261002'
manifest_path = ROOT / 'logs/native_detail_start723_20261002/raw/logs/native_detail_20261002_v1/manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
rows = []
for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
    relative = f'logs/official_three_dataset_protocols_20260923/{dataset}.json'
    path = SOURCE / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['source_sha256'][relative]
    protocol = json.loads(path.read_text(encoding='utf-8'))
    field = protocol['environment_key']
    assert field == ('scene' if dataset == 'MSVR310' else 'camera')
    records = protocol['records']['train']
    assert len(records) == protocol['counts']['train']
    counts = defaultdict(Counter)
    for record in records:
        counts[record['identity']][record[field]] += 1
    assert len(counts) == len(protocol['train_label_map'])
    environments = sorted({r[field] for r in records})
    environment_records = Counter(r[field] for r in records)
    environment_identities = {e: {i for i, c in counts.items() if c[e]} for e in environments}
    multiple = {i for i, c in counts.items() if len(c) >= 2}
    pairs = []
    for a in environments:
        for b in environments:
            if a == b:
                continue
            shared = environment_identities[a] & environment_identities[b]
            supported = {i for i in shared if len(environment_identities[b] - {i}) > 0}
            pairs.append({'environment_A': a, 'environment_B': b,
                'shared_identity_count': len(shared),
                'identities_with_positive_and_negative_in_B': len(supported),
                'supported_anchor_records_in_A': sum(counts[i][a] for i in supported),
                'legal_positive_record_pairs_A_to_B': sum(counts[i][a] * counts[i][b] for i in supported)})
    assert sum(environment_records.values()) == len(records)
    rows.append({'dataset': dataset, 'protocol_sha256': digest,
        'environment_key': field, 'train_records': len(records), 'train_identities': len(counts),
        'environment_record_counts': dict(environment_records),
        'environment_identity_counts': {e: len(ids) for e, ids in environment_identities.items()},
        'identities_by_environment_count': dict(sorted(Counter(len(c) for c in counts.values()).items())),
        'multi_environment_identities': len(multiple), 'single_environment_identities': len(counts) - len(multiple),
        'anchor_records_with_a_cross_environment_positive_in_inventory': sum(sum(counts[i].values()) for i in multiple),
        'ordered_environment_pairs': pairs})

output = {'schema': 'trifusion-n3-static-training-environment-capacity-v1',
    'status': 'STATIC_INVENTORY_COUNT_COMPLETE', 'created_at': datetime.now().astimezone().isoformat(),
    'rows': rows, 'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'boundaries': ['Training annotations only; no query/gallery identities or scores used.',
        'MSVR310 scene is a time-period label, not camera or physical-scene truth.',
        'Static inventory capacity is not realized batch/role gradient support or evidence of meta-learning effectiveness.',
        'No data-loader or sampler replay, image/model forward, optimizer, GPU job or test-statistics update.',
        'Old step traces do not preserve batch identity/environment tuples; future actual support must be recorded before any N3 claim.']}
destination = ROOT / 'refine-logs/native_detail_v1/N3_SOURCE_CAPACITY_20261002.json'
assert not destination.exists()
destination.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': output['status'], 'rows': [{k: r[k] for k in (
    'dataset', 'train_records', 'train_identities', 'multi_environment_identities',
    'single_environment_identities', 'anchor_records_with_a_cross_environment_positive_in_inventory')} for r in rows]}))
