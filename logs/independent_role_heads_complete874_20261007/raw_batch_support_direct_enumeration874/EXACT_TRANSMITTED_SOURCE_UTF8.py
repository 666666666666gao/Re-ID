expected=[{'dataset': 'RGBNT201', 'totals': {'anchor_positions': 169536, 'ordinary_positive_positions': 1186752, 'legal_positive_positions': 198030, 'legal_negative_positions': 9494016, 'legal_directed_triplets': 11089680, 'anchors_with_legal_positive': 57416, 'anchors_with_multiple_positive_positions': 52929, 'anchors_with_multiple_distinct_positive_records': 52929, 'inventory_supported_anchor_positions': 58240, 'camera_proxy_supported_anchor_positions': 57416, 'camera_proxy_false_support': 0, 'camera_proxy_missed_support': 0, 'batches': 2649, 'duplicate_record_positions': 0}, 'zero_support_batches': 92, 'legal_positive_position_histogram': {'0': 112120, '1': 4487, '2': 10728, '3': 14480, '4': 14816, '5': 8688, '6': 3576, '7': 641}, 'distinct_legal_positive_record_histogram': {'0': 112120, '1': 4487, '2': 10728, '3': 14480, '4': 14816, '5': 8688, '6': 3576, '7': 641}}, {'dataset': 'MSVR310', 'totals': {'anchor_positions': 45184, 'ordinary_positive_positions': 135552, 'legal_positive_positions': 46550, 'legal_negative_positions': 2711040, 'legal_directed_triplets': 2793000, 'anchors_with_legal_positive': 21708, 'anchors_with_multiple_positive_positions': 17124, 'anchors_with_multiple_distinct_positive_records': 16986, 'inventory_supported_anchor_positions': 22852, 'camera_proxy_supported_anchor_positions': 44492, 'camera_proxy_false_support': 22804, 'camera_proxy_missed_support': 20, 'batches': 706, 'duplicate_record_positions': 4249}, 'zero_support_batches': 0, 'legal_positive_position_histogram': {'0': 23476, '1': 4584, '2': 9406, '3': 7718}, 'distinct_legal_positive_record_histogram': {'0': 23476, '1': 4722, '2': 9318, '3': 7668}}, {'dataset': 'RGBNT100', 'totals': {'anchor_positions': 400512, 'ordinary_positive_positions': 6007680, 'legal_positive_positions': 5174482, 'legal_negative_positions': 44857344, 'legal_directed_triplets': 579541984, 'anchors_with_legal_positive': 400512, 'anchors_with_multiple_positive_positions': 400512, 'anchors_with_multiple_distinct_positive_records': 400512, 'inventory_supported_anchor_positions': 400512, 'camera_proxy_supported_anchor_positions': 400512, 'camera_proxy_false_support': 0, 'camera_proxy_missed_support': 0, 'batches': 3129, 'duplicate_record_positions': 0}, 'zero_support_batches': 0, 'legal_positive_position_histogram': {'4': 12, '5': 33, '6': 40, '7': 324, '8': 1352, '9': 4977, '10': 14352, '11': 37560, '12': 77620, '13': 114984, '14': 105630, '15': 43628}, 'distinct_legal_positive_record_histogram': {'4': 12, '5': 33, '6': 40, '7': 324, '8': 1352, '9': 4977, '10': 14352, '11': 37560, '12': 77620, '13': 114984, '14': 105630, '15': 43628}}]
input_sha={'/data/gaob/Re-ID/Trifusion/refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json': '9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e', '/data/gaob/Re-ID/Trifusion/trained-model/global_task_role_v1_20261004_824_full_semantic_RGBNT201/training_batch_order.jsonl': 'e446282654c74949fb594856631b246642593b73229331c30b555418a0358993', '/data/gaob/Re-ID/Trifusion/trained-model/global_task_role_v1_20261004_824_full_semantic_RGBNT201/training.json': '0ae39806f2beb7d801e56b15ba2d83eb4475f169d868099b4e34770942a0029b', '/data/gaob/Re-ID/Trifusion/logs/training_feature_scale_protocols_20261002/RGBNT201.json': 'b2409c2d992d04adcf0c78354d306afd6af7ae9507738fd4e83de480e9a9b90f', '/data/gaob/Re-ID/Trifusion/trained-model/global_task_role_v1_20261004_824_full_semantic_MSVR310/training_batch_order.jsonl': '9c8a03beccbccedfa065cb13a3b20f5e64673d79693c4971034b54ab94e147e8', '/data/gaob/Re-ID/Trifusion/trained-model/global_task_role_v1_20261004_824_full_semantic_MSVR310/training.json': '257b5954f28b961bada050fbe8b1193976607ce39c79afff0feb49a3e26d0e85', '/data/gaob/Re-ID/Trifusion/logs/training_feature_scale_protocols_20261002/MSVR310.json': '015a3cafef8e36ba71da2784de6835959c0549c442624c8fb9e9b37bd8c8229d', '/data/gaob/Re-ID/Trifusion/trained-model/global_task_role_v1_20261004_824_full_semantic_RGBNT100/training_batch_order.jsonl': '75012100fd55f1da89d113c6159d99682a60deff8e92d11d54c7adfcd90da7b2', '/data/gaob/Re-ID/Trifusion/trained-model/global_task_role_v1_20261004_824_full_semantic_RGBNT100/training.json': 'be4b36f36cb70405245619804b75a2b2f3f02cafc65a0f06d71ce14b7f0e694e', '/data/gaob/Re-ID/Trifusion/logs/training_feature_scale_protocols_20261002/RGBNT100.json': 'b2c14adf947b06b27b9576a712f96192c1393ad6ed3ec1be5259b63f88300169'}
from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(Path(n))==d for n,d in input_sha.items())
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
verified=[]
for wanted in expected:
    dataset=wanted['dataset'];control=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==(dataset,'semantic'))
    protocol=json.loads((root/'logs/training_feature_scale_protocols_20261002'/f'{dataset}.json').read_text())
    environment=protocol['environment_key'];records=protocol['records']['train']
    lookup={Path(r['paths'][0]).name:r for r in records}
    eligible_identity={r['label'] for r in records if any(s['label']==r['label'] and s[environment]!=r[environment] for s in records)}
    totals=Counter();positive_hist=Counter();distinct_hist=Counter();zero_batches=0
    for line in (Path(control['run_dir'])/'training_batch_order.jsonl').read_text().splitlines():
        b=json.loads(line);rows=[lookup[name] for name in b['paths']];totals['batches']+=1
        totals['duplicate_record_positions']+=len(rows)-len(set(b['paths']));supported=0
        for i,r in enumerate(rows):
            pos=[j for j,s in enumerate(rows) if i!=j and s['label']==r['label'] and s[environment]!=r[environment]]
            negative=[j for j,s in enumerate(rows) if s['label']!=r['label']]
            ordinary=[j for j,s in enumerate(rows) if i!=j and s['label']==r['label']]
            camera_pos=[j for j,s in enumerate(rows) if i!=j and s['label']==r['label'] and s['camera']!=r['camera']]
            distinct={b['paths'][j] for j in pos};positive_hist[len(pos)]+=1;distinct_hist[len(distinct)]+=1
            supported+=bool(pos)
            totals.update(dict(anchor_positions=1,ordinary_positive_positions=len(ordinary),
                legal_positive_positions=len(pos),legal_negative_positions=len(negative),
                legal_directed_triplets=len(pos)*len(negative),anchors_with_legal_positive=int(bool(pos)),
                anchors_with_multiple_positive_positions=int(len(pos)>=2),
                anchors_with_multiple_distinct_positive_records=int(len(distinct)>=2),
                inventory_supported_anchor_positions=int(r['label'] in eligible_identity),
                camera_proxy_supported_anchor_positions=int(bool(camera_pos)),
                camera_proxy_false_support=int(bool(camera_pos) and not pos),
                camera_proxy_missed_support=int(not camera_pos and bool(pos))))
        zero_batches+=supported==0
    assert dict(totals)==wanted['totals'],dataset
    assert zero_batches==wanted['zero_support_batches']
    assert {str(k):v for k,v in sorted(positive_hist.items())}==wanted['legal_positive_position_histogram']
    assert {str(k):v for k,v in sorted(distinct_hist.items())}==wanted['distinct_legal_positive_record_histogram']
    verified.append(dict(dataset=dataset,batches=totals['batches'],anchors=totals['anchor_positions'],
        direct_position_enumeration_equal=True,duplicate_sensitive_distinct_counts_equal=True))
assert all(sha(Path(n))==d for n,d in input_sha.items())
print(json.dumps(dict(status='FULL_DIRECT_ENUMERATION_MATCH',at=datetime.now().astimezone().isoformat(),
    verified=verified,input_sha256=input_sha,model_forwards=0,optimizer_updates=0,
    boundary='Independent quadratic enumeration of every historical batch position pair confirms optimized count census. Label-only availability, not margin hardness, loss activity, gradients, current training success or causal retrieval evidence.')))
