"""Build full-training manifests from the audited official filename inventory."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def build(dataset, root):
    name = dataset['dataset']
    assert name in ('RGBNT201', 'MSVR310')
    assert dataset['train_test_identity_disjoint']
    assert dataset['three_modal_filename_sets_equal']
    manifest = dataset['record_manifest']
    split = 'train_171' if name == 'RGBNT201' else 'bounding_box_train'
    test_split = 'test' if name == 'RGBNT201' else 'bounding_box_test'
    source = manifest[split]
    source_ids = sorted({row['identity'] for row in source})
    test_ids = sorted({row['identity'] for row in manifest[test_split]})
    assert set(source_ids).isdisjoint(test_ids)
    expected = (3951, 171) if name == 'RGBNT201' else (1032, 155)
    assert (len(source), len(source_ids)) == expected
    label_map = {str(identity): i for i, identity in enumerate(source_ids)}
    records = []
    for i, row in enumerate(source):
        if name == 'RGBNT201':
            paths = [f'{split}/{m}/{row["path"]}' for m in ('RGB', 'NI', 'TI')]
            camera, environment = row['camera'] - 1, row['camera'] - 1
        else:
            parts = Path(row['path']).parts
            assert len(parts) == 3 and parts[1] == 'vis'
            paths = [f'{split}/{parts[0]}/{m}/{parts[2]}' for m in ('vis', 'ni', 'th')]
            camera, environment = row['camera'], row['scene']
        for path in paths:
            assert (root / name / path).is_file(), path
        records.append(dict(index=i, identity=row['identity'], label=label_map[str(row['identity'])],
                            camera=camera, scene=environment, paths=paths))
    assert len({tuple(r['paths']) for r in records}) == len(records)
    cameras = sorted({r['camera'] for r in records})
    assert cameras == (list(range(4)) if name == 'RGBNT201' else list(range(8)))
    return dict(schema='trifusion-official-full-train-v1', dataset=name,
                scope='complete official training identities; no model or test image access',
                dataset_root=str(root / name), environment_key='camera' if name == 'RGBNT201' else 'scene',
                source_split=split, records=records, source_ids=source_ids,
                source_label_map=label_map, source_record_indices=list(range(len(records))),
                source_camera_values=cameras, official_test_ids=test_ids,
                fold='full_train', source_count=len(records), identity_count=len(source_ids),
                camera_labels_zero_based=True, seed=42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = args.inventory.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.inventory_sha256
    inventory = json.loads(raw)
    selected = [d for d in inventory['datasets'] if d['dataset'] in ('RGBNT201', 'MSVR310')]
    assert len(selected) == 2 and len({d['dataset'] for d in selected}) == 2
    protocols = [build(d, args.data_root) for d in selected]
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    outputs = []
    for protocol in protocols:
        protocol['inventory_sha256'] = args.inventory_sha256
        payload = (json.dumps(protocol, ensure_ascii=False, indent=2) + '\n').encode()
        path = args.output_dir / (protocol['dataset'] + '_full_train.json')
        path.write_bytes(payload)
        outputs.append(dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                            records=protocol['source_count'], identities=protocol['identity_count']))
    receipt = dict(status='FULL_TRAIN_LABEL_MANIFESTS_BUILT', generated_at=datetime.now().astimezone().isoformat(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   inventory_sha256=args.inventory_sha256, outputs=outputs, model_forwards=0, optimizer_updates=0)
    (args.output_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
