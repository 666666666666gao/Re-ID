"""Read-only postcheck of all thirteen recorded instance-memory statistics."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, config, protocol_path, output):
    assert not output.exists()
    spec = json.loads(config.read_bytes())
    summary = json.loads((root / 'summary.json').read_bytes())
    cpu = json.loads((root / 'cpu_verification.json').read_bytes())
    protocol = json.loads(protocol_path.read_bytes())
    assert sha(protocol_path) == spec['project_file_sha256']['protocols/msvr310_train_oof_v1.json']
    assert summary['config_sha256'] == sha(config)
    assert summary['status'] == 'PASS_COMPLETE_FIXED_STATE_PROBE'
    assert cpu['status'] == 'PASS_COMPLETE_FIXED_STATE_PROBE_CPU'
    assert cpu['summary_sha256'] == sha(root / 'summary.json')
    states = []
    total_rows = total_values = total_checks = 0
    for fold in summary['folds']:
        for name, receipt in fold['states'].items():
            directory = root / f"fold_{fold['fold']}_{name}"
            assert receipt == json.loads((directory / 'receipt.json').read_bytes())
            for filename, binding in receipt['files'].items():
                path = directory / filename
                assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
            rows = [json.loads(line) for line in (directory / 'steps.jsonl').read_text(encoding='utf-8').splitlines()]
            data = np.fromfile(directory / 'distances.f32', dtype=np.float32)
            assert len(rows) == receipt['batches']
            offset = 0
            totals = {}
            for row in rows:
                ids = np.asarray(row['identities'])
                scenes = np.asarray(row['scenes'])
                records = [protocol['records'][i] for i in row['record_indices']]
                assert [r['identity'] for r in records] == ids.tolist()
                assert [r['scene'] for r in records] == scenes.tolist()
                history = row['memory']
                mids = np.asarray([r['identity'] for r in history])
                mscenes = np.asarray([r['scene'] for r in history])
                n = len(ids)
                count = n*n + n*len(history)
                assert row['distance_offset_bytes'] == 4*offset and row['distance_float_count'] == count
                current = data[offset:offset+n*n].reshape(n, n)
                cached = data[offset+n*n:offset+count].reshape(n, len(history))
                assert np.isfinite(current).all() and np.isfinite(cached).all()
                positive = (ids[:, None] == ids[None, :]) & ~np.eye(n, dtype=bool)
                negative = ids[:, None] != ids[None, :]
                hp = np.where(positive, current, -np.inf).max(1)
                hn = np.where(negative, current, np.inf).min(1)
                mp = ids[:, None] == mids[None, :]
                if history:
                    ph = np.where(mp, cached, -np.inf).max(1)
                    nh = np.where(~mp, cached, np.inf).min(1)
                    up, un = np.maximum(hp, ph), np.minimum(hn, nh)
                    harder_positive, harder_negative = int((ph > hp).sum()), int((nh < hn).sum())
                else:
                    up, un = hp, hn
                    harder_positive = harder_negative = 0
                margin = np.float32(.3)
                expected = dict(
                    memory_records=len(history), memory_positive_pairs=int(mp.sum()),
                    memory_negative_pairs=int((~mp).sum()),
                    memory_cross_scene_positive_pairs=int((mp & (scenes[:, None] != mscenes[None, :])).sum()),
                    memory_negative_violations_against_batch_hard_positive=int(((~mp) & (cached < hp[:, None]+margin)).sum()),
                    harder_positive_anchors=harder_positive, harder_negative_anchors=harder_negative,
                    current_wrong_order_anchors=int((hp >= hn).sum()),
                    expanded_wrong_order_anchors=int((up >= un).sum()),
                    current_triplet=float(np.maximum(hp-hn+margin, 0).mean()),
                    expanded_triplet=float(np.maximum(up-un+margin, 0).mean()),
                    expanded_hinge_positive_anchors=int((up-un+margin > 0).sum()),
                    maximum_memory_age=max((r['age'] for r in history), default=0))
                assert set(expected) == set(row['statistics']) and len(expected) == 13
                for key, value in expected.items():
                    recorded = row['statistics'][key]
                    if key in ('current_triplet', 'expanded_triplet'):
                        assert abs(value-recorded) < 2e-6, (fold['fold'], name, row['step'], key)
                    else:
                        assert value == recorded, (fold['fold'], name, row['step'], key)
                    totals[key] = totals.get(key, 0) + value
                    total_checks += 1
                offset += count
            assert offset == len(data)
            total_rows += len(rows)
            total_values += len(data)
            states.append(dict(fold=fold['fold'], state=name, batches=len(rows), sum_of_per_batch_statistics=totals))
    assert len(states) == 9 and total_rows == cpu['batches'] and total_values == cpu['distance_elements']
    result = dict(status='PASS_ALL_THIRTEEN_MEMORY_STATISTICS', verified_at=datetime.now().astimezone().isoformat(),
                  summary_sha256=sha(root/'summary.json'), cpu_sha256=sha(root/'cpu_verification.json'),
                  config_sha256=sha(config), protocol_sha256=sha(protocol_path), verifier_sha256=sha(Path(__file__)),
                  batches=total_rows, distance_elements=total_values, statistic_checks=total_checks, states=states,
                  scope='All 13 statistics rederived from saved matrices/metadata; no model backward or 14-component loss reconstruction.',
                  model_forwards=0, optimizer_updates=0, official_image_reads=0)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'batches', 'distance_elements', 'statistic_checks')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.input_dir, args.config, args.protocol, args.output)
