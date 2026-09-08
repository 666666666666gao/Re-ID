"""Fresh-reviewer, read-only CPU verification; no author verifier imported."""
from collections import OrderedDict, Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
ART = ROOT.parent / 'artifacts/msvr310_role_set_gradient_check_v1_seed42_958fb21'
hashes = {}


def sha(path):
    path = Path(path).resolve()
    key = str(path)
    if key not in hashes:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(block)
        hashes[key] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    return hashes[key]['sha256']


def read(path):
    return json.loads(Path(path).read_bytes())


def absolute(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


configs = {}
checks = []


def check_binding(name, path, expected):
    actual = sha(path)
    checks.append({'declaration': name, 'path': str(Path(path).resolve()),
                   'expected': expected, 'actual': actual, 'pass': actual == expected})
    assert actual == expected, name


def config_tree(path):
    path = absolute(str(path)).resolve()
    if str(path) in configs:
        return
    cfg = read(path)
    configs[str(path)] = cfg
    sha(path)
    for field in ('source_sha256', 'project_file_sha256', 'project_source_file_sha256',
                  'project_files', 'fixed_file_sha256', 'fixed_files', 'signal_source_file_sha256'):
        if field in cfg:
            for name, expected in cfg[field].items():
                target = Path(cfg['signal_source']) / name if field == 'signal_source_file_sha256' else absolute(name)
                check_binding(str(path) + ':' + field + ':' + name, target, expected)

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and key.lower().endswith('_sha256'):
                    plain = key[:-7]
                    target = obj.get(plain)
                    if isinstance(target, str) and absolute(target).is_file():
                        check_binding(str(path) + ':' + key, absolute(target), value)
                if isinstance(value, str) and value.startswith('configs/') and value.endswith('.json'):
                    config_tree(value)
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    walk(cfg)


config_tree('configs/MSVR310/Role-set-gradient-check-v1.json')
config_tree('configs/MSVR310/Role-set-math-v1.json')
spec = read(ROOT / 'configs/MSVR310/Role-set-gradient-check-v1.json')
protocol = read(spec['protocol'])
records = protocol['records']
summary = read(ART / 'probe/summary.json')
cpu = read(ART / 'cpu.json')
pipeline = read(ART / 'pipeline.json')
assert summary['status'] == 'COMPLETE_ROLE_SET_PARAMETER_GRADIENT_CHECK'
assert pipeline['status'] == 'COMPLETE_FIXED_STATE_ENGINEERING_CHECK'
assert len(summary['folds']) == 3 and len(pipeline['stages']) == 2
assert all(s['exit_code'] == 0 for s in pipeline['stages'])
assert pipeline['config_sha256'] == summary['config_sha256'] == sha(ROOT / 'configs/MSVR310/Role-set-gradient-check-v1.json')
assert pipeline['cpu_sha256'] == sha(ART / 'cpu.json')
assert cpu['summary_sha256'] == sha(ART / 'probe/summary.json')
assert summary['optimizer_updates'] == summary['heldout_record_forwards'] == summary['official_image_reads'] == 0

# Dataset identity/camera/scene labels are independently parsed from filenames.
data_root = Path(protocol['dataset_root_at_inventory'])
id_scenes = {}
for idx, record in enumerate(records):
    assert record['index'] == idx
    assert len(record['paths']) == 3
    for mod, value in zip(('vis', 'ni', 'th'), record['paths']):
        p = Path(value)
        match = re.fullmatch(r'(\d+)_s(\d+)_v(\d+)_(\d+)\.jpg', p.name)
        assert match and tuple(map(int, match.groups()[:3])) == (record['identity'], record['scene'], record['camera'])
        assert p.parts[0] == 'bounding_box_train' and p.parts[2] == mod
        assert int(p.parts[1]) == record['identity'] and (data_root / p).is_file()
    id_scenes.setdefault(record['identity'], set()).add(record['scene'])
strata = [[i for i in sorted(id_scenes) if (len(id_scenes[i]) >= 2) == eligible] for eligible in (True, False)]

# Source checkpoint and source-only initialization lineage, without model execution.
history_cfg = read(absolute(spec['previous_config']))
coordinate_cfg = read(absolute(history_cfg['coordinate_config']))
memory_cfg = read(absolute(coordinate_cfg['memory_config']))
base_cfg = read(absolute(memory_cfg['base_config']))
baseline = read(base_cfg['BASELINE']['SUMMARY'])
meta = read(absolute(base_cfg['SOURCE_METADATA']['PATH']))
baseline_metadata = []
import torch
torch.set_num_threads(4)
for fold, b0, result in zip(protocol['folds'], baseline['folds'], summary['folds']):
    assert fold['fold'] == b0['fold'] == result['fold']
    expected_held = sorted(i for group in strata for i in group[fold['fold']::3])
    assert fold['heldout_ids'] == expected_held
    assert set(fold['source_ids']) == set(id_scenes) - set(expected_held)
    assert set(fold['source_record_indices']) == {i for i, r in enumerate(records) if r['identity'] in fold['source_ids']}
    assert set(fold['gallery_record_indices']) == {i for i, r in enumerate(records) if r['identity'] in fold['heldout_ids']}
    binding = result['initialization']
    assert binding['source_ids'] == fold['source_ids'] and binding['heldout_ids'] == fold['heldout_ids']
    assert binding['role_initialization_seed'] == 42 and binding['role_weights_loaded'] is False
    check_binding('source_checkpoint_fold_' + str(fold['fold']), b0['checkpoint'], b0['checkpoint_sha256'])
    payload = torch.load(b0['checkpoint'], map_location='cpu', weights_only=True)
    assert payload['fold'] == fold['fold'] and payload['source_ids'] == fold['source_ids'] and payload['heldout_ids'] == fold['heldout_ids']
    assert binding['signal_checkpoint_sha256'] == b0['checkpoint_sha256']
    baseline_metadata.append({'fold': fold['fold'], 'checkpoint': b0['checkpoint'],
                              'checkpoint_sha256': sha(b0['checkpoint']), 'source_ids': len(fold['source_ids']),
                              'heldout_ids': len(fold['heldout_ids']), 'checkpoint_identity_binding_pass': True,
                              'source_records': len(fold['source_record_indices']),
                              'parameter_tensors_declared': len(binding['trainable_names']),
                              'encoder_tensors_declared': sum(n.startswith('encoder.') for n in binding['trainable_names'])})
    del payload

fold_stats = []
batch_stats = []
total_elements = 0
all_unique = set()
max_loss_error = 0.
all_logs = [json.loads(s) for s in (ART / 'probe.log').read_text().splitlines() if s.startswith('{')]
assert [(x['fold'], x['step']) for x in all_logs] == [(f, s) for f in range(3) for s in range(1, 9)]
for fold, result in zip(protocol['folds'], summary['folds']):
    f = fold['fold']
    directory = ART / 'probe' / ('fold_' + str(f))
    receipt = read(directory / 'receipt.json')
    assert receipt == {k: v for k, v in result.items() if k != 'initialization'}
    assert receipt['batches'] == 8 and receipt['optimizer_updates'] == receipt['checkpoint_writes'] == 0
    assert receipt['initial_state_sha256'] == receipt['final_state_sha256'] == result['initialization']['initial_state_sha256']
    for name, value in receipt['files'].items():
        check_binding('fold_' + str(f) + ':' + name, directory / name, value['sha256'])
        assert (directory / name).stat().st_size == value['bytes']
    rows = [json.loads(s) for s in (directory / 'steps.jsonl').read_text().splitlines()]
    assert len(rows) == 8
    queue = OrderedDict()
    seen = set()
    local_stats = []
    with (directory / 'distances.f32').open('rb') as stream:
        for step, row in enumerate(rows):
            assert row['step'] == step + 1
            indices = row['record_indices']
            assert indices == meta['folds'][f]['batches'][step]['record_indices']
            assert len(indices) == 64 and set(indices) <= set(fold['source_record_indices'])
            ids = [records[i]['identity'] for i in indices]
            assert ids == row['identities'] and sorted(Counter(ids).values()) == [8] * 8
            seen.update(indices)
            for idx in [i for i, k in queue.items() if step - k > 8]:
                del queue[idx]
            expected_memory = [{'record_index': i, 'identity': records[i]['identity'], 'scene': records[i]['scene'],
                                'age': step-k, 'stored_step': k} for i, k in queue.items() if i not in set(indices)]
            assert row['memory'] == expected_memory
            memory = expected_memory
            m = len(memory)
            assert len({x['record_index'] for x in memory}) == m
            assert row['distance_offset_bytes'] == stream.tell()
            n = 4 * 64 * (64 + m)
            assert n == row['distance_float_count']
            a = np.fromfile(stream, dtype='<f4', count=n).reshape(4, 64, 64+m)
            assert np.isfinite(a).all() and float(a.min()) >= 0 and float(a.max()) <= 2.00001
            assert max(float(np.max(np.abs(x[:, :64] - x[:, :64].T))) for x in a) < 2e-6
            total_elements += n
            candidate_ids = ids + [x['identity'] for x in memory]
            proposed, counts, active, losses, hards, history_extra = [], [], [], [], [], 0
            for i in range(64):
                pos = [j for j, ident in enumerate(candidate_ids) if ident == ids[i] and j != i]
                neg = [j for j, ident in enumerate(candidate_ids) if ident != ids[i]]
                hp = max(float(a[0, i, j]) for j in pos)
                proposals = [min(neg, key=lambda j: (float(x[i, j]), j)) for x in a]
                selected = sorted(set(proposals))
                extra = [j for j in selected if j != proposals[0]]
                terms = [max(0., hp - float(a[0, i, j]) + .3) for j in selected]
                hards.append(max(0., hp-float(a[0, i, proposals[0]])+.3))
                losses.append(math.fsum(terms)/len(selected))
                proposed.append(proposals)
                counts.append(len(selected))
                active.append(sum(hp-float(a[0, i, j])+.3 > 0 for j in extra))
                history_extra += sum(j >= 64 and hp-float(a[0, i, j])+.3 > 0 for j in extra)
            assert proposed == row['proposals'] and counts == row['negative_counts'] and active == row['extra_active_counts']
            h, c = math.fsum(hards)/64, math.fsum(losses)/64
            err = max(abs(h-row['hard_loss']), abs(c-row['candidate_loss']))
            assert err < 2e-6 and c <= h + 1e-12
            max_loss_error = max(max_loss_error, err)
            for role in ('cnn', 'transformer', 'mamba'):
                for metric in row['roles'][role].values():
                    x, y, diff = (metric[k] for k in ('first_norm', 'second_norm', 'difference_norm'))
                    cosine = metric['cosine']
                    assert all(math.isfinite(v) and v >= 0 for v in (x,y,diff))
                    assert (cosine is None) == (x == 0 or y == 0)
                    if cosine is not None:
                        assert abs(cosine) <= 1.00001
                        assert abs(diff**2 - (x*x+y*y-2*x*y*cosine)) < 1e-7*max(1.,x*x+y*y)
            expected_groups = sorted(set(x['stored_step'] for x in memory))
            assert set(row['history_vjp_groups']) <= set(expected_groups)
            assert row['fresh_role_record_forwards'] == 64 * len(expected_groups)
            assert row['history_role_record_forwards'] == 64 * len(row['history_vjp_groups'])
            stat = {'fold': f, 'step': step+1, 'memory_records': m, 'memory_groups': len(expected_groups),
                    'history_vjp_groups': row['history_vjp_groups'], 'hard_loss': h, 'candidate_loss': c,
                    'active_extra_negative_exposures': sum(active), 'active_history_extra_negative_exposures': history_extra,
                    'anchors_with_extra_negatives': sum(x>1 for x in counts),
                    'maximum_selected_negatives': max(counts), 'roles': row['roles']}
            local_stats.append(stat)
            batch_stats.append(stat)
            if step >= 2:
                for idx in indices:
                    queue.pop(idx, None)
                    queue[idx] = step
                while len(queue) > 512:
                    queue.popitem(last=False)
        assert stream.read() == b''
    assert sorted(seen) == receipt['unique_source_records']
    all_unique.update(seen)
    for field, only_history in [('changed_gradient_batches_above_repeat_noise', False), ('changed_gradient_history_batches_above_repeat_noise', True)]:
        value = {e: sum((not only_history or bool(r['memory'])) and r['roles'][e]['hard_vs_role_set']['difference_norm'] > r['roles'][e]['role_set_repeat_noise']['difference_norm'] for r in rows) for e in ('cnn','transformer','mamba')}
        assert value == receipt[field]
    assert sum(x['active_extra_negative_exposures'] for x in local_stats) == receipt['active_extra_negative_exposures']
    proof = receipt['direct_history_chain_rule']
    cmp = proof['comparison']
    recomputed_relative = cmp['difference_norm']/max(cmp['first_norm'], cmp['second_norm'])
    assert recomputed_relative == proof['relative_l2_error'] and recomputed_relative <= proof['tolerance'] == .005
    assert proof['step'] == next(r['step'] for r in rows if r['memory']) == 4
    assert len({r['stored_step'] for r in rows[3]['memory']}) == 1
    assert receipt['current_role_record_forwards'] == 512
    assert receipt['extra_fresh_role_record_forwards'] == 64 + sum(r['fresh_role_record_forwards'] for r in rows)
    assert receipt['history_role_record_forwards'] == sum(r['history_role_record_forwards'] for r in rows)
    assert receipt['extra_direct_role_record_forwards'] == 64
    fold_stats.append({'fold': f, 'unique_source_records': len(seen), 'source_records': len(fold['source_record_indices']),
                       'source_identities_seen': len({records[i]['identity'] for i in seen}), 'batches': 8,
                       'history_batches': sum(bool(r['memory']) for r in rows),
                       'active_extra_negative_exposures': receipt['active_extra_negative_exposures'],
                       'active_history_extra_negative_exposures': sum(x['active_history_extra_negative_exposures'] for x in local_stats),
                       'direct_history_chain_rule': proof,
                       'peak_allocated_mib': receipt['peak_allocated_mib'],
                       'all8_gradient_changes_above_repeat': receipt['changed_gradient_batches_above_repeat_noise'],
                       'history5_gradient_changes_above_repeat': receipt['changed_gradient_history_batches_above_repeat_noise'],
                       'reported_state_before_equals_after': True})

files = []
for p in sorted(ART.rglob('*')):
    if p.is_file():
        files.append({'name': p.relative_to(ART).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
assert len(files) == 14 and all(not x['name'].endswith(('.pth','.pt','.npy')) for x in files)
assert sum(x['bytes'] for x in files) <= spec['maximum_expected_additional_disk_bytes']
assert pipeline['free_bytes_before'] >= spec['minimum_free_bytes']
assert int(pipeline['gpu_before'].split(',')[1]) < 500
assert total_elements == cpu['distance_elements'] == 834560
assert cpu['batches'] == len(batch_stats) == 24
result = {'status': 'PASS_INDEPENDENT_REMOTE_ARITHMETIC_AND_PROVENANCE_CHECKS',
          'checked_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
          'numpy': np.__version__, 'torch': torch.__version__, 'cpu_only': True,
          'author_verifier_imported': False, 'new_model_forwards': 0, 'new_optimizer_updates': 0,
          'new_artifact_writes_remote': 0, 'dataset_image_bytes_read': 0,
          'code_commit_now': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
          'run_commit': summary['project_commit'], 'baseline_metadata': baseline_metadata,
          'protocol_label_records_verified': len(records), 'dataset_path_existence_checks': 3*len(records),
          'unique_current_records_across_folds': len(all_unique),
          'batches': 24, 'current_view_exposures': 1536, 'distance_elements': total_elements,
          'independent_float64_max_loss_error': max_loss_error,
          'folds': fold_stats, 'batches_detail': batch_stats,
          'declared_hash_bindings': checks, 'configs_read': sorted(configs), 'audited_files': hashes,
          'artifact_files': files, 'artifact_total_bytes': sum(x['bytes'] for x in files),
          'process_exists_now': {str(pid): Path('/proc/'+str(pid)).exists() for pid in [pipeline['wrapper_pid']] + [x['original_pid'] for x in pipeline['stages']]},
          'free_bytes_now': shutil.disk_usage(ART).free,
          'limitations': ['No original model-gradient or state re-execution; stored gradient summaries remain runtime witnesses.',
                         'Filename labels and checkpoint metadata checked; no new independent dataset acquisition.',
                         'RNG states, historical fields and parameter-gradient tensors were not saved by the probe.']}
print(json.dumps(result, indent=2))
