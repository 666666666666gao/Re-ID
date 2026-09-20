"""CPU check of every saved source row, not a model-gradient reconstruction."""
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import math
import sys
import torch

repo = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
sys.path.insert(0, str(repo))
from tools.msvr_smooth_ap import paired_objectives
from tools.run_signal_preserving_v5 import weighted_training_loss

root = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pipeline = json.loads(Path(str(root) + '_pipeline.json').read_bytes())
assert pipeline['status'] == 'COMPLETE' and pipeline['exit_code'] == 0
summary = json.loads((root / 'summary.json').read_bytes())
assert summary['status'] == 'COMPLETE_SOURCE_OBJECTIVE_GRADIENTS'
specpath = repo / 'configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json'
spec = json.loads(specpath.read_bytes())
assert sha(specpath) == summary['contract_sha256']
for name, digest in spec['project_file_sha256'].items():
    assert sha(repo / name) == digest, name
protocol = json.loads((repo / spec['protocol']).read_bytes())
assert sha(Path(spec['q1_summary'])) == spec['q1_summary_sha256']
q1 = json.loads(Path(spec['q1_summary']).read_bytes())
config = json.loads((repo / 'configs/MSVR310/TriFusion-source-style-paired-v1.json').read_bytes())
expected = [f'fold_{f}_{e}' for f in range(3) for e in ('control', 'smooth_ap')]
assert [c['directory'] for c in summary['conditions']] == expected
torch.set_num_threads(4)
result = dict(status='RUNNING', steps=0, role_rows=0, distance_values=0,
              max_loss_error=0., max_comparison_identity_error=0., conditions=[],
              optimizer_updates=0, model_forwards=0,
              scope='All 1560 saved distance/loss/identity/memory rows and 4680 role-statistic rows; scalar consistency only, no full-source parameter vectors persisted.')
for number, cond in enumerate(summary['conditions']):
    fold = protocol['folds'][number // 2]
    end = ('control', 'smooth_ap')[number % 2]
    directory = root / cond['directory']
    assert cond['steps'] == 260 and cond['model_state_unchanged'] and cond['gradients_absent']
    assert cond['optimizer_updates'] == cond['heldout_record_forwards'] == cond['official_image_reads'] == 0
    assert cond['scale'] == 256 and len(cond['parameters']) == 189
    assert cond['observed_records'] == sorted(fold['source_record_indices'])
    assert cond['direct_history_proof']['relative_error'] <= .005
    for name, entry in cond['files'].items():
        p = directory / name
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], p
    rows = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
    original = q1['folds'][number // 2]['endpoints'][end]
    original_path = Path(original['checkpoint']).parent / 'memory_steps.jsonl'
    assert sha(original_path) == original['training']['audit_files']['memory_steps.jsonl']['sha256']
    original_rows = [json.loads(line) for line in original_path.read_text().splitlines()]
    assert len(rows) == 260 and [r['step'] for r in rows] == list(range(1, 261))
    assert cond['model_state_sha256'] == original['training']['final_state_sha256']
    data = torch.from_file(str(directory / 'distances.f32'), shared=False,
                           size=(directory / 'distances.f32').stat().st_size // 4, dtype=torch.float32)
    used = 0
    memory = OrderedDict()
    seen = set()
    for step, row in enumerate(rows):
        indices = row['record_indices']; seen.update(indices)
        assert len(indices) == 64 and set(indices) <= set(fold['source_record_indices'])
        assert indices == original_rows[step]['record_indices']
        assert row['pixel_sha256'] == original_rows[step]['pixel_sha256']
        assert row['identities'] == [protocol['records'][i]['identity'] for i in indices]
        assert row['epoch'] == step // 13 + 1 and row['scale'] == 256
        for i in list(memory):
            if step - memory[i] > 8: del memory[i]
        expected_memory = [dict(record_index=i, identity=int(protocol['records'][i]['identity']),
                                scene=int(protocol['records'][i]['scene']), age=step - age,
                                stored_step=age) for i, age in memory.items() if i not in set(indices)]
        assert row['memory'] == expected_memory
        assert row['active_fused_metric'] == (end if step >= 65 else 'control')
        assert row['distance_offset_bytes'] == used * 4
        count = len(expected_memory)
        part = data[used:used + row['distance_float_count']]
        assert len(part) == 64 * (64 + count) and bool(torch.isfinite(part).all())
        dc = part[:4096].reshape(64, 64); dh = part[4096:].reshape(64, count)
        hard, smooth, _ = paired_objectives(dc, dh, row['identities'], [r['identity'] for r in expected_memory])
        target = hard if row['active_fused_metric'] == 'control' else smooth
        other = dict(row['components']); other['triplet_fused'] = 0.
        errors = [abs(float(target) - row['fused_loss']),
                  abs(float(weighted_training_loss(row['components'], config)) - row['total_loss']),
                  abs(float(weighted_training_loss(other, config)) - row['other_loss'])]
        assert max(errors) <= 2e-6, (cond['directory'], step, errors)
        result['max_loss_error'] = max(result['max_loss_error'], *errors)
        for k in ('current_decomposition', 'full_decomposition'):
            assert row[k]['relative_to_sum_of_component_norms'] <= .005
        assert set(row['roles']) == {'cnn', 'transformer', 'mamba'}
        for role in row['roles'].values():
            for comparison in role.values():
                a, b, d = (comparison[k] for k in ('first_norm', 'second_norm', 'difference_norm'))
                assert all(math.isfinite(x) and x >= 0 for x in (a, b, d))
                cosine = comparison['cosine']
                if a == 0 or b == 0:
                    assert cosine is None
                else:
                    assert math.isfinite(cosine) and abs(cosine) <= 1 + 1e-10
                    error = abs(d*d - (a*a+b*b-2*a*b*cosine)) / (1+a*a+b*b)
                    assert error < 1e-10
                    result['max_comparison_identity_error'] = max(result['max_comparison_identity_error'], error)
            result['role_rows'] += 1
        used += row['distance_float_count']; result['steps'] += 1
        if step >= 65:
            for i in indices:
                memory.pop(i, None); memory[i] = step
            while len(memory) > 512: memory.popitem(last=False)
    assert seen == set(fold['source_record_indices']) and used == len(data)
    result['distance_values'] += used
    result['conditions'].append(dict(directory=cond['directory'], steps=260, roles=780,
                                     unique_source_records=len(seen), distance_values=used))
    print(json.dumps(result['conditions'][-1]), flush=True)
    del data
assert result['steps'] == 1560 and result['role_rows'] == 4680
result.update(status='PASS_COMPLETE_SOURCE_OBJECTIVE_GRADIENT_LEDGER',
              summary_sha256=sha(root / 'summary.json'))
target = root / 'source_verification.json'; assert not target.exists()
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
