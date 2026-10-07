from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import shutil

root = Path('/data/gaob/Re-ID/Trifusion')
campaign = root / 'logs/incremental_pending_m0_v1_20261007_880'
journal = root / 'logs/incremental_pending_m0_launch_20261007_880'
parent = root / 'logs/incremental_role_objective_m0_v1_20261007_878'
parent_journal = root / 'logs/incremental_role_objective_m0_launch_20261007_878'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((journal / 'EXIT.json').read_text())['exit_code'] == 0
child = json.loads((journal / 'CHILD.json').read_text())
assert not (Path('/proc') / str(child['pid'])).exists()
assert json.loads((parent_journal / 'EXIT.json').read_text())['exit_code'] == 1
state = json.loads((campaign / 'campaign.json').read_text())
assert state['status'] == 'M0_SCREENING_COMPLETE_WITH_ORIGINAL_FAILURE' and not state['formal_eligible']
assert len(state['jobs']) == 5 and state['new_qualified'] + state['new_failed'] == 5
scope = json.loads((root / 'refine-logs/incremental_role_objective_v1/PENDING_M0_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope) == 398 and all(sha(root / name) == digest for name, digest in scope.items())
controls = json.loads((root / 'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_text())
assert len(controls['artifact_sha256']) == 45 and all(sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
fixed = json.loads((root / 'refine-logs/incremental_role_objective_v1/FIXED_M0_SCALE_INPUT_SEAL.json').read_text())
assert len(fixed['artifact_sha256']) == 9 and all(sha(Path(name)) == digest for name, digest in fixed['artifact_sha256'].items())
files = {p for folder in (campaign, journal) for p in folder.rglob('*') if p.is_file()}
rows = []
specs = [('RGBNT201', 'md_batch_ratio', parent, 'ORIGINAL_ACTIVITY_GATE_FAIL')] + [
    (job['dataset'], job['objective'], campaign, job['status']) for job in state['jobs']]
for dataset, objective, owner, gate_status in specs:
    output = root / f'trained-model/{owner.name}_m0_{objective}_{dataset}'
    receipt = json.loads((output / 'training.json').read_text())
    assert receipt['status'] == 'M0_PASS' and receipt['production_m0_diagnostics']['effective_optimizer_updates'] == 8
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters'] == 281
    assert receipt['m0']['reload_max_abs_difference'] <= 1e-5
    assert all(value == 8 for value in receipt['production_m0_diagnostics']['author_bn_batches_tracked'].values())
    steps = [json.loads(line) for line in (output / 'training_steps.jsonl').read_text().splitlines()]
    assert len(steps) == 8 and all(row['incremental_isolated_shared_global_gradient_absent'] for row in steps)
    norms = {name: sum(row['incremental_isolated_query_key_gradient_norms'][name] for row in steps)
        for name in steps[0]['incremental_isolated_query_key_gradient_norms']}
    correction = sum(row['incremental_isolated_correction_gradient_norm'] for row in steps)
    assert len(norms) == 6 and all(math.isfinite(value) and value >= 0 for value in list(norms.values()) + [correction])
    active = correction > 0 and all(value > 0 for value in norms.values())
    assert active == (gate_status == 'QUALIFIED')
    old = root / f'trained-model/global_task_role_v1_20261004_824_m0_semantic_{dataset}/training_batch_order.jsonl'
    batches = [json.loads(line) for line in (output / 'training_batch_order.jsonl').read_text().splitlines()][:8]
    assert len(batches) == 8 and batches == [json.loads(line) for line in old.read_text().splitlines()][:8]
    probe = output / 'm0_reload_probe.pth'
    if owner == parent:
        assert probe.exists() and sha(probe) == receipt['m0']['reload_probe_sha256']
    else:
        assert not probe.exists()
        classified = owner / ('acceptance' if active else 'activity_failures') / f'{dataset}_{objective}.json'
        record = json.loads(classified.read_text())
        assert all(sha(Path(name)) == digest for name, digest in record['artifact_sha256'].items())
        assert record['probe']['sha256'] == receipt['m0']['reload_probe_sha256']
    files.update(p for p in output.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl', '.log', '.txt'))
    rows.append(dict(dataset=dataset, objective=objective, gate_status=gate_status,
        production_status=receipt['status'], optimizer_updates=8, gradient_tensors=281,
        reload_max_abs_difference=receipt['m0']['reload_max_abs_difference'],
        correction_gradient_norm_sum=correction, query_key_gradient_norm_sums=norms,
        source_batch_order_equal=True, probe_retained=(owner == parent)))
retirements = [json.loads(line) for line in (campaign / 'probe_retirement.jsonl').read_text().splitlines()]
assert len(retirements) == 5
print(json.dumps(dict(status='SIX_CONDITION_M0_MATRIX_PHYSICAL_SHA_VERIFIED', at=datetime.now().astimezone().isoformat(),
    source_count=398, current45_inputs_unchanged=True, fixed9_inputs_unchanged=True, rows=rows,
    qualified=sum(row['gate_status'] == 'QUALIFIED' for row in rows), failed=sum(row['gate_status'] != 'QUALIFIED' for row in rows),
    original_updates=8, new_updates=40, total_updates=48, formal_runs=0, formal_eligible=False,
    retired_probe_count=5, retired_probe_bytes=sum(row['bytes'] for row in retirements), free_bytes=shutil.disk_usage(root).free,
    files={str(p.relative_to(root)):dict(text=p.read_text(), sha256=sha(p)) for p in sorted(files)},
    boundary='Original first-eight failure retained without retry. Five previously pending conditions completed once with unchanged objective, scale, steps and isolated gate. Production M0 activity is distinct from auxiliary gate activity and retrieval performance. No formal training or new mAP.')))
