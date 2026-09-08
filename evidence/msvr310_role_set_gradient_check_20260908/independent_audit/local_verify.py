"""Read and snapshot all supplied text; check intake, claims, hashes and aggregates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
INTAKE = Path('C:/Users/gb/.codex_tmp/role_set_gradient_check_complete_20260908')
OBSERVATION = Path('C:/Users/gb/.codex_tmp/role_set_gradient_check_observation_20260908_1825.json')
FILES = '''AGENTS.md
configs/MSVR310/Role-set-gradient-check-v1.json
configs/MSVR310/Role-set-math-v1.json
configs/MSVR310/TriFusion-history-gradient-paired-v1.json
protocols/msvr310_train_oof_v1.json
tools/msvr_role_set_relations.py
tools/check_msvr_role_set_relations.py
tools/probe_msvr_role_set_gradients.py
tools/verify_msvr_role_set_gradients.py
tools/run_msvr_role_set_gradients.py
tools/msvr_freshness_probe.py
tools/msvr_instance_memory.py
tools/probe_msvr_history_candidate_gradients.py
tools/train_msvr_history_gradient.py
tools/train_msvr_fresh_coordinate.py
tools/train_msvr_instance_memory.py
tools/train_msvr310_source_style.py
tools/train_msvr310_trifusion_oof.py
tools/train_msvr310_signal_oof.py
tools/run_signal_preserving_v5.py
tools/build_v12_complete_path_oof_targets.py
modeling/trifusion/signal_preserving_v8.py
modeling/trifusion/aligned_data.py
refine-logs/msvr310_role_set_v1/EXPERIMENT_PLAN.md
refine-logs/msvr310_role_set_v1/GRADIENT_CHECK_PLAN.md
refine-logs/msvr310_role_set_v1/EXPERIMENT_TRACKER.md
refine-logs/msvr310_role_set_v1/TRAINING_IMPLEMENTATION_DRAFT.md
results/MSVR310_ROLE_SET_MATH_2026-09-08.md
results/MSVR310_ROLE_SET_GRADIENT_CHECK_2026-09-08.md
evidence/msvr310_role_set_gradient_check_20260908/aggregate.json'''.splitlines()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


remote = read(OUT / 'remote_verification.json')
assert remote['status'] == 'PASS_INDEPENDENT_REMOTE_ARITHMETIC_AND_PROVENANCE_CHECKS'
manifest = []
loaded = {}
for label, path, target in ([('repo/' + s, ROOT/s, OUT/'snapshots/repo'/s) for s in FILES]
                           + [('intake/' + p.relative_to(INTAKE).as_posix(), p, OUT/'snapshots/intake'/p.relative_to(INTAKE)) for p in sorted(INTAKE.rglob('*')) if p.is_file()]
                           + [('observation.json', OBSERVATION, OUT/'snapshots/observation.json')]):
    data = path.read_bytes()
    text = data.decode('utf-8')
    if path.suffix == '.json':
        loaded[label] = json.loads(text)
    elif path.suffix == '.jsonl':
        loaded[label] = [json.loads(line) for line in text.splitlines()]
    else:
        loaded[label] = text
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    manifest.append({'input': str(path), 'snapshot': str(target), 'sha256': sha(data),
                     'bytes': len(data), 'lines': len(text.splitlines())})

intake = loaded['intake/intake.json']
pipeline = loaded['intake/pipeline.json']
summary = loaded['intake/probe/summary.json']
cpu = loaded['intake/cpu.json']
obs = loaded['observation.json']
assert intake['pipeline'] == pipeline == obs['pipeline']
assert obs['summary'] == summary
assert obs['log_tail'] == loaded['intake/probe.log']
assert intake['processes'] == obs['processes'] == remote['process_exists_now'] == {'33657':False,'33661':False,'33899':False}
assert intake['files'] == remote['artifact_files']
for item in intake['files']:
    if not item['name'].endswith('.f32'):
        data = (INTAKE/item['name']).read_bytes()
        assert len(data) == item['bytes'] and sha(data) == item['sha256']

rows = sum((loaded[f'intake/probe/fold_{f}/steps.jsonl'] for f in range(3)), [])
assert len(rows) == 24
aggregate = {'batches': len(rows), 'current_anchor_exposures': sum(len(r['record_indices']) for r in rows),
             'history_batches': sum(bool(r['memory']) for r in rows),
             'extra_active_exposures': sum(sum(r['extra_active_counts']) for r in rows), 'roles': {}}
for e in ('cnn','transformer','mamba'):
    stats = [r['roles'][e]['hard_vs_role_set'] for r in rows]
    repeats = [r['roles'][e]['role_set_repeat_noise'] for r in rows]
    aggregate['roles'][e] = {'cosine_min': min(s['cosine'] for s in stats), 'cosine_max': max(s['cosine'] for s in stats),
                            'difference_norm_min': min(s['difference_norm'] for s in stats),
                            'difference_norm_max': max(s['difference_norm'] for s in stats),
                            'repeat_difference_max': max(s['difference_norm'] for s in repeats),
                            'changed': sum(s['difference_norm'] > r['difference_norm'] for s,r in zip(stats,repeats))}
aggregate['all_fold_states_unchanged'] = all(f['initial_state_sha256'] == f['final_state_sha256'] for f in summary['folds'])
aggregate['direct_vjp_relative_error_max'] = max(f['direct_history_chain_rule']['relative_l2_error'] for f in summary['folds'])
aggregate['peak_allocated_mib'] = max(f['peak_allocated_mib'] for f in summary['folds'])
assert aggregate == loaded['repo/evidence/msvr310_role_set_gradient_check_20260908/aggregate.json']

local_hash_matches = []
for remote_name, receipt in remote['audited_files'].items():
    prefix = '/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'
    if not remote_name.startswith(prefix):
        continue
    name = remote_name[len(prefix):]
    p = ROOT / name
    if p.is_file():
        data = p.read_bytes()
        raw, lf = sha(data), sha(data.replace(b'\r\n', b'\n'))
        local_hash_matches.append({'path': name, 'remote_sha256': receipt['sha256'], 'local_sha256': raw,
                                   'lf_normalized_sha256': lf, 'match': 'byte_exact' if raw == receipt['sha256'] else 'LF_only' if lf == receipt['sha256'] else 'MISMATCH'})
assert all(x['match'] != 'MISMATCH' for x in local_hash_matches)

report = loaded['repo/results/MSVR310_ROLE_SET_GRADIENT_CHECK_2026-09-08.md']
assert all(s in report for s in ['333','316','350','432','418','421','1271','834560','3643844'])
assert remote['artifact_total_bytes'] == 3643844
assert [f['unique_source_records'] for f in remote['folds']] == [333,316,350]
assert cpu['max_loss_error'] == 2.9802322387695312e-08
elapsed = (datetime.fromisoformat(pipeline['ended_at'])-datetime.fromisoformat(pipeline['started_at'])).total_seconds()

result = {'status':'PASS_LOCAL_INTAKE_REMOTE_MANIFEST_AND_CLAIM_AGGREGATES',
          'checked_at':datetime.now(timezone.utc).isoformat(),
          'local_commit_now':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'run_commit':pipeline['code_commit'], 'pipeline_elapsed_seconds':elapsed,
          'local_input_count':len(manifest), 'intake_files_received':sum(x['input'].startswith(str(INTAKE)) for x in manifest),
          'all_intake_text_and_jsonl_parsed':True, 'observations_equal_primary_intake':True,
          'remote_manifest_equal_intake':True, 'all_aggregate_keys_and_numbers_equal':True,
          'aggregate_recomputed':aggregate, 'local_remote_code_and_config_matches':local_hash_matches,
          'audited_input_hashes':manifest,
          'limitations':['The original float32 CPU arithmetic loss residual is independently bounded, not numerically identical to this reviewer float64 arithmetic.',
                         'No original GPU parameter gradients or saved field/RNG tensors were independently rerun.']}
(OUT/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','local_input_count','pipeline_elapsed_seconds','aggregate_recomputed')},indent=2))
