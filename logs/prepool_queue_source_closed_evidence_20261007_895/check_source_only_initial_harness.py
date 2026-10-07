"""Private SOURCE_ONLY audit. No production imports, Torch, SSH or subprocesses."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

HERE = Path(__file__).resolve().parent
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
MAP_DIR = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891')
NEW = (
    'modeling/trifusion/prepool_dense_correspondence.py',
    'tools/run_prepool_dense_correspondence.py',
    'tools/queue_prepool_dense_correspondence.py',
    'tools/report_prepool_dense_correspondence.py',
)
LAUNCH = Path('C:/Users/gb/.codex_tmp/launch_prepool_dense895.py')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

checks = {'scope': 'SOURCE_ONLY', 'python': sys.version, 'source_sha256': {}, 'ast': {}}
for name in NEW:
    path = REPO / name
    ast.parse(path.read_text(encoding='utf-8'), filename=name)
    assert path.read_bytes() == (HERE / 'snapshot' / name).read_bytes()
    checks['source_sha256'][name] = sha(path)
    checks['ast'][name] = 'PASS'
ast.parse(LAUNCH.read_text(encoding='utf-8'))
assert LAUNCH.read_bytes() == (HERE / 'snapshot/launch_prepool_dense895.py').read_bytes()
checks['source_sha256'][str(LAUNCH)] = sha(LAUNCH)
checks['ast'][str(LAUNCH)] = 'PASS'
for node in ast.parse(LAUNCH.read_text(encoding='utf-8')).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ('supervisor', 'code') and isinstance(node.value, ast.Constant):
            tree = ast.parse(ast.literal_eval(node.value))
            suffixes = [list(map(ord, n.args[0].right.value)) for n in ast.walk(tree)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == 'write_text' and n.args
                        and isinstance(n.args[0], ast.BinOp)
                        and isinstance(n.args[0].right, ast.Constant)
                        and isinstance(n.args[0].right.value, str)]
            checks['ast']['nested_' + node.targets[0].id] = {'parse': 'PASS', 'json_suffix_codepoints': suffixes}

scope = json.loads((REPO / 'refine-logs/prepool_dense_correspondence_v1/SOURCE_SCOPE.json').read_bytes())
old_path = REPO / 'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_SOURCE_SCOPE.json'
old = json.loads(old_path.read_bytes())
assert len(old['source_sha256']) == 424
assert scope['inherited_424_scope_sha256'] == sha(old_path)
assert {k: v for k, v in scope['source_sha256'].items() if k not in NEW} == old['source_sha256']
mapping = json.loads((MAP_DIR / 'FILE_MAP.json').read_bytes())
counts = {'mapped_archive': 0, 'workspace_exact': 0, 'local_byte_mismatches_with_exact_archive': 0}
for name, digest in scope['source_sha256'].items():
    path = REPO / name
    if name in mapping:
        archived = MAP_DIR / mapping[name]['local_file']
        assert sha(archived) == digest
        counts['mapped_archive'] += 1
        if path.exists() and sha(path) != digest:
            counts['local_byte_mismatches_with_exact_archive'] += 1
            assert path.read_bytes().replace(b'\r\n', b'\n') == archived.read_bytes().replace(b'\r\n', b'\n')
    else:
        assert sha(path) == digest
        counts['workspace_exact'] += 1
checks['source_scope'] = dict(count=428, inherited_count=424, inherited_exact=True, **counts)
seal = json.loads((REPO / 'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_bytes())
assert len(seal['rows']) == 6 and len(seal['artifact_sha256']) == 45
assert sorted((r['dataset'], r['variant']) for r in seal['rows']) == sorted(
    (dataset, variant) for dataset in ('RGBNT201', 'MSVR310', 'RGBNT100') for variant in ('semantic', 'global_only'))
checks['control_seal'] = {'rows': 6, 'artifacts': 45, 'composition': '3 raw semantic + 3 independent global',
    'remote_artifact_bytes': 'NOT_READ; no SSH or Torch in this review'}

# Execute only the exact two AST-extracted controller functions below, with
# every launcher/resource/artifact boundary replaced by deterministic stdlib
# fixtures. This proves control flow, not actual training or source acceptance.
queue_path = REPO / 'tools/queue_prepool_dense_correspondence.py'
parsed = ast.parse(queue_path.read_text(encoding='utf-8'))
selected = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in ('coordinate', 'accepted_row')]
assert len(selected) == 2
code = compile(ast.Module(body=selected, type_ignores=[]), str(queue_path), 'exec')
fixtures = HERE / 'STDLIB_FIXTURES_R1'
fixtures.mkdir()
datasets = ('RGBNT201', 'MSVR310', 'RGBNT100')
schema = 'trifusion-prepool-dense-correspondence-v1'
replays = []

for case in ('formal_batch_witness_failure', 'initializer_binding_failure', 'ordinary_m0_child_failure'):
    case_dir = fixtures / case
    case_dir.mkdir()
    campaign = case_dir / 'campaign'
    report = case_dir / 'report'
    binding = {'seed': 42, 'added_model_parameters': 0, 'protocol_sha256': 'fixture_protocol'}
    controls = []
    for dataset in datasets:
        old_dir = case_dir / ('old_' + dataset)
        old_dir.mkdir()
        (old_dir / 'training_batch_order.jsonl').write_bytes(b'fixture_expected_batch\n')
        for variant in ('semantic', 'global_only'):
            controls.append({'dataset': dataset, 'variant': variant, 'initializer': binding, 'run_dir': str(old_dir)})
    control_path = case_dir / 'controls.json'
    control_path.write_text(json.dumps({'rows': controls, 'artifact_sha256': {}}))
    calls = []

    class Root:
        def __str__(self):
            return '/data/gaob/Re-ID/Trifusion'
        def __truediv__(self, part):
            return case_dir / part

    def write(path, value):
        path.write_text(json.dumps(value))

    def run_logged(camp, state, row, log_name):
        dataset = next(d for d in datasets if log_name.startswith(d + '_'))
        mode = row['mode']
        calls.append({'dataset': dataset, 'mode': mode})
        row.update(exit_code=0, status='COMPLETE')
        if mode == 'prepare':
            (camp / 'initialization').mkdir(exist_ok=True)
            b = dict(binding)
            if case == 'initializer_binding_failure' and dataset == datasets[0]:
                b['seed'] = 43
            write(camp / 'initialization' / (dataset + '.json'),
                  {'schema': schema, 'status': 'INITIALIZATION_VERIFIED', 'binding': b})
        if mode == 'm0' and case == 'ordinary_m0_child_failure' and dataset == datasets[0]:
            row.update(exit_code=1, status='FAILED')
            state['status'] = 'FAILED'
            write(camp / 'campaign.json', state)
            return 1
        if mode == 'accept-m0':
            (camp / 'acceptance').mkdir(exist_ok=True)
            write(camp / 'acceptance' / (dataset + '.json'), {'initializer': binding})
        if mode == 'evaluate':
            output = case_dir / f'trained-model/{camp.name}_full_{dataset}'
            output.mkdir(parents=True)
            metrics = {'mAP': 10, 'Rank-1': 20, 'Rank-5': 30, 'Rank-10': 40}
            condition = {'auxiliary_weight': 1.0}
            history = [{'epoch': epoch, 'official_fused': metrics} for epoch in range(1, 51)]
            write(output / 'training.json', {'schema': schema, 'status': 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE',
                'initializer': binding, 'condition': condition, 'history': history, 'best_epoch': 50})
            write(output / 'official_metrics.json', {'schema': schema, 'status': 'COMPLETE',
                'condition': condition, 'selected_epoch': 50, 'metrics': metrics, 'training_epochs': 50,
                'seed': 42, 'protocol_sha256': 'fixture_protocol', 'checkpoint_sha256': 'fixture_sha',
                'distance_sha256': 'fixture_sha', 'training_best_distance_sha256': 'fixture_sha',
                'independent_upstream_metrics_equal': True, 'reranking': False})
            write(output / 'own_global_metrics.json', {'schema': schema, 'status': 'COMPLETE', 'metrics': metrics,
                'same_original_fused_forward': True, 'new_global_NN_forwards': 0,
                'epoch_selection': 'same_fused_mAP_best', 'distance_sha256': 'fixture_sha'})
            data = b'fixture_wrong_batch\n' if case == 'formal_batch_witness_failure' and dataset == datasets[0] else b'fixture_expected_batch\n'
            (output / 'training_batch_order.jsonl').write_bytes(data)
        write(camp / 'campaign.json', state)
        return 0

    base = SimpleNamespace(queue=SimpleNamespace(stamp=lambda: 'SOURCE_ONLY_FIXTURE', write=write), sha=lambda p: 'fixture_sha')
    ns = dict(__file__=str(queue_path), ROOT=Root(), SCHEMA=schema, DATASETS=datasets, STORAGE_BYTES=3758096384,
        CONTROLS=control_path, Path=Path, json=json, sys=SimpleNamespace(executable='NEVER_EXECUTED'),
        os=SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': '0,1'}, getpid=lambda: 0),
        shutil=SimpleNamespace(disk_usage=lambda p: SimpleNamespace(free=3758096384)),
        subprocess=SimpleNamespace(run=lambda *a, **kw: SimpleNamespace(returncode=0)),
        base=base, previous=SimpleNamespace(run_logged=run_logged), source_map=lambda: {},
        command=lambda dataset, mode, camp, output: ['NEVER_EXECUTED', dataset, mode])
    exec(code, ns)
    failure = None
    exit_code = None
    try:
        exit_code = ns['coordinate'](SimpleNamespace(campaign=campaign, report_dir=report))
    except AssertionError as error:
        frames = []
        tb = error.__traceback__
        while tb:
            frames.append({'file': tb.tb_frame.f_code.co_filename, 'line': tb.tb_lineno,
                           'function': tb.tb_frame.f_code.co_name})
            tb = tb.tb_next
        failure = {'kind': type(error).__name__, 'frames': frames}
    state = json.loads((campaign / 'campaign.json').read_bytes())
    result = {'case': case, 'exception': failure, 'exit_code': exit_code, 'calls': calls,
        'visited_datasets': sorted({c['dataset'] for c in calls}),
        'persisted_job_statuses': {r['dataset']: r['status'] for r in state['jobs']},
        'accepted_matrix_written': (campaign / 'accepted_matrix.json').exists(),
        'report_log_written': (campaign / 'report.log').exists()}
    if case != 'ordinary_m0_child_failure':
        assert failure and result['visited_datasets'] == ['RGBNT201'] and not result['accepted_matrix_written']
    else:
        assert not failure and len(result['visited_datasets']) == 3 and result['accepted_matrix_written']
        result['matrix'] = json.loads((campaign / 'accepted_matrix.json').read_bytes())
    replays.append(result)

checks['replay_receipt'] = 'STDLIB_REPLAY.json'
checks['runtime_actions'] = {'production_imports': 0, 'torch_imports': 0, 'ssh': 0,
    'actual_subprocess_launches': 0, 'production_model_forwards': 0, 'optimizer_updates': 0}
(HERE / 'SOURCE_CHECKS.json').write_text(json.dumps(checks, indent=2) + '\n')
(HERE / 'STDLIB_REPLAY.json').write_text(json.dumps({'boundary': 'Exact coordinate + accepted_row AST functions; synthetic file receipts and mocked external boundaries. Not model/evaluator evidence.', 'cases': replays}, indent=2) + '\n')
print(json.dumps({'source_scope': checks['source_scope'], 'ast_files': len(checks['source_sha256']),
    'cases': [{k: r[k] for k in ('case', 'visited_datasets', 'accepted_matrix_written', 'persisted_job_statuses')} for r in replays]}))
