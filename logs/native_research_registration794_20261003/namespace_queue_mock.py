"""Stdlib-only review: execute exact AST control functions with fake compute.

No source imports, torch imports, model construction, CUDA, subprocess launch,
remote connection, or package installation. Fixtures live in this review folder.
"""
import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
OUT = Path(__file__).resolve().parent


def definitions(relative, names, namespace):
    tree = ast.parse((REPO / relative).read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPO / relative), 'exec'), namespace)


def module(name, **values):
    value = ModuleType(name)
    value.__dict__.update(values)
    return value


def cli_path(version, mode):
    events = []
    target = OUT / 'fixtures' / f'{version}_{mode}'
    target.mkdir(parents=True, exist_ok=True)
    diagnostics = SimpleNamespace(result=lambda: events.append('production_m0_diagnostics.result') or {'effective_optimizer_updates': 8})
    runner = SimpleNamespace(sha256=lambda _path: 'mock-file-sha', read_protocol=lambda *_args: {'mock': True})
    clean = SimpleNamespace(control=SimpleNamespace())

    def foundation_train(args, _protocol):
        events.append('foundation.train.original_body')
        assert foundation.build_core is native.build_core
        assert foundation.optimization is partition.optimization
        assert foundation.SCHEMA == native.SCHEMA
        (args.output_dir / 'training.json').write_text(json.dumps({'status': 'M0_PASS' if mode == 'm0' else 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'}))

    foundation = module('foundation', train=foundation_train,
        evaluate=lambda *_args: events.append('foundation.evaluate'))
    common = dict(argparse=argparse, datetime=datetime, Path=Path, json=json, sys=sys)
    native = module('native', **common, foundation=foundation, clean=clean, runner=runner,
        SCHEMA='v1', VARIANTS=('global_only', 'semantic', 'native'),
        RawFeatureSemanticTriFusion=object(), M0_DIAGNOSTICS=diagnostics,
        build_core=lambda *_: (None, None, {}), condition=object(), train_loader=object(),
        optimization=object(), loss_values=object())
    definitions('tools/run_independent_native_evidence.py', ('train', 'configure', 'main'), native.__dict__)
    cuda = SimpleNamespace(max_memory_allocated=lambda _device: 1, max_memory_reserved=lambda _device: 2)
    partition = module('partition', **common, entry=native, SCHEMA='v5',
        original_foundation_train=foundation.train, torch=SimpleNamespace(cuda=cuda),
        build_core=lambda *_: (None, None, {}), condition=object(), optimization=object())
    definitions('tools/run_native_partitioned.py', ('train', 'configure'), partition.__dict__)
    research = module('research', **common, entry=partition, SCHEMA='trifusion-independent-native-research-v6',
        original_build_core=partition.build_core,
        __file__=str(REPO / 'tools/run_native_research.py'))
    definitions('tools/run_native_research.py', ('configure', 'build_core'), research.__dict__)
    entry = partition if version == 'v5' else research
    entry.configure()
    argv = ['review', '--dataset', 'RGBNT201', '--variant', 'native', '--mode', mode,
        '--protocol', str(target / 'protocol.json'), '--signal-source', str(target / 'source'),
        '--clip-weight', str(target / 'clip.pt'), '--initialization', str(target / 'initial.json'),
        '--output-dir', str(target), '--seed', '42', '--epochs', '50']
    original_argv, sys.argv = sys.argv, argv
    try:
        native.main()
    finally:
        sys.argv = original_argv
    receipt = json.loads((target / 'training.json').read_text()) if mode != 'evaluate' else None
    return {'events': events, 'receipt': receipt, 'schema': foundation.SCHEMA}


def queue_namespace():
    namespace = dict(argparse=argparse, json=json, os=os, Path=Path, sys=sys)
    definitions('tools/queue_native_research.py',
        ('coordinate', 'command', 'output_dir', 'configure', 'run_logged', 'wait_for_devices'), namespace)
    return namespace


def coordinate_mock(case, fail_mode=None):
    target = OUT / 'fixtures' / case
    assert not target.exists(), target
    target.mkdir(parents=True)
    commands, gates = [], []
    class Root:
        def __str__(self):
            return '/data/gaob/Re-ID/Trifusion'
        def __truediv__(self, child):
            return target / child
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    def result(_campaign, dataset, variant):
        return {'dataset': dataset, 'variant': variant}
    def verify(_campaign, dataset, variant):
        gates.append((dataset, variant))
        return result(_campaign, dataset, variant)
    base = SimpleNamespace(queue=SimpleNamespace(write=write, stamp=lambda: 'mock'),
        verify=result, sha=lambda _path: 'mock-sha')
    namespace = queue_namespace()
    def run_logged(campaign, state, row, _log_name):
        mode = row['mode']
        argv = row['command']
        dataset = argv[argv.index('--dataset') + 1]
        variant = argv[argv.index('--variant') + 1] if '--variant' in argv else None
        commands.append([dataset, variant, mode])
        if mode == 'prepare':
            write(campaign / 'initialization' / f'{dataset}_{variant}.json', {})
        if mode == 'initial_forward_pair':
            write(campaign / f'initial_forward_pair_{dataset}.json', {})
        code = int(fail_mode == (dataset, variant, mode))
        row.update(exit_code=code, status='FAILED' if code else 'COMPLETE')
        if code:
            state['status'] = 'FAILED'
        return code
    reports = []
    def report_run(argv, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        reports.append(argv)
        return SimpleNamespace(returncode=0)
    namespace.update(ROOT=Root(), DATASETS=('RGBNT201','MSVR310','RGBNT100'),
        VARIANTS=('global_only','semantic','native'), SCHEMA='v6', SOURCE=target/'source',
        WEIGHTS=target/'weights', PROTOCOLS=target/'protocols', RESERVE_BYTES=2*1024**3,
        CAMPAIGN_STORAGE_BYTES=18*384*1024**2+856*1024**2, base=base,
        shutil=SimpleNamespace(disk_usage=lambda _path: SimpleNamespace(free=100*1024**3)),
        source_map=lambda: {}, verify_m0=verify, run_logged=run_logged,
        subprocess=SimpleNamespace(run=report_run, STDOUT=-2))
    namespace['configure']()
    args = SimpleNamespace(campaign=target/'campaign', report_dir=target/'report')
    code = namespace['coordinate'](args)
    state = json.loads((args.campaign/'campaign.json').read_text())
    return {'exit_code':code, 'commands':commands, 'm0_gates':gates,
        'report_count':len(reports), 'terminal_status':state['status']}


def logging_and_wait_mock():
    namespace = queue_namespace()
    writes, sleeps, gpu_queries, launches = [], [], [], []
    readings = iter(['0, 800\n1, 12\n', '0, 12\n1, 10\n'])
    def query(argv, **_kwargs):
        gpu_queries.append(argv)
        return next(readings)
    base = SimpleNamespace(require_sources=lambda _campaign: None,
        queue=SimpleNamespace(stamp=lambda:'mock', write=lambda path,value:writes.append(json.loads(json.dumps(value)))))
    namespace.update(base=base, ROOT=OUT, RESERVE_BYTES=0,
        shutil=SimpleNamespace(disk_usage=lambda _path: SimpleNamespace(free=100)),
        time=SimpleNamespace(sleep=lambda seconds:sleeps.append(seconds)),
        subprocess=SimpleNamespace(check_output=query))
    state = {}
    namespace['wait_for_devices'](OUT, state)
    assert sleeps == [240] and state['status'] == 'RUNNING'
    assert all('--id=0,1' in argv for argv in gpu_queries)
    namespace['wait_for_devices'] = lambda *_args: None
    outputs = []
    for exit_code in (0, 7):
        def popen(argv, **kwargs):
            launches.append({'argv':argv, 'devices':kwargs['env']['CUDA_VISIBLE_DEVICES']})
            return SimpleNamespace(pid=1234, wait=lambda:exit_code)
        namespace['subprocess'] = SimpleNamespace(Popen=popen, STDOUT=-2)
        state, row = {'status':'RUNNING'}, {'command':['mock']}
        actual = namespace['run_logged'](OUT, state, row, f'mock_process_{exit_code}.log')
        assert actual == exit_code and 'active_command' not in state
        outputs.append({'returncode':actual, 'row':row, 'state':state})
    assert all(row['devices'] == '0,1' for row in launches)
    return {'sleep_seconds':sleeps, 'gpu_queries':gpu_queries, 'launches':launches, 'outputs':outputs}


def report_mock(dataset):
    routes = []
    data = {
        name: {'fused':SimpleNamespace(numpy=lambda name=name:name),
               'query_ids':[10,11], 'gallery_ids':[10,11,12],
               'query_cameras':[0,1], 'gallery_cameras':[1,0,2],
               'query_scenes':[0,1], 'gallery_scenes':[1,0,2]}
        for name in ('semantic','native')}
    scores = {'semantic':{'average_precision':[0.5,1.0], 'first_match_rank':[2,1]},
              'native':{'average_precision':[1.0,0.25], 'first_match_rank':[1,4]}}
    def scorer(kind):
        def call(distances, *_args):
            routes.append(kind)
            return scores[distances]
        return call
    loads = []
    def load(path, **kwargs):
        loads.append(kwargs)
        return data[path.parent.name]
    ns = dict(Path=Path, original_compare=lambda *_args:{'unchanged_original_diagnosis':True},
        report=SimpleNamespace(torch=SimpleNamespace(load=load)),
        diagnosis=SimpleNamespace(scene_scores=scorer('scene'),camera_scores=scorer('camera')))
    definitions('tools/report_native_research.py',('compare',),ns)
    matrix = {'rows':[{'variant':name,'dataset':dataset,'run_dir':name} for name in ('semantic','native')]}
    output = ns['compare'](matrix,dataset,'semantic','native')
    assert output['unchanged_original_diagnosis'] is True
    assert output['query_changes'] == [
        {'query_index':0,'identity':10,'control_ap':0.5,'candidate_ap':1.0,'control_first_rank':2,'candidate_first_rank':1},
        {'query_index':1,'identity':11,'control_ap':1.0,'candidate_ap':0.25,'control_first_rank':1,'candidate_first_rank':4}]
    assert routes == (['scene']*2 if dataset=='MSVR310' else ['camera']*2)
    assert all(item['map_location']=='cpu' for item in loads)
    return {'result':output,'scorer_routes':routes,'loads':loads}


result = {'scope':'SOURCE_AND_STDLIB_MOCK_ONLY', 'model_runtime':False, 'torch_imported':False}
result['cli'] = {f'{v}_{mode}':cli_path(v, mode) for v in ('v5','v6') for mode in ('m0','train','evaluate')}
assert result['cli']['v5_m0']['events'].count('production_m0_diagnostics.result') == 1
assert result['cli']['v6_m0']['events'].count('production_m0_diagnostics.result') == 1
assert result['cli']['v6_train']['events'] == ['foundation.train.original_body']
assert result['cli']['v6_evaluate']['events'] == ['foundation.evaluate']
result['queue_success'] = coordinate_mock('queue_success_r2')
expected = []
for dataset in ('RGBNT201','MSVR310','RGBNT100'):
    expected += [[dataset,variant,'prepare'] for variant in ('global_only','semantic','native')]
    expected += [[dataset,None,'initial_forward_pair']]
    expected += [[dataset,variant,mode] for variant in ('global_only','semantic','native') for mode in ('m0','train','evaluate')]
assert result['queue_success']['commands'] == expected
assert result['queue_success']['report_count'] == 1 and result['queue_success']['terminal_status'] == 'COMPLETE'
result['queue_failure'] = coordinate_mock('queue_failure_r2', ('RGBNT201','native','m0'))
assert result['queue_failure']['commands'] == expected[:11]
assert result['queue_failure']['report_count'] == 0 and result['queue_failure']['terminal_status'] == 'FAILED'
result['logging_wait'] = logging_and_wait_mock()
result['report_query_rows'] = {name:report_mock(name) for name in ('RGBNT201','MSVR310')}
result['status'] = 'PASS'
(OUT/'MOCK_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','v5_m0_diagnostic_writes':1,'v6_m0_diagnostic_writes':1,
    'success_command_count':len(expected),'failure_stops_after_command':len(result['queue_failure']['commands']),
    'gpu_devices':'0,1','resource_wait_seconds':240,'model_runtime':False}))
