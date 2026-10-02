"""F2: reuse the completed F1 queue, changing only training feature scaling."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_foundation_recipe as base

SCHEMA = 'trifusion-training-feature-scale-v1'
DATASETS = base.DATASETS
RECIPES = ('normalized', 'raw')
SOURCE, WEIGHTS = base.SOURCE, base.WEIGHTS
PROTOCOLS = ROOT/'logs/training_feature_scale_protocols_20261002'
PREDECESSOR = ROOT/'logs/foundation_complete739_20261002/raw/logs/foundation_recipe_20261002_v1/manifest.json'
SOURCES = ('tools/run_training_feature_scale.py', 'tools/queue_training_feature_scale.py',
    'tools/check_training_feature_scale_pair.py', 'tools/report_training_feature_scale.py',
    'tools/analyze_correspondence_distances.py', 'refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md',
    'refine-logs/training_feature_scale_v1/EXPERIMENT_CODE_REVIEW.md')
OTHER_RUNS = Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/three_seed_extension')
GPU_RELEASE = {
    0: 'RGBNT100_demo_s44',
    1: 'RGBNT100_ordinary_s44',
    2: 'RGBNT100_dual_s44',
    3: 'MSVR310_dual_s44',
}


def released_gpus():
    released = []
    for gpu, name in GPU_RELEASE.items():
        terminal = OTHER_RUNS/name/'evaluation_exit.json'
        if terminal.is_file() and json.loads(terminal.read_text())['exit_code'] == 0:
            released.append(gpu)
    return released


def run_phase(campaign, state, phase):
    pending = [job for job in state['jobs'] if job['phase'] == phase]
    active = []
    failed = False
    state.update(status='RUNNING', phase=phase, updated_at=base.queue.stamp())
    while pending or active:
        for job, process in list(active):
            code = process.poll()
            if code is None:
                continue
            if code == 0:
                folder = base.queue.child_campaign(campaign, phase, job['dataset'], job['variant'])
                job['result'] = base.require_complete(folder, job['dataset'])
            job.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code,
                       completed_at=base.queue.stamp())
            failed |= code != 0
            active.remove((job, process))
            print(json.dumps({'event':'completed', **job}), flush=True)
        if not failed:
            memory = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                               '--format=csv,noheader,nounits'], text=True)
            occupied = {job['gpu'] for job, _ in active}
            released = released_gpus()
            available = [int(row.split(',')[0]) for row in memory.splitlines()
                         if int(row.split(',')[0]) in released
                         and int(row.split(',')[0]) not in occupied and int(row.split(',')[1]) < 500]
            state['released_gpu_indices'] = released
            for gpu in available:
                if not pending:
                    break
                assert shutil.disk_usage(ROOT).free >= 10*1024**3
                job = pending.pop(0)
                job.update(gpu=gpu, status='RUNNING', started_at=base.queue.stamp(),
                           command=start_command(campaign, job, gpu))
                name = f"{phase}_{job['variant']}_{job['dataset']}"
                with (campaign/f'{name}.log').open('x', encoding='utf-8') as log:
                    process = subprocess.Popen(job['command'], cwd=ROOT,
                                               stdout=log, stderr=subprocess.STDOUT)
                job['pid'] = process.pid
                active.append((job, process))
                print(json.dumps({'event':'started', **job}), flush=True)
        state['updated_at'] = base.queue.stamp()
        if failed:
            state['status'] = 'FAILED'
        base.queue.write(campaign/'campaign.json', state)
        if failed and not active:
            return 1
        if pending or active:
            time.sleep(240)
    return 0


def source_map():
    old = json.loads(PREDECESSOR.read_text())['source_sha256']
    assert len(old) == 249
    sealed = json.loads(base.PREDECESSOR.read_text())['source_sha256']
    protocols = {f'logs/official_three_dataset_protocols_20260923/{name}.json' for name in DATASETS}
    result = {}
    for name, digest in old.items():
        actual = base.sha(ROOT/name)
        assert actual == (sealed[name] if name in protocols else digest), name
        result[name] = actual
    for dataset in DATASETS:
        name = f'logs/training_feature_scale_protocols_20261002/{dataset}.json'
        before = json.loads((base.PROTOCOLS/f'{dataset}.json').read_text())
        after = json.loads((ROOT/name).read_text())
        assert after['dataset_root'] == f'/data/gaob/Re-ID/dataset/{dataset}'
        assert {k:v for k,v in before.items() if k!='dataset_root'} == {
            k:v for k,v in after.items() if k!='dataset_root'}
        result[name] = base.sha(ROOT/name)
    return dict(result, **{name:base.sha(ROOT/name) for name in SOURCES})


def command(dataset, recipe, mode, campaign, output):
    return [sys.executable, '-B', str(ROOT/'tools/run_training_feature_scale.py'),
        '--dataset', dataset, '--recipe', recipe, '--mode', mode,
        '--protocol', str(PROTOCOLS/f'{dataset}.json'), '--signal-source', str(SOURCE),
        '--clip-weight', str(WEIGHTS/'ViT-B-16.pt'),
        '--initialization', str(campaign/'initialization'/f'{dataset}_{recipe}.json'),
        '--output-dir', str(output), '--seed', '42', '--epochs', '50']


def start_command(campaign, job, gpu):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--campaign', str(campaign),
        '--phase', job['phase'], '--dataset', job['dataset'], '--variant', job['variant'], '--gpu', str(gpu)]


def configure():
    base.SCHEMA, base.RECIPES = SCHEMA, RECIPES
    base.source_map, base.command, base.start_command = source_map, command, start_command
    base.coordinate = coordinate


def expected_binding(campaign, dataset, recipe):
    return base.expected_binding(campaign, dataset, recipe)


def coordinate(args):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert args.gpu in released_gpus()
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    sources = source_map()
    initial = {}
    base.queue.write(args.campaign/'campaign.json', {'status':'INITIALIZING','controller_pid':os.getpid(),'jobs':[]})
    for dataset in DATASETS:
        for recipe in RECIPES:
            with (args.campaign/f'prepare_{recipe}_{dataset}.log').open('x') as log:
                subprocess.run(command(dataset,recipe,'prepare',args.campaign,ROOT/'trained-model/f2_prepare_unused'),
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu)),stdout=log,
                    stderr=subprocess.STDOUT,check=True)
            path=args.campaign/'initialization'/f'{dataset}_{recipe}.json'
            initial[str(path)]=base.sha(path)
        a,b=(expected_binding(args.campaign,dataset,recipe) for recipe in RECIPES)
        assert a['initial_model_state_sha256']==b['initial_model_state_sha256']
        with (args.campaign/f'initial_forward_pair_{dataset}.log').open('x') as log:
            subprocess.run([sys.executable,'-B',str(ROOT/'tools/check_training_feature_scale_pair.py'),
                '--campaign',str(args.campaign),'--dataset',dataset],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        path=args.campaign/f'initial_forward_pair_{dataset}.json'
        initial[str(path)]=base.sha(path)
    assert sources==source_map()
    jobs=[{'phase':phase,'dataset':dataset,'variant':recipe,'status':'PENDING'}
          for phase in ('m0','full') for recipe in RECIPES for dataset in DATASETS]
    base.queue.write(args.campaign/'manifest.json',{'schema':SCHEMA,'seed':42,'epochs':50,
        'poll_seconds':240,'source_sha256':sources,'initialization_sha256':initial,'jobs':jobs,
        'gpu_release': {str(gpu):str(OTHER_RUNS/name/'evaluation_exit.json')
                        for gpu,name in GPU_RELEASE.items()},
        'boundary':'One training feature-scale control; both deployment embeddings L2; only 2026 GPU0-3.'})
    state={'status':'RUNNING','controller_pid':os.getpid(),'started_at':base.queue.stamp(),
           'jobs':jobs,'report_invocations':0}
    base.queue.start_command,base.queue.require_complete=start_command,base.require_complete
    for phase in ('m0','full'):
        code=run_phase(args.campaign,state,phase)
        if code:
            state.update(status='FAILED',completed_at=base.queue.stamp())
            base.queue.write(args.campaign/'campaign.json',state)
            return code
        if phase=='m0':
            for dataset in DATASETS:
                for recipe in RECIPES:
                    base.verify_m0(args.campaign,dataset,recipe)
    rows=[base.verify(args.campaign,dataset,recipe) for recipe in RECIPES for dataset in DATASETS]
    base.queue.write(args.campaign/'accepted_matrix.json',{'schema':SCHEMA,'accepted':6,'expected':6,'rows':rows})
    state.update(status='COMPLETE',completed_at=base.queue.stamp(),report_invocations=1)
    base.queue.write(args.campaign/'campaign.json',state)
    with (args.campaign/'report.log').open('x') as log:
        process=subprocess.run([sys.executable,'-B',str(ROOT/'tools/report_training_feature_scale.py'),
            '--campaign',str(args.campaign),'--output-dir',str(args.report_dir)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    state.update(report_exit_code=process.returncode,report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign/'campaign.json',state)
    return process.returncode


if __name__=='__main__':
    configure()
    raise SystemExit(base.main())
