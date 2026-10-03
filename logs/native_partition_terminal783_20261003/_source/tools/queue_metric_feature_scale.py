"""F3: six fresh M0s, then six full50 controls; 2026 GPU0-3 only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_training_feature_scale as f2

base = f2.base
SCHEMA = 'trifusion-metric-feature-scale-v1'
DATASETS = base.DATASETS
RECIPES = ('normalized', 'metric_raw')
SOURCE, WEIGHTS, PROTOCOLS = f2.SOURCE, f2.WEIGHTS, f2.PROTOCOLS
PREVIOUS = ROOT/'logs/training_feature_scale_20261003_v2'
SOURCES = ('tools/run_metric_feature_scale.py', 'tools/queue_metric_feature_scale.py',
           'tools/check_metric_feature_scale_pair.py', 'tools/report_metric_feature_scale.py',
           'refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md',
           'refine-logs/metric_feature_scale_v1/EXPERIMENT_CODE_REVIEW.md')


def source_map():
    original = json.loads((PREVIOUS/'manifest.json').read_text())['source_sha256']
    assert len(original) == 259 and f2.source_map() == original
    return dict(original, **{name:base.sha(ROOT/name) for name in SOURCES})


def command(dataset, recipe, mode, campaign, output):
    return [sys.executable, '-B', str(ROOT/'tools/run_metric_feature_scale.py'),
            '--dataset',dataset,'--recipe',recipe,'--mode',mode,
            '--protocol',str(PROTOCOLS/f'{dataset}.json'),'--signal-source',str(SOURCE),
            '--clip-weight',str(WEIGHTS/'ViT-B-16.pt'),
            '--initialization',str(campaign/'initialization'/f'{dataset}_{recipe}.json'),
            '--output-dir',str(output),'--seed','42','--epochs','50']


def start_command(campaign, job, gpu):
    return [sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--campaign',str(campaign),
            '--phase',job['phase'],'--dataset',job['dataset'],'--variant',job['variant'],'--gpu',str(gpu)]


def run_phase(campaign, state, phase):
    pending = [job for job in state['jobs'] if job['phase']==phase]
    active = []
    failed = False
    state.update(status='RUNNING',phase=phase,updated_at=base.queue.stamp())
    while pending or active:
        for job,process in list(active):
            code = process.poll()
            if code is None:
                continue
            if code == 0:
                child = base.queue.child_campaign(campaign,phase,job['dataset'],job['variant'])
                job['result'] = base.require_complete(child,job['dataset'])
            job.update(status='FAILED' if code else 'COMPLETE',exit_code=code,
                       completed_at=base.queue.stamp())
            failed |= code != 0
            active.remove((job,process))
            print(json.dumps({'event':'completed',**job}),flush=True)
        if not failed:
            memory = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                               '--format=csv,noheader,nounits'],text=True)
            occupied = {job['gpu'] for job,_ in active}
            available = [int(row.split(',')[0]) for row in memory.splitlines()
                         if int(row.split(',')[0]) in range(4)
                         and int(row.split(',')[0]) not in occupied and int(row.split(',')[1])<500]
            for gpu in available:
                if not pending:
                    break
                assert shutil.disk_usage(ROOT).free >= 10*1024**3
                job = pending.pop(0)
                job.update(gpu=gpu,status='RUNNING',started_at=base.queue.stamp(),
                           command=start_command(campaign,job,gpu))
                with (campaign/f"{phase}_{job['variant']}_{job['dataset']}.log").open('x') as log:
                    process = subprocess.Popen(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job['pid'] = process.pid
                active.append((job,process))
                print(json.dumps({'event':'started',**job}),flush=True)
        state['updated_at'] = base.queue.stamp()
        if failed:
            state['status'] = 'FAILED'
        base.queue.write(campaign/'campaign.json',state)
        if failed and not active:
            return 1
        if pending or active:
            time.sleep(240)
    return 0


def configure():
    base.SCHEMA,base.RECIPES = SCHEMA,RECIPES
    base.source_map,base.command,base.start_command = source_map,command,start_command
    base.coordinate = coordinate


def expected_binding(campaign, dataset, recipe):
    return base.expected_binding(campaign,dataset,recipe)


def coordinate(args):
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion'
    previous = json.loads((PREVIOUS/'campaign.json').read_text())
    assert previous['status']=='COMPLETE' and previous['report_invocations']==1 and previous['report_exit_code']==0
    assert not (Path('/proc')/str(previous['controller_pid'])).exists()
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    sources = source_map()
    initial = {}
    base.queue.write(args.campaign/'campaign.json',{'status':'INITIALIZING','controller_pid':os.getpid(),'jobs':[]})
    for dataset in DATASETS:
        for recipe in RECIPES:
            with (args.campaign/f'prepare_{recipe}_{dataset}.log').open('x') as log:
                subprocess.run(command(dataset,recipe,'prepare',args.campaign,ROOT/'trained-model/f3_prepare_unused'),
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
            path = args.campaign/'initialization'/f'{dataset}_{recipe}.json'
            initial[str(path)] = base.sha(path)
        a,b = (expected_binding(args.campaign,dataset,recipe) for recipe in RECIPES)
        assert a['initial_model_state_sha256']==b['initial_model_state_sha256']
        with (args.campaign/f'initial_forward_pair_{dataset}.log').open('x') as log:
            subprocess.run([sys.executable,'-B',str(ROOT/'tools/check_metric_feature_scale_pair.py'),
                            '--campaign',str(args.campaign),'--dataset',dataset],cwd=ROOT,
                           env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        path = args.campaign/f'initial_forward_pair_{dataset}.json'
        initial[str(path)] = base.sha(path)
    assert sources==source_map()
    jobs = [{'phase':phase,'dataset':dataset,'variant':recipe,'status':'PENDING'}
            for phase in ('m0','full') for recipe in RECIPES for dataset in DATASETS]
    base.queue.write(args.campaign/'manifest.json',{'schema':SCHEMA,'seed':42,'epochs':50,
        'poll_seconds':240,'source_sha256':sources,'initialization_sha256':initial,'jobs':jobs,
        'previous_f2_manifest_sha256':base.sha(PREVIOUS/'manifest.json'),
        'boundary':'Fresh paired Triplet-input-only comparison under normalized BN/CE; L2 deployment; 2026 physical GPU0-3/max4. F2 historical results are context only.'})
    state = {'status':'RUNNING','controller_pid':os.getpid(),'started_at':base.queue.stamp(),
             'jobs':jobs,'report_invocations':0}
    for phase in ('m0','full'):
        code = run_phase(args.campaign,state,phase)
        if code:
            state.update(status='FAILED',completed_at=base.queue.stamp())
            base.queue.write(args.campaign/'campaign.json',state)
            return code
        if phase=='m0':
            for dataset in DATASETS:
                for recipe in RECIPES:
                    base.verify_m0(args.campaign,dataset,recipe)
    rows = [base.verify(args.campaign,dataset,recipe) for recipe in RECIPES for dataset in DATASETS]
    base.queue.write(args.campaign/'accepted_matrix.json',{'schema':SCHEMA,'accepted':6,'expected':6,'rows':rows})
    state.update(status='COMPLETE',completed_at=base.queue.stamp(),report_invocations=1)
    base.queue.write(args.campaign/'campaign.json',state)
    with (args.campaign/'report.log').open('x') as log:
        process = subprocess.run([sys.executable,'-B',str(ROOT/'tools/report_metric_feature_scale.py'),
                                  '--campaign',str(args.campaign),'--output-dir',str(args.report_dir)],cwd=ROOT,
                                 env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    state.update(report_exit_code=process.returncode,report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign/'campaign.json',state)
    return process.returncode


if __name__=='__main__':
    configure()
    raise SystemExit(base.main())
