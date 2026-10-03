"""Six matched clean-public-CLIP endpoints, M0 first, no retries."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_correspondence_refinement as queue
from tools.queue_correspondence_roles import PROTOCOLS, SOURCE, WEIGHTS
from tools.collect_correspondence_roles import sha

DATASETS = ('RGBNT201','RGBNT100','MSVR310')
CONDITIONS = ('global_only','roles')
SCHEMA = 'trifusion-clean-public-clip-joint-v1'
PREDECESSOR = ROOT/'logs/shared_private_evidence_20261001_v2'
SUMMARY = ROOT/'results/shared_private_evidence_complete_20261001/SUMMARY.json'
SUMMARY_SHA = '6537cb1cf8379cffdd1e94bf5569ef23d604940c359e41f0a9b9aaf47d80e9b2'
SOURCES = ('tools/run_clean_clip_joint.py','tools/prepare_clean_clip_joint.py',
           'tools/queue_clean_clip_joint.py','refine-logs/clean_clip_joint_v1/EXPERIMENT_PLAN.md')


def source_map():
    previous = json.loads((PREDECESSOR/'manifest.json').read_text())['source_sha256']
    assert len(previous) == 229
    assert all(sha(ROOT/name) == value for name,value in previous.items())
    return dict(previous, **{name:sha(ROOT/name) for name in SOURCES})


def free_gpus():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                   '--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[0]):int(row.split(',')[1]) for row in raw.splitlines()}


def options(dataset, variant, mode, initialization_dir, output):
    return argparse.Namespace(dataset=dataset,readout=variant,mode=mode,seed=42,epochs=50,
        protocol=(PROTOCOLS/f'{dataset}.json').resolve(),signal_source=SOURCE.resolve(),
        clip_weight=(WEIGHTS/'ViT-B-16.pt').resolve(),
        initialization=(initialization_dir/f'{dataset}_{variant}.json').resolve(),
        output_dir=output.resolve(),visual_update='low_lr',baseline_sha256=sha(WEIGHTS/'ViT-B-16.pt'))


def command(dataset,variant,mode,initialization_dir,output):
    args = options(dataset,variant,mode,initialization_dir,output)
    result = [sys.executable,'-B',str(ROOT/'tools/run_clean_clip_joint.py'),
              '--dataset',dataset,'--mode',mode,'--readout',variant]
    for name in ('protocol','signal_source','clip_weight','initialization','output_dir'):
        result += ['--'+name.replace('_','-'),str(getattr(args,name))]
    return result+['--seed','42','--epochs','50']


def expected_binding(dataset,variant,initialization_dir):
    path = initialization_dir/f'{dataset}_{variant}.json'
    witness = json.loads(path.read_text())
    assert witness['schema'] == SCHEMA and witness['status'] == 'FRESH_PUBLIC_INITIALIZATION_VERIFIED'
    binding = witness['binding']
    assert binding['dataset'] == dataset and binding['readout'] == variant and binding['seed'] == 42
    assert binding['public_clip_sha256'] == sha(WEIGHTS/'ViT-B-16.pt')
    condition = {'visual_update':'low_lr','readout':variant,'visual_lr':5e-6,
                 'visual_parameter_dtype':'float32','initialization':'public_CLIP_fresh_camera_heads_modules',
                 'public_clip_sha256':binding['public_clip_sha256'],
                 'initialization_witness_sha256':sha(path),'camera_update':'new_module_lr_0.00035'}
    return dict(binding,condition=condition,initialization_witness_sha256=sha(path))


def verify_m0(output,dataset,variant,initialization_dir):
    receipt = json.loads((output/'training.json').read_text())
    binding = expected_binding(dataset,variant,initialization_dir)
    assert receipt['schema'] == SCHEMA and receipt['status'] == 'M0_PASS'
    assert receipt['dataset'] == dataset and receipt['seed'] == 42 and receipt['initializer'] == binding
    assert len(receipt['history']) == 1 and receipt['history'][0]['steps'] == 8
    m0 = receipt['m0']
    assert m0['nonzero_gradient_parameters'] == m0['trainable_parameters'] == binding['trainable_parameter_tensors']
    assert receipt['frozen_signal_state_unchanged'] and receipt['visual_parameters_changed']
    assert receipt['fresh_camera_parameters_changed']
    assert receipt['camera_state_before_sha256'] == binding['fresh_camera_initial_sha256']
    assert receipt['camera_state_after_sha256'] != receipt['camera_state_before_sha256']
    assert m0['reload_max_abs_difference'] <= 1e-5
    assert m0['reload_probe_sha256'] == sha(output/'m0_reload_probe.pth')
    return receipt


def verify(output,m0_output,dataset,variant,initialization_dir):
    verify_m0(m0_output,dataset,variant,initialization_dir)
    binding = expected_binding(dataset,variant,initialization_dir)
    training = json.loads((output/'training.json').read_text())
    result = json.loads((output/'official_metrics.json').read_text())
    assert training['schema'] == result['schema'] == SCHEMA
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and result['status'] == 'COMPLETE'
    assert training['dataset'] == result['dataset'] == dataset
    assert training['initializer'] == binding and training['condition'] == result['condition'] == binding['condition']
    assert [row['epoch'] for row in training['history']] == list(range(1,51))
    best = max(training['history'],key=lambda row:(row['official_fused']['mAP'],row['epoch']))
    assert result['selected_epoch'] == training['best_epoch'] == best['epoch']
    assert all(abs(result['metrics'][name]-best['official_fused'][name]) < 1e-5 for name in result['metrics'])
    assert result['training_epochs'] == 50 and result['seed'] == 42
    assert result['protocol_sha256'] == binding['protocol_sha256']
    assert result['baseline_sha256'] == binding['public_clip_sha256']
    assert result['checkpoint_sha256'] == sha(output/'best_map.pth')
    assert result['distance_sha256'] == sha(output/'official_distances.pt')
    assert result['independent_upstream_metrics_equal'] and not result['reranking']
    assert training['visual_parameters_changed'] and training['fresh_camera_parameters_changed']
    return {'dataset':dataset,'variant':variant,'status':'VERIFIED_COMPLETE','run_dir':str(output),
            'm0_dir':str(m0_output),'metrics':result['metrics'],'best_epoch':best['epoch'],
            'checkpoint_sha256':result['checkpoint_sha256'],'distance_sha256':result['distance_sha256'],
            'receipt_sha256':sha(output/'official_metrics.json'),'initializer':binding,
            'training_and_epoch_eval_seconds':(datetime.fromisoformat(training['completed_at'])-
                                               datetime.fromisoformat(training['started_at'])).total_seconds(),
            'timing_boundary':'Training receipt interval including epoch evaluation; excludes model construction and upstream training.',
            'peak_training_and_epoch_eval_allocated_bytes':training['peak_training_and_epoch_eval_allocated_bytes']}


def require_sources(campaign):
    manifest = json.loads((campaign/'manifest.json').read_text())
    assert manifest['source_sha256'] == source_map()
    assert manifest['preflight_sha256'] == sha(Path(manifest['preflight_path']))
    for path,digest in manifest['initialization_sha256'].items():
        assert sha(Path(path)) == digest
    return manifest


def worker(args):
    manifest = require_sources(args.campaign)
    initialization_dir = Path(manifest['preflight_path']).parent
    child = queue.child_campaign(args.campaign,'clean_clip',args.dataset,args.variant)
    assert not child.exists()
    child.mkdir()
    state = {'status':'RUNNING','dataset':args.dataset,'variant':args.variant,'gpu':args.gpu,
             'controller_pid':os.getpid(),'started_at':queue.stamp(),'jobs':[]}
    preflight = json.loads(Path(manifest['preflight_path']).read_text())
    reused = [row for row in preflight['jobs'] if (row['dataset'],row['variant']) == (args.dataset,args.variant)]
    assert len(reused) == (1 if args.dataset == 'RGBNT201' else 0)
    for mode in ('m0','train','evaluate'):
        require_sources(args.campaign)
        assert shutil.disk_usage(ROOT).free >= 10*1024**3
        if mode == 'm0' and reused:
            row = dict(reused[0],origin='verified_preflight')
            verify_m0(Path(row['output_dir']),args.dataset,args.variant,initialization_dir)
            state['jobs'].append(row)
            queue.write(child/'campaign.json',state)
            continue
        output = ROOT/f"trained-model/{child.name}_seed42_{'m0' if mode == 'm0' else 'full'}"
        row = {'mode':mode,'status':'RUNNING','started_at':queue.stamp(),'output_dir':str(output),
               'command':command(args.dataset,args.variant,mode,initialization_dir,output)}
        with (child/f'{mode}.log').open('x') as log:
            process = subprocess.Popen(row['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu)),
                                       stdout=log,stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            state['jobs'].append(row)
            queue.write(child/'campaign.json',state)
            code = process.wait()
        row.update(status='FAILED' if code else 'COMPLETE',exit_code=code,completed_at=queue.stamp())
        state['status'] = 'FAILED' if code else 'RUNNING'
        queue.write(child/'campaign.json',state)
        if code: return code
        if mode == 'm0': verify_m0(output,args.dataset,args.variant,initialization_dir)
    result = verify(output,Path(state['jobs'][0]['output_dir']),args.dataset,args.variant,initialization_dir)
    state.update(status='COMPLETE',completed_at=queue.stamp(),verification=result)
    queue.write(child/'campaign.json',state)
    return 0


def require_complete(child,dataset):
    return json.loads((child/'campaign.json').read_text())['verification']


def start_command(campaign,job,gpu):
    return [sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--campaign',str(campaign),
            '--dataset',job['dataset'],'--variant',job['variant'],'--gpu',str(gpu)]


def coordinate(args):
    assert sha(SUMMARY) == SUMMARY_SHA
    previous = json.loads((PREDECESSOR/'campaign.json').read_text())
    assert previous['status'] == 'COMPLETE' and len(previous['jobs']) == 9
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in previous['jobs'])
    preflight = json.loads(args.preflight.read_text())
    assert preflight['status'] == 'COMPLETE' and preflight['source_sha256'] == source_map()
    expected = [(dataset,variant) for variant in CONDITIONS for dataset in DATASETS]
    initial = preflight['initializations']
    assert len(initial) == 6 and {(row['dataset'],row['variant']) for row in initial} == set(expected)
    assert all(row['common_states_bitwise_equal'] and sha(Path(row['path'])) == row['sha256'] for row in initial)
    assert [(row['dataset'],row['variant']) for row in preflight['jobs']] == [('RGBNT201',variant) for variant in CONDITIONS]
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in preflight['jobs'])
    for row in preflight['jobs']:
        verify_m0(Path(row['output_dir']),row['dataset'],row['variant'],args.preflight.parent)
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    jobs = [{'phase':'clean_clip','dataset':dataset,'variant':variant,'status':'PENDING'} for dataset,variant in expected]
    manifest = {'schema':SCHEMA,'seed':42,'epochs':50,'poll_seconds':queue.POLL_SECONDS,'jobs':jobs,
                'preflight_path':str(args.preflight),'preflight_sha256':sha(args.preflight),
                'initialization_sha256':{row['path']:row['sha256'] for row in initial},
                'source_sha256':source_map(),'predecessor_summary_sha256':SUMMARY_SHA,
                'boundary':'Clean public visual/fresh trained camera and heads; matched independent global/roles; no retry.'}
    queue.write(args.campaign/'manifest.json',manifest)
    state = {'status':'RUNNING','phase':'clean_clip','controller_pid':os.getpid(),'started_at':queue.stamp(),'jobs':jobs}
    queue.start_command = start_command
    queue.require_complete = require_complete
    code = queue.run_phase(args.campaign,state,'clean_clip')
    if code == 0:
        accepted = []
        for job in state['jobs']:
            child = queue.child_campaign(args.campaign,'clean_clip',job['dataset'],job['variant'])
            accepted.append(require_complete(child,job['dataset']))
        assert len(accepted) == 6
        queue.write(args.campaign/'accepted_matrix.json',{'schema':SCHEMA,'accepted':6,'expected':6,'rows':accepted,
                    'boundary':'Full50 worker validation; J1 scientific gate pending sole complete CPU paired report.'})
    state.update(status='FAILED' if code else 'COMPLETE',completed_at=queue.stamp())
    queue.write(args.campaign/'campaign.json',state)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--variant',choices=CONDITIONS)
    parser.add_argument('--gpu',type=int,choices=range(4))
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.worker:
        assert args.dataset and args.variant and args.gpu is not None
        return worker(args)
    assert args.preflight
    args.preflight = args.preflight.resolve()
    return coordinate(args)


if __name__ == '__main__':
    raise SystemExit(main())
