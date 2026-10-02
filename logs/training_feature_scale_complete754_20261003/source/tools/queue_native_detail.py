"""Six N1 endpoints using the existing no-retry, 240-second GPU scheduler."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_clean_clip_joint as panel

DATASETS = panel.DATASETS
CONDITIONS = ('high','low')
SCHEMA = 'trifusion-native-detail-v1'
PREDECESSOR = ROOT/'logs/clean_clip_joint_20261002_v1'
SUMMARY = ROOT/'results/clean_clip_joint_complete_20261002/SUMMARY.json'
SUMMARY_SHA = '784e3938a0203ea043543804c6e2933b43877405d7ca624116d59e59d35538fd'
SOURCES = ('modeling/trifusion/native_detail_roles.py', 'tools/run_native_detail.py',
           'tools/prepare_native_detail.py', 'tools/queue_native_detail.py',
           'refine-logs/native_detail_v1/EXPERIMENT_PLAN.md')
sha, queue, free_gpus = panel.sha, panel.queue, panel.free_gpus


def source_map():
    previous = json.loads((PREDECESSOR/'manifest.json').read_text())['source_sha256']
    assert len(previous) == 233
    assert all(sha(ROOT/name) == value for name,value in previous.items())
    return dict(previous, **{name:sha(ROOT/name) for name in SOURCES})


def options(dataset, variant, mode, initialization_dir, output):
    args = panel.options(dataset,'roles',mode,initialization_dir,output)
    args.variant = variant
    args.initialization = (initialization_dir/f'{dataset}_{variant}.json').resolve()
    return args


def command(dataset,variant,mode,initialization_dir,output):
    args = options(dataset,variant,mode,initialization_dir,output)
    result = [sys.executable,'-B',str(ROOT/'tools/run_native_detail.py'),
              '--dataset',dataset,'--mode',mode,'--variant',variant]
    for name in ('protocol','signal_source','clip_weight','initialization','output_dir'):
        result += ['--'+name.replace('_','-'),str(getattr(args,name))]
    return result+['--seed','42','--epochs','50']


def expected_binding(dataset,variant,initialization_dir):
    path = initialization_dir/f'{dataset}_{variant}.json'
    witness = json.loads(path.read_text())
    assert witness['schema'] == SCHEMA and witness['status'] == 'FRESH_PUBLIC_INITIALIZATION_VERIFIED'
    binding = witness['binding']
    assert binding['dataset'] == dataset and binding['detail_resolution'] == variant and binding['seed'] == 42
    assert binding['public_clip_sha256'] == sha(panel.WEIGHTS/'ViT-B-16.pt')
    from tools.run_native_detail import condition
    args = options(dataset,variant,'prepare',initialization_dir,ROOT/'trained-model/native_prepare_unused')
    return dict(binding,condition=condition(args),initialization_witness_sha256=sha(path))


def install():
    # Reuse the validated worker and verification against this registered contract.
    panel.SCHEMA = SCHEMA
    panel.CONDITIONS = CONDITIONS
    panel.source_map = source_map
    panel.command = command
    panel.expected_binding = expected_binding


def start_command(campaign,job,gpu):
    return [sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--campaign',str(campaign),
            '--dataset',job['dataset'],'--variant',job['variant'],'--gpu',str(gpu)]


def coordinate(args):
    assert sha(SUMMARY) == SUMMARY_SHA
    previous = json.loads((PREDECESSOR/'campaign.json').read_text())
    assert previous['status'] == 'COMPLETE' and len(previous['jobs']) == 6
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in previous['jobs'])
    preflight = json.loads(args.preflight.read_text())
    assert preflight['status'] == 'COMPLETE' and preflight['source_sha256'] == source_map()
    expected = [(dataset,variant) for variant in CONDITIONS for dataset in DATASETS]
    initial = preflight['initializations']
    assert len(initial) == 6 and {(row['dataset'],row['variant']) for row in initial} == set(expected)
    assert all(row['all_high_low_states_bitwise_equal'] and row['original_roles_state_equal_except_stem']
               and sha(Path(row['path'])) == row['sha256'] for row in initial)
    assert [(row['dataset'],row['variant']) for row in preflight['jobs']] == [('RGBNT201',v) for v in CONDITIONS]
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in preflight['jobs'])
    for row in preflight['jobs']:
        panel.verify_m0(Path(row['output_dir']),row['dataset'],row['variant'],args.preflight.parent)
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    jobs = [{'phase':'clean_clip','dataset':d,'variant':v,'status':'PENDING'} for d,v in expected]
    queue.write(args.campaign/'manifest.json',
        {'schema':SCHEMA,'seed':42,'epochs':50,'poll_seconds':240,'jobs':jobs,
         'preflight_path':str(args.preflight),'preflight_sha256':sha(args.preflight),
         'initialization_sha256':{row['path']:row['sha256'] for row in initial},
         'source_sha256':source_map(),'predecessor_summary_sha256':SUMMARY_SHA,
         'boundary':'N1 image-native values and same-parameter low-resolution control; six full50, no retries.'})
    state = {'status':'RUNNING','phase':'clean_clip','controller_pid':os.getpid(),
             'started_at':queue.stamp(),'jobs':jobs}
    queue.start_command, queue.require_complete = start_command, panel.require_complete
    code = queue.run_phase(args.campaign,state,'clean_clip')
    if code == 0:
        accepted = [panel.require_complete(queue.child_campaign(args.campaign,'clean_clip',job['dataset'],job['variant']),
                                           job['dataset']) for job in state['jobs']]
        queue.write(args.campaign/'accepted_matrix.json',
                    {'schema':SCHEMA,'accepted':6,'expected':6,'rows':accepted,
                     'boundary':'Full50/strict reload complete; scientific N1-A/B pending complete paired report.'})
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
    install()
    if args.worker:
        assert args.dataset and args.variant and args.gpu is not None
        return panel.worker(args)
    assert args.preflight
    args.preflight = args.preflight.resolve()
    return coordinate(args)


if __name__ == '__main__':
    raise SystemExit(main())
