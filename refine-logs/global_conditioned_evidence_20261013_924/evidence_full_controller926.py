"""Two fresh RGBNT201 full50/first strict endpoints; no retry or old queue."""
import hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime
from pathlib import Path

ROOT=Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN=Path(__file__).resolve().parent
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=lambda:datetime.now().astimezone().isoformat()


def verify(pins):
    assert all(sha(ROOT/n)==h for n,h in pins['runtime_source_sha256'].items())
    assert all(sha(ROOT/n)==row['sha256'] for n,row in pins['m0_file_sha256'].items())
    assert sha(ROOT/'pertrained-model/ViT-B-16.pt')==pins['public_clip_sha256']
    assert sha(ROOT/'logs/training_feature_scale_protocols_20261002/RGBNT201.json')==pins['protocol_sha256']


def devices():
    values=[tuple(int(v.strip()) for v in line.split(',')) for line in subprocess.check_output(
        ['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert {row[0] for row in values}=={0,1} and all(row[1]<500 for row in values),values
    return values


def worker():
    pins=json.loads((CAMPAIGN/'INPUTS.json').read_text())
    verify(pins)
    m0=json.loads((ROOT/'logs/global_conditioned_evidence_m0_20261013_925/campaign.json').read_text())
    assert m0['status']=='BOTH_PREPARE_M0_ACCEPTED_NO_FORMAL_TRAINING'
    assert not Path('/proc/'+str(m0['supervisor_pid'])).exists()
    assert all(not Path('/proc/'+str(row['child_pid'])).exists() for row in m0['stages'])
    outputs={kind:ROOT/f'trained-model/global_conditioned_evidence_20261013_926_full_{kind}_RGBNT201'
             for kind in ('gradient','random')}
    assert all(not p.exists() for p in outputs.values())
    state={'status':'RUNNING','started_at':stamp(),'worker_pid':os.getpid(),'physical_gpus':[0,1],
        'seed':42,'epochs_per_endpoint':50,'stages':[],'accepted_endpoints':[],'goal_status':'ACTIVE_UNMET'}
    path=CAMPAIGN/'campaign.json'
    write=lambda:path.write_text(json.dumps(state,indent=2)+'\n')
    started=time.perf_counter()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for index,kind in enumerate(('gradient','random')):
        verify(pins)
        admission=devices()
        required=((2-index)*384+384+128+512)*1024**2
        free=shutil.disk_usage(ROOT).free
        assert free>=required,(free,required)
        state['stages'].append({'event':'admission','exposure':kind,'at':stamp(),
            'devices':admission,'disk_free_bytes':free,'required_bytes':required})
        write()
        init=ROOT/f'logs/global_conditioned_evidence_m0_20261013_925/initialization/RGBNT201_{kind}.json'
        for mode in ('train','evaluate'):
            verify(pins)
            if mode=='evaluate':
                assert shutil.disk_usage(ROOT).free >= (128+512)*1024**2
            command=[PYTHON,'-B',str(ROOT/'tools/run_global_conditioned_evidence.py'),
                '--dataset','RGBNT201','--exposure',kind,'--mode',mode,
                '--protocol',str(ROOT/'logs/training_feature_scale_protocols_20261002/RGBNT201.json'),
                '--signal-source',str(ROOT/'comparators/Signal-cd1b0a6'),
                '--clip-weight',str(ROOT/'pertrained-model/ViT-B-16.pt'),
                '--initialization',str(init),'--output-dir',str(outputs[kind])]
            with (CAMPAIGN/f'{kind}_{mode}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                stage={'exposure':kind,'mode':mode,'started_at':stamp(),'child_pid':child.pid,'command':command}
                state['stages'].append(stage)
                write()
                exit_code=child.wait()
            stage.update(exit_code=exit_code,completed_at=stamp())
            (CAMPAIGN/f'{kind}_{mode}_EXIT.json').write_text(json.dumps(stage,indent=2)+'\n')
            write()
            if exit_code!=0:
                state.update(status='ORIGINAL_STAGE_FAILED',completed_at=stamp(),elapsed_seconds=time.perf_counter()-started)
                write()
                return exit_code
            training=json.loads((outputs[kind]/'training.json').read_text())
            binding=json.loads(init.read_text())['binding']
            assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and training['initializer']==binding
            assert [row['epoch'] for row in training['history']]==list(range(1,51))
            assert training['frozen_parameters_unchanged'] and training['visual_parameters_changed'] and training['fresh_camera_parameters_changed']
            if mode=='evaluate':
                result=json.loads((outputs[kind]/'official_metrics.json').read_text())
                selected=max(training['history'],key=lambda row:(row['official_fused']['mAP'],row['epoch']))
                assert result['status']=='COMPLETE' and result['training_epochs']==50
                assert result['selected_epoch']==training['best_epoch']==selected['epoch']
                assert all(abs(result['metrics'][k]-selected['official_fused'][k])<1e-5 for k in result['metrics'])
                assert result['checkpoint_sha256']==sha(outputs[kind]/'best_map.pth')
                assert result['distance_sha256']==sha(outputs[kind]/'official_distances.pt')
                state['accepted_endpoints'].append({'exposure':kind,'output_dir':str(outputs[kind]),
                    'metrics':result['metrics'],'selected_epoch':result['selected_epoch'],
                    'official_metrics_sha256':sha(outputs[kind]/'official_metrics.json')})
                write()
    verify(pins)
    state.update(status='COMPLETE_TWO_FULL50_FIRST_STRICT',completed_at=stamp(),elapsed_seconds=time.perf_counter()-started)
    write()
    return 0


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--worker':
        raise SystemExit(worker())
    assert len(sys.argv)==1
    with (CAMPAIGN/'WORKER.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/python3',str(Path(__file__).resolve()),'--worker'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        original={'supervisor_pid':os.getpid(),'worker_pid':child.pid,'started_at':stamp(),
            'command':['/usr/bin/python3',str(Path(__file__).resolve()),'--worker']}
        (CAMPAIGN/'SUPERVISOR.json').write_text(json.dumps(original,indent=2)+'\n')
        code=child.wait()
    original.update(exit_code=code,completed_at=stamp())
    (CAMPAIGN/'CONTROLLER_EXIT.json').write_text(json.dumps(original,indent=2)+'\n')
    raise SystemExit(code)
