"""Run the registered three F1-initialized full50 jobs, one per physical GPU."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.official_three_dataset_model import sha256


def main():
    plan_path=Path(sys.argv[1]).resolve()
    plan=json.loads(plan_path.read_bytes())
    assert plan['schema']=='trifusion-f1-initialized-three-dataset-v1'
    assert ROOT==Path('/data/gaob/Re-ID/Trifusion')
    campaign=ROOT/plan['campaign'];train_root=ROOT/plan['training_root']
    campaign.mkdir();train_root.mkdir()
    lock=Lock()
    status=dict(schema=plan['schema'],status='RUNNING',plan_sha256=sha256(plan_path),
                started_at=datetime.now().astimezone().isoformat(),jobs=[])
    for worker in plan['workers']:
        assert worker['gpu'] in (0,1)
        for dataset in worker['datasets']:
            status['jobs'].append(dict(dataset=dataset,gpu=worker['gpu'],status='PENDING'))
    assert sorted(row['dataset'] for row in status['jobs'])==['MSVR310','RGBNT100','RGBNT201']

    def save():
        (campaign/'campaign.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')

    def update(row,value,**fields):
        with lock:
            row.update(status=value,**fields);save()

    def verify_sources():
        assert all(sha256(ROOT/name)==digest for name,digest in plan['source_sha256'].items())

    def run(row,mode,directory):
        verify_sources()
        command=[sys.executable,'-B',str(ROOT/'tools/run_f1_initialized_roles.py'),
                 '--dataset',row['dataset'],'--mode',mode,
                 '--protocol',str(ROOT/'logs/official_three_dataset_protocols_20260923'/f"{row['dataset']}.json"),
                 '--signal-source',str(ROOT/'comparators/Signal-cd1b0a6'),
                 '--clip-weight',str(ROOT/'pertrained-model/ViT-B-16.pt'),
                 '--foundation-manifest',str(ROOT/plan['foundation_manifest']),
                 '--signal-references',str(ROOT/plan['signal_references']),
                 '--output-dir',str(directory),'--seed','42']
        start=time.perf_counter()
        update(row,mode.upper(),phase_started_at=datetime.now().astimezone().isoformat(),command=command)
        with (campaign/f"{row['dataset']}.{mode}.log").open('x',encoding='utf-8') as log:
            process=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(row['gpu']),
                                   PYTHONNOUSERSITE='1'),stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            update(row,'FAILED',failed_phase=mode,returncode=process.returncode)
            raise RuntimeError(f"{row['dataset']} {mode} failed: {process.returncode}")
        with lock:
            row.setdefault('phase_seconds',{})[mode]=time.perf_counter()-start;save()

    def worker(spec):
        # No legacy progress reads before the already registered observation time.
        not_before=datetime.fromisoformat(spec['not_before']).timestamp()
        time.sleep(max(0,not_before-time.time()))
        prior=ROOT/spec['after_campaign']
        while json.loads(prior.read_bytes())['status']!='COMPLETE':time.sleep(240)
        for dataset in spec['datasets']:
            row=next(row for row in status['jobs'] if row['dataset']==dataset)
            free=shutil.disk_usage(ROOT).free
            while free<=3*1024**3:
                update(row,'WAITING_FOR_DISK',free_disk_bytes=free)
                time.sleep(240)
                free=shutil.disk_usage(ROOT).free
            memory=subprocess.check_output(['nvidia-smi','-i',str(spec['gpu']),
                '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            assert int(memory.strip())<500
            job=campaign/dataset;job.mkdir()
            run(row,'prepare',job/'prepare')
            assert json.loads((job/'prepare/initialization.json').read_bytes())['status']=='F1_INITIALIZATION_AND_FORWARD_VERIFIED'
            run(row,'m0',job/'m0')
            assert json.loads((job/'m0/training.json').read_bytes())['status']=='M0_PASS'
            directory=train_root/dataset
            run(row,'train',directory)
            training=json.loads((directory/'training.json').read_bytes())
            assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and training['training']['epochs']==50
            run(row,'evaluate',directory)
            result=json.loads((directory/'official_metrics.json').read_bytes())
            assert result['status']=='COMPLETE' and result['baseline_f1_metrics_equal']
            delta=result['fused_vs_full_signal']['metric_delta_pp']
            update(row,'COMPLETE',metrics_path=str(directory/'official_metrics.json'),
                   metrics=result['outputs']['fused']['metrics'],signal_delta_pp=delta,
                   all_four_above_full_signal=all(value>0 for value in delta.values()),
                   completed_at=datetime.now().astimezone().isoformat())

    verify_sources();save()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,spec) for spec in plan['workers']]
        for future in futures:future.result()
    status.update(status='COMPLETE',completed_at=datetime.now().astimezone().isoformat());save()


if __name__=='__main__':main()
