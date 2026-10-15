"""Continue one F1 vehicle endpoint whose original prepare stopped before training."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.official_three_dataset_model import sha256


def main():
    plan_path=Path(sys.argv[1]).resolve()
    plan=json.loads(plan_path.read_bytes())
    assert plan['schema']=='trifusion-f1-vehicle-prepare-continuation-v1'
    dataset=plan['dataset'];gpu=plan['gpu']
    assert dataset in ('RGBNT100','MSVR310') and gpu in (0,1)
    assert ROOT==Path('/data/gaob/Re-ID/Trifusion')

    def verify_sources():
        assert all(sha256(ROOT/name)==digest for name,digest in plan['source_sha256'].items())

    verify_sources()
    original=json.loads((ROOT/plan['original_campaign']).read_bytes())
    failed=next(row for row in original['jobs'] if row['dataset']==dataset)
    assert failed['status']=='FAILED' and failed['failed_phase']=='prepare' and failed['returncode']==1
    assert sha256(ROOT/plan['original_prepare_log'])==plan['original_prepare_log_sha256']
    assert (ROOT/plan['original_prepare_log']).read_text().rstrip().endswith("KeyError: 'SIGNAL'")
    assert not (ROOT/plan['original_training_directory']).exists()
    assert shutil.disk_usage(ROOT).free>3*1024**3
    memory=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert int(memory.strip())<500
    campaign=ROOT/plan['campaign'];campaign.mkdir()
    status=dict(schema=plan['schema'],status='RUNNING',dataset=dataset,gpu=gpu,seed=42,
                plan_sha256=sha256(plan_path),started_at=datetime.now().astimezone().isoformat(),
                original_failure=failed,phase_seconds={})

    def save():
        (campaign/'campaign.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')

    for mode in ('prepare','m0','train','evaluate'):
        verify_sources()
        directory=ROOT/plan['training_directory'] if mode in ('train','evaluate') else campaign/mode
        command=[sys.executable,'-B',str(ROOT/'tools/run_f1_vehicle_initialized_roles.py'),
                 '--dataset',dataset,'--mode',mode,
                 '--protocol',str(ROOT/'logs/official_three_dataset_protocols_20260923'/f'{dataset}.json'),
                 '--signal-source',str(ROOT/'comparators/Signal-cd1b0a6'),
                 '--clip-weight',str(ROOT/'pertrained-model/ViT-B-16.pt'),
                 '--foundation-manifest',str(ROOT/plan['foundation_manifest']),
                 '--signal-references',str(ROOT/plan['signal_references']),
                 '--output-dir',str(directory),'--seed','42']
        status.update(phase=mode,phase_started_at=datetime.now().astimezone().isoformat(),command=command)
        save();start=time.perf_counter()
        with (campaign/f'{mode}.log').open('x',encoding='utf-8') as log:
            process=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                                   PYTHONNOUSERSITE='1'),stdout=log,stderr=subprocess.STDOUT)
        status['phase_seconds'][mode]=time.perf_counter()-start
        if process.returncode:
            status.update(status='FAILED',returncode=process.returncode);save()
            raise RuntimeError(f'{dataset} {mode} failed: {process.returncode}')
        if mode=='prepare':
            assert json.loads((directory/'initialization.json').read_bytes())['status']=='F1_INITIALIZATION_AND_FORWARD_VERIFIED'
        if mode=='m0':
            assert json.loads((directory/'training.json').read_bytes())['status']=='M0_PASS'
        if mode=='train':
            t=json.loads((directory/'training.json').read_bytes())
            assert t['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and t['training']['epochs']==50
        save()
    result=json.loads((directory/'official_metrics.json').read_bytes())
    assert result['status']=='COMPLETE' and result['baseline_f1_metrics_equal']
    delta=result['fused_vs_full_signal']['metric_delta_pp']
    status.update(status='COMPLETE',completed_at=datetime.now().astimezone().isoformat(),
                  metrics_path=str(directory/'official_metrics.json'),metrics=result['outputs']['fused']['metrics'],
                  signal_delta_pp=delta,all_four_above_full_signal=all(value>0 for value in delta.values()))
    save()


if __name__=='__main__':main()
