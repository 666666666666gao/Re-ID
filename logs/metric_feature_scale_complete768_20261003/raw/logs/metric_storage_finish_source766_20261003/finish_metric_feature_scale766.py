"""Finish the unstarted original F3 evaluation and first CPU report only."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,str(ROOT))
from tools import queue_metric_feature_scale as panel

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def stamp():
    return datetime.now().astimezone().isoformat()

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=range(4),required=True)
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text())
    assert spec['root']==str(ROOT) and spec['port']==2026
    campaign=Path(spec['campaign']);finish=Path(spec['finish_directory'])
    assert not finish.exists()
    panel.configure()
    panel.base.require_sources(campaign)
    assert panel.source_map()==spec['source_sha256']
    assert sha(campaign/'campaign.json')==spec['original_campaign_sha256']
    state=json.loads((campaign/'campaign.json').read_text())
    assert state['status']=='FAILED' and state['report_invocations']==0
    assert not (Path('/proc')/str(state['controller_pid'])).exists()
    failed=[row for row in state['jobs'] if row['status']=='FAILED']
    assert len(failed)==1 and (failed[0]['phase'],failed[0]['dataset'],failed[0]['variant'],failed[0]['exit_code'])==('full','RGBNT100','metric_raw',1)
    assert len(state['jobs'])==12 and sum(row['status']=='COMPLETE' and row['exit_code']==0 for row in state['jobs'])==11
    assert not (Path('/proc')/str(failed[0]['pid'])).exists()
    child=campaign/f'{campaign.name}_full_metric_raw_RGBNT100'
    assert sha(child/'campaign.json')==spec['original_child_campaign_sha256']
    child_state=json.loads((child/'campaign.json').read_text())
    assert len(child_state['jobs'])==1
    train=child_state['jobs'][0]
    assert train['mode']=='train' and train['status']=='COMPLETE' and train['exit_code']==0
    run=Path(train['output_dir'])
    assert run==ROOT/'trained-model'/f'{campaign.name}_full_metric_raw_RGBNT100'
    assert not (child/'evaluate.log').exists() and not (run/'official_metrics.json').exists() and not (run/'official_distances.pt').exists()
    assert not (campaign/'accepted_matrix.json').exists() and not (campaign/'report.log').exists()
    report=Path(spec['report_directory']);assert not report.exists()
    training=json.loads((run/'training.json').read_text())
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert [row['epoch'] for row in training['history']]==list(range(1,51))
    assert sha(run/'training.json')==spec['training_sha256']
    for row in spec['protected_f3']:
        assert Path(row['path']).stat().st_size==row['bytes'] and sha(Path(row['path']))==row['sha256']
    retirement=json.loads(Path(spec['retirement_receipt']).read_text())
    assert retirement['status']=='RETIRED_EXACT_24_CLOSED_M0' and retirement['retired_bytes']==8444122668
    assert shutil.disk_usage(ROOT).free>=10*1024**3
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    assert dict((int(row.split(',')[0]),int(row.split(',')[1])) for row in memory.splitlines())[args.gpu]<500
    finish.mkdir()
    for name,path in (('original_campaign.json',campaign/'campaign.json'),
                      ('original_child_campaign.json',child/'campaign.json'),
                      ('original_worker_failure.log',campaign/'full_metric_raw_RGBNT100.log')):
        (finish/name).write_bytes(path.read_bytes())
    record={'status':'FIRST_ORIGINAL_EVALUATION_RUNNING','started_at':stamp(),
            'controller_pid':os.getpid(),'gpu':args.gpu,'spec_sha256':sha(args.spec),
            'original_controller_pid':state['controller_pid'],'original_worker_pid':failed[0]['pid'],
            'original_campaign_sha256':spec['original_campaign_sha256'],
            'source_sha256':spec['source_sha256'],'evaluation_invocations':1,'report_invocations':0,
            'boundary':'Storage-only finish: six original full50 trainings retained, missing evaluation called for the first time on the original fixed mAP-best checkpoint, no training/seed/coeff/source/tolerance changes. Original failure archived; original controller not restarted.'}
    command=panel.command('RGBNT100','metric_raw','evaluate',campaign,run)
    assert command.count('evaluate')==1 and 'train' not in command
    row={'mode':'evaluate','status':'RUNNING','started_at':stamp(),'output_dir':str(run),'command':command,
         'completion_provenance':'first invocation after original disk guard stopped before evaluation'}
    with (child/'evaluate.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu)),stdout=log,stderr=subprocess.STDOUT)
        row['pid']=process.pid;record['evaluation_pid']=process.pid
        child_state['jobs'].append(row)
        write(child/'campaign.json',child_state);write(finish/'STATUS.json',record)
        code=process.wait()
    row.update(status='FAILED' if code else 'COMPLETE',exit_code=code,completed_at=stamp())
    write(child/'campaign.json',child_state)
    record.update(evaluation_exit_code=code,status='EVALUATION_FAILED' if code else 'VERIFYING_ALL6')
    write(finish/'STATUS.json',record)
    assert code==0
    result=panel.base.verify(campaign,'RGBNT100','metric_raw')
    child_state.update(status='COMPLETE',verification=result,completed_at=stamp(),
                       original_disk_guard_failure=spec['original_child_campaign_sha256'],
                       finish_only_controller_pid=os.getpid())
    write(child/'campaign.json',child_state)
    original_failure=failed[0].copy()
    failed[0].update(status='COMPLETE',exit_code=0,pid=os.getpid(),completed_at=stamp(),result=result,
                     original_failure=original_failure,
                     completion_provenance='original full50 train exit0 plus first strict evaluation exit0 by recorded storage-only finish controller')
    rows=[panel.base.verify(campaign,dataset,variant) for variant in panel.RECIPES for dataset in panel.DATASETS]
    assert len(rows)==6 and all(job['status']=='COMPLETE' and job['exit_code']==0 for job in state['jobs'])
    write(campaign/'accepted_matrix.json',{'schema':panel.SCHEMA,'accepted':6,'expected':6,'rows':rows})
    state.update(status='COMPLETE',completed_at=stamp(),report_invocations=1,
                 storage_finish_directory=str(finish),storage_finish_controller_pid=os.getpid(),
                 original_failed_campaign_sha256=spec['original_campaign_sha256'])
    write(campaign/'campaign.json',state)
    record.update(status='FIRST_ORIGINAL_CPU_REPORT_RUNNING',report_invocations=1)
    write(finish/'STATUS.json',record)
    with (campaign/'report.log').open('x') as log:
        result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/report_metric_feature_scale.py'),
                               '--campaign',str(campaign),'--output-dir',str(report)],cwd=ROOT,
                              env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode,report_completed_at=stamp())
    write(campaign/'campaign.json',state)
    record.update(status='ALL6_STRICT_AND_FIRST_REPORT_COMPLETE' if result.returncode==0 else 'REPORT_FAILED',
                  report_exit_code=result.returncode,completed_at=stamp())
    write(finish/'STATUS.json',record)
    assert result.returncode==0
    assert panel.source_map()==spec['source_sha256']
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    main()
