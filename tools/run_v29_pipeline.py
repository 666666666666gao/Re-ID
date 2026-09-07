#!/usr/bin/env python3
"""Persist one V29 training process and its complete terminal verification."""
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,os,subprocess,time

PYTHON='/root/miniconda3/envs/tri_reid/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    repo=Path.cwd().resolve();run=args.run_dir.resolve();prefix=str(run)
    assert repo==Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert run.parent==Path('/root/autodl-tmp/trifusion-v2/artifacts')
    assert not run.exists()
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==args.code_commit
    assert sha(args.config)==args.config_sha256 and sha(args.plan)==args.plan_sha256
    config=json.loads(args.config.read_bytes())
    for path,expected in config['SOURCE_FILE_SHA256'].items(): assert sha(path)==expected,path
    started=time.time()
    def write(suffix,row):
        with Path(prefix+suffix).open('x',encoding='utf-8') as handle: json.dump(row,handle,indent=2);handle.write('\n')
    write('_launcher.json',{'wrapper_pid':os.getpid(),'repository_commit':args.code_commit,
                           'started_at':datetime.now().astimezone().isoformat(),'pipeline_sha256':sha(__file__)})
    env=dict(os.environ)
    env.update({'PYTHONPATH':str(repo/'modeling')+':'+str(repo),'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'})
    command=[PYTHON,'-u','tools/train_signal_preserving_v29.py','--config',str(args.config),
             '--config-sha256',args.config_sha256,'--plan',str(args.plan),'--plan-sha256',args.plan_sha256,
             '--output-dir',str(run)]
    with Path(prefix+'.log').open('x') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        write('_launch.json',{'original_pid':process.pid,'wrapper_pid':os.getpid(),'repository_commit':args.code_commit,
                             'config_sha256':args.config_sha256,'plan_sha256':args.plan_sha256,'argv':command,
                             'started_at':datetime.now().astimezone().isoformat()})
        code=process.wait()
    write('_exit.json',{'original_pid':process.pid,'wrapper_pid':os.getpid(),'exit_code':code,
                       'ended_at':datetime.now().astimezone().isoformat(),'elapsed_seconds':time.time()-started})
    if code:
        write('_pipeline_exit.json',{'stage':'TRAIN_PROCESS_FAILED','exit_code':code,'wrapper_pid':os.getpid()})
        return code
    summary=json.loads((run/'run_summary.json').read_bytes())
    env['CUDA_VISIBLE_DEVICES']=''
    if summary['status']=='M0_FAIL':
        commands=[('m0_verification',[PYTHON,'tools/verify_v29_m0.py','--summary',str(run/'run_summary.json'),
                    '--repo',str(repo),'--output',str(run/'m0_complete_verification.json')])]
    else:
        assert summary['status'] in ('Q1_PASS','Q1_FAIL') and len(summary['folds'])==3
        commands=[('terminal_verification',[PYTHON,'tools/verify_v29_complete_terminal.py','--repo',str(repo),
                   '--run-dir',str(run),'--output',str(run/'complete_terminal_verification.json')]),
                  ('terminal_report',[PYTHON,'tools/report_v29_complete_comparison.py','--summary',str(run/'run_summary.json'),
                   '--training-dir',str(run),'--verification',str(run/'complete_terminal_verification.json'),
                   '--output-json',str(run/'complete_comparison.json'),'--output-md',str(run/'complete_comparison.md'),
                   '--query-csv',str(run/'all_query_comparison.csv'),'--identity-csv',str(run/'all_identity_comparison.csv')])]
    for stage,command in commands:
        stage_start=time.time()
        with Path(prefix+'_'+stage+'.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write('_'+stage+'_launch.json',{'original_pid':child.pid,'wrapper_pid':os.getpid(),'command':command,
                                            'started_at':datetime.now().astimezone().isoformat()})
            code=child.wait()
        write('_'+stage+'_exit.json',{'original_pid':child.pid,'exit_code':code,
                                      'ended_at':datetime.now().astimezone().isoformat(),'elapsed_seconds':time.time()-stage_start})
        if code:
            write('_pipeline_exit.json',{'stage':stage.upper()+'_FAILED','exit_code':code,'wrapper_pid':os.getpid()})
            return code
    write('_pipeline_exit.json',{'stage':'COMPLETE_VERIFIED_'+summary['status'],'exit_code':0,'wrapper_pid':os.getpid(),
                                'ended_at':datetime.now().astimezone().isoformat(),'elapsed_seconds':time.time()-started})
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--code-commit',required=True)
    raise SystemExit(main(parser.parse_args()))
