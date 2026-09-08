"""Persistent source-only terminal intake/text analysis; never restarts GPU jobs."""
import argparse
from datetime import datetime
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    plan=json.loads(args.plan.read_bytes());root=args.output_dir
    assert not root.exists();root.mkdir();(root/'observations').mkdir()
    state=dict(status='STARTING',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
               plan_sha256=sha(args.plan),stages=[],poll_count=0,python=sys.executable)
    def save():
        temporary=root/'state.next.json'
        temporary.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        temporary.replace(root/'state.json')
    def check_sources():
        assert all(sha(path)==digest for path,digest in plan['script_sha256'].items())
    def execute(name,command,repeat=False):
        check_sources();started=time.monotonic()
        row=dict(name=name,started_at=datetime.now().astimezone().isoformat(),command=command)
        state['stages'].append(row);save()
        with (root/(name+'.stdout.log')).open('a' if repeat else 'x',encoding='utf-8') as out, (root/(name+'.stderr.log')).open('a' if repeat else 'x',encoding='utf-8') as err:
            child=subprocess.Popen(command,cwd=plan['repository'],stdout=out,stderr=err,stdin=subprocess.DEVNULL)
            row['pid']=child.pid;save();code=child.wait()
        row.update(exit_code=code,elapsed_seconds=time.monotonic()-started,ended_at=datetime.now().astimezone().isoformat());save()
        return code
    check_sources();state['status']='WAITING_FOR_FIRST_SCHEDULED_OBSERVATION';save()
    time.sleep(plan['initial_delay_seconds'])
    while True:
        state['poll_count']+=1;state['status']='OBSERVING';save()
        code=execute('observe',[sys.executable,'-X','utf8',plan['observer'],'--output-dir',str(root/'observations')],repeat=True)
        if code:
            state['status']='OBSERVATION_FAILED_TRAINING_STATUS_UNKNOWN';save();return code
        observation=json.loads((root/'observations/history_gradient_training_latest_observation_20260908.json').read_bytes())
        pipeline=observation['pipeline'];state['latest_observed_at']=observation['checked_at']
        state['remote_status']=pipeline['status'];state['remote_stage']=pipeline['stages'][-1]['stage']
        assert pipeline['code_commit']==plan['execution_commit'] and pipeline['config_sha256']==plan['config_sha256']
        if pipeline['status']=='RUNNING':
            wrapper=observation['processes'][0]
            if not wrapper['present'] or wrapper['command'].strip()!=plan['wrapper_command']:
                state['status']='RUNNING_WRAPPER_NOT_CONFIRMED_REQUIRES_RECHECK';save();return 2
            state['status']='WAITING_VERIFIED_REMOTE_WRAPPER';save();time.sleep(plan['interval_seconds']);continue
        if pipeline['status'] not in ('COMPLETE_VERIFIED_Q1_PASS','COMPLETE_VERIFIED_Q1_FAIL'):
            state['status']='REMOTE_PIPELINE_STOPPED_NO_RESTART';save();return 3
        assert all(stage['exit_code']==0 for stage in pipeline['stages']) and len(pipeline['stages'])==5
        break
    state['status']='RECEIVING_COMPLETE_TERMINAL_TEXT';save()
    commands=[('intake',[sys.executable,'-X','utf8',plan['collector']]),
              ('training_text',[sys.executable,'-X','utf8',plan['training_analyzer'],'--input-dir',plan['intake_directory'],'--mode','q1','--output',str(root/'training_text_analysis.json')])]
    for name,command in commands:
        code=execute(name,command)
        if code:
            state['status']='POSTPROCESS_FAILED_'+name.upper();save();return code
    summary=Path(plan['intake_directory'])/'q1/summary.json'
    command=[sys.executable,'-X','utf8',plan['ranking_analyzer'],'--input-dir',plan['intake_directory'],
             '--candidate','history_gradient','--summary-sha256',sha(summary),
             '--cpu-status','PASS_COMPLETE_HISTORY_GRADIENT_Q1','--output-dir',str(root/'rankings')]
    code=execute('ranking_text',command)
    if code:
        state['status']='POSTPROCESS_FAILED_RANKING_TEXT';save();return code
    analysis=json.loads((root/'training_text_analysis.json').read_bytes())
    rankings=json.loads((root/'rankings/ranking_replay.json').read_bytes())
    assert analysis['status']=='PASS_COMPLETE_HISTORY_GRADIENT_TRAINING_TEXT' and analysis['total_updates']==1560
    assert rankings['status']=='PASS_COMPLETE_TERMINAL_RANKING_TEXT' and rankings['query_count']==600
    state.update(status='COMPLETE_LOCAL_TEXT_AND_RANKING_VERIFIED_AWAITING_INDEPENDENT_AUDIT',
                 completed_at=datetime.now().astimezone().isoformat(),scientific_status=rankings['scientific_status'],
                 terminal_summary_sha256=sha(summary),training_analysis_sha256=sha(root/'training_text_analysis.json'),
                 ranking_analysis_sha256=sha(root/'rankings/ranking_replay.json'))
    save();return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
