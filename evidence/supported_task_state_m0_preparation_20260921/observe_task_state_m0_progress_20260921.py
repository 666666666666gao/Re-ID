from pathlib import Path
from datetime import datetime
import json,subprocess,shutil
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_task_state_v1_seed42_cb4f4c3')
pipeline=json.loads((root/'pipeline.json').read_bytes())
pids=[pipeline['wrapper_pid']]+[s['original_pid'] for s in pipeline['stages']]
processes=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,stat,etime,args'],capture_output=True,text=True)
progress=[]
for directory in sorted((root/'m0').iterdir()):
    if not directory.is_dir():continue
    path=directory/'memory_steps.jsonl'
    count=0;last=None
    if path.exists():
        payload=path.read_bytes();lines=payload.splitlines()
        # Observe only complete, newline-terminated writes from the live logger.
        if payload[-1:]!=b'\n':lines=lines[:-1]
        count=len(lines)
        if lines:last=json.loads(lines[-1])
    progress.append(dict(endpoint=directory.name,complete_step_records=count,
        training_report_complete=(directory/'training.json').is_file(),
        optimizer_state_exists=(directory/'optimizer_state.pt').is_file(),
        last_support=None if last is None else last['support']))
stage=pipeline['stages'][-1]['stage']
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),pipeline=pipeline,
    processes=processes.stdout,m0_progress=progress,
    output_free_bytes=shutil.disk_usage(root).free,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
    log_tail=(root/(stage+'.log')).read_text()[-1800:]),indent=2))
