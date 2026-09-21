from pathlib import Path
from datetime import datetime
import json,subprocess,shutil
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_task_state_v1_seed42_cb4f4c3')
pipeline=json.loads((root/'pipeline.json').read_bytes())
pids=[pipeline['wrapper_pid']]+[s['original_pid'] for s in pipeline['stages']]
processes=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,stat,etime,args'],capture_output=True,text=True)
epochs=[]
for line in (root/'q1.log').read_text().splitlines():
    if line.startswith('{"event": "supported_task_state_epoch"'):
        epochs.append(json.loads(line))
endpoints=[]
for directory in sorted((root/'q1').iterdir()):
    if not directory.is_dir():continue
    path=directory/'memory_steps.jsonl'
    count=0
    if path.exists():
        payload=path.read_bytes();lines=payload.splitlines()
        if payload[-1:]!=b'\n':lines=lines[:-1]
        count=len(lines)
    endpoints.append(dict(endpoint=directory.name,complete_step_records=count,
        training_complete=(directory/'training.json').is_file(),receipt_complete=(directory/'receipt.json').is_file()))
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),pipeline=pipeline,
    processes=processes.stdout,q1_endpoints=endpoints,epoch_records=epochs,
    output_free_bytes=shutil.disk_usage(root).free,
    project_free_bytes=shutil.disk_usage('/root/autodl-tmp/trifusion-v2/TriFusion-ReID').free,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)),indent=2))
