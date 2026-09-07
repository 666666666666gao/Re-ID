import os,json,subprocess,time,datetime
from pathlib import Path
argv=['/root/miniconda3/envs/tri_reid/bin/python', '-u', '/root/autodl-tmp/trifusion-v2/transport/v28_fp32_terminal_waiter_bf8de95.py']
repo='/root/autodl-tmp/trifusion-v2/TriFusion-ReID'
stem='/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v28_joint_tokens_fp32_seed42_bf8de95_verification_waiter'
started=time.time()
env=dict(os.environ)
env.update({"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"4"})
with Path(stem+".log").open("xb") as log:
    child=subprocess.Popen(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    Path(stem+"_launch.json").write_text(json.dumps({"original_pid":child.pid,"wrapper_pid":os.getpid(),"argv":argv,"started_at":datetime.datetime.now().astimezone().isoformat(),"waiter_sha256":'9648270beb382a7d68cf3f94b858eae1043ebd49db4cd9292af87142a45964ce',"poll_seconds":240,"training_execution_commit":"bf8de956e685311dd70631395009a2c06a2c8591"}))
    code=child.wait()
Path(stem+"_exit.json").write_text(json.dumps({"exit_code":code,"original_pid":child.pid,"ended_at":datetime.datetime.now().astimezone().isoformat(),"elapsed_seconds":time.time()-started}))
raise SystemExit(code)
