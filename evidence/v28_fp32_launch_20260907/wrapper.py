import os,json,subprocess,time,datetime
from pathlib import Path
argv=['/root/miniconda3/envs/tri_reid/bin/python', '-u', 'tools/train_signal_preserving_v28_fp32.py', '--config', '/root/autodl-tmp/trifusion-v2/TriFusion-ReID/configs/RGBNT201/TriFusion-signal-preserving-v28-joint-tokens-fp32-rtx3090.json', '--config-sha256', '43b75c87e2e7795759912b9051fae012b2cc39a23279b5d3d6bc85600fae9029', '--plan', '/root/autodl-tmp/trifusion-v2/TriFusion-ReID/refine-logs/trifusion_v28_joint_tokens_fp32/EXPERIMENT_PLAN.md', '--plan-sha256', '64111a40d44e9a28ff4e9ab62670d608254480fc5c04e44f6841a9613b281e7b', '--output-dir', '/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v28_joint_tokens_fp32_seed42_bf8de95']
repo='/root/autodl-tmp/trifusion-v2/TriFusion-ReID'
run='/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v28_joint_tokens_fp32_seed42_bf8de95'
env=dict(os.environ)
env.update({"PYTHONPATH":repo+"/modeling:"+repo,"CUDA_VISIBLE_DEVICES":"0","PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"4"})
started=time.time()
with Path(run+".log").open("xb") as log:
    child=subprocess.Popen(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    Path(run+"_launch.json").write_text(json.dumps({"original_pid":child.pid,"wrapper_pid":os.getpid(),"argv":argv,"started_at":datetime.datetime.now().astimezone().isoformat(),"repository_commit":'bf8de956e685311dd70631395009a2c06a2c8591'}))
    code=child.wait()
Path(run+"_exit.json").write_text(json.dumps({"exit_code":code,"original_pid":child.pid,"ended_at":datetime.datetime.now().astimezone().isoformat(),"elapsed_seconds":time.time()-started}))
raise SystemExit(code)
