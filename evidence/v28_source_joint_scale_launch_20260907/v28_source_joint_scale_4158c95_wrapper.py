from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys,time
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run=Path('/root/autodl-tmp/trifusion-v2/artifacts/v28_source_joint_scale_seed42_4158c95')
python='/root/miniconda3/envs/tri_reid/bin/python'
head='4158c95ca639721e584d1e7271219f0f3b557cc3'
contract='configs/diagnostics/V28-source-joint-scale-v1.json'
contract_sha='f32293b64d9343ab308b3e9cd95cd41a6722630a34026ff42fafcf0ef2cd398e'
def write(suffix,data):
    Path(str(run)+suffix).write_bytes((json.dumps(data,indent=2)+chr(10)).encode())
env=dict(os.environ,PYTHONPATH=str(repo/'modeling')+':'+str(repo),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
started=time.time()
gpu_command=[python,'-u','tools/diagnose_v28_source_joint_scale.py','--contract',contract,'--contract-sha256',contract_sha,'--output-dir',str(run)]
gpu=subprocess.Popen(gpu_command,cwd=repo,env=env)
write('_launch.json',{'stage':'GPU_SOURCE_DIAGNOSTIC','original_pid':gpu.pid,'wrapper_pid':os.getpid(),'started_at':datetime.now().astimezone().isoformat(),'execution_commit':head,'contract_sha256':contract_sha,'command':gpu_command})
gpu_code=gpu.wait()
write('_gpu_exit.json',{'original_pid':gpu.pid,'exit_code':gpu_code,'ended_at':datetime.now().astimezone().isoformat(),'elapsed_seconds':time.time()-started})
if gpu_code!=0:
    write('_exit.json',{'stage':'GPU_DIAGNOSTIC_FAILED','exit_code':gpu_code,'wrapper_pid':os.getpid(),'ended_at':datetime.now().astimezone().isoformat()})
    sys.exit(gpu_code)
cpu_started=time.time()
cpu_env=dict(env,CUDA_VISIBLE_DEVICES='')
cpu_command=[python,'-u','tools/verify_v28_source_joint_scale.py','--contract',contract,'--contract-sha256',contract_sha,'--run-dir',str(run)]
cpu=subprocess.Popen(cpu_command,cwd=repo,env=cpu_env)
write('_cpu_launch.json',{'stage':'FULL_CPU_GEOMETRY_VERIFICATION','original_pid':cpu.pid,'wrapper_pid':os.getpid(),'started_at':datetime.now().astimezone().isoformat(),'command':cpu_command})
cpu_code=cpu.wait()
write('_cpu_exit.json',{'original_pid':cpu.pid,'exit_code':cpu_code,'ended_at':datetime.now().astimezone().isoformat(),'elapsed_seconds':time.time()-cpu_started})
write('_exit.json',{'stage':'COMPLETE_GPU_AND_CPU_VERIFICATION' if cpu_code==0 else 'CPU_VERIFICATION_FAILED','exit_code':cpu_code,'wrapper_pid':os.getpid(),'ended_at':datetime.now().astimezone().isoformat(),'elapsed_seconds':time.time()-started})
sys.exit(cpu_code)
