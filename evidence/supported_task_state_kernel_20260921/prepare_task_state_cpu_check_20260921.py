from pathlib import Path
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
out=Path('D:/Program Files/UserCache/gb/codex/tmp/task_state_cpu_check_payload_20260921.py')
kernel=(repo/'tools/msvr_task_state_optimizer.py').read_text()
test=(repo/'tools/check_msvr_task_state_math.py').read_text()
payload='''import os,sys,types,hashlib,json,shutil,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
repo='/root/autodl-tmp/trifusion-v2/TriFusion-ReID'
sys.path.insert(0,repo)
import tools
'''
payload+='kernel='+repr(kernel)+'\ntest='+repr(test)+'\n'
payload+='''module=types.ModuleType('tools.msvr_task_state_optimizer')
sys.modules[module.__name__]=module
exec(compile(kernel,'synthetic_kernel_payload','exec'),module.__dict__)
test_module=types.ModuleType('synthetic_checks')
exec(compile(test,'synthetic_test_payload','exec'),test_module.__dict__)
result=test_module.checks()
result['kernel_sha256']=hashlib.sha256(kernel.encode()).hexdigest()
result['test_sha256']=hashlib.sha256(test.encode()).hexdigest()
result['remote_head']=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
result['free_bytes']={p:shutil.disk_usage(p).free for p in (repo,'/root/trifusion-storage/artifacts')}
print(json.dumps(result,indent=2))
'''
out.write_text(payload,encoding='utf-8')
print(out)
