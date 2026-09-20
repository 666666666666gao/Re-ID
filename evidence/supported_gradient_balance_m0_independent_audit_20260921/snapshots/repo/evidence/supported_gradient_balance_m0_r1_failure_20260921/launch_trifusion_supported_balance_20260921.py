from pathlib import Path
import json,hashlib,subprocess,shlex,datetime,shutil

repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
commit='92a75e4ac46cccb79bb6f78f913998c0d0039ce8'
config=repo/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json'
expected='0b5ff0107e8dda6634e0c060a4b1a87e8e7e2fa117c7beb0cf5792b6ebedc15b'
parent=Path('/root/trifusion-storage/artifacts')
output=parent/'msvr310_supported_gradient_balance_v1_seed42_92a75e4'
log=parent/(output.name+'_wrapper.log')
receipt=parent/(output.name+'_launch.json')
session='tri_supported_balance_92a75e4'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==commit
assert hashlib.sha256(config.read_bytes()).hexdigest()==expected
assert not output.exists() and not log.exists() and not receipt.exists()
review=json.loads((repo/'refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict']=='PASS_WITH_LIMITS' and review['config_sha256']==expected
assert review['blocking_findings']==[]
audit=json.loads((repo/'refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_AUDIT_Q1.json').read_bytes())
assert audit['closure_status']=='CLOSED_WITH_LIMITS' and audit['deterministic_checks_status']=='pass'
assert shutil.disk_usage(parent).free>=4294967296
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert len(gpu.strip().splitlines())==1 and int(gpu.split(',')[1])<500
args=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.run_msvr_supported_gradient_balance',
      '--config',str(config),'--config-sha256',expected,'--code-commit',commit,'--output-dir',str(output)]
command='cd '+shlex.quote(str(repo))+' && exec '+shlex.join(args)+' > '+shlex.quote(str(log))+' 2>&1'
subprocess.run(['screen','-dmS',session,'bash','-lc',command],check=True)
data=dict(status='SCREEN_LAUNCH_RETURNED',started_at=datetime.datetime.now().astimezone().isoformat(),
          code_commit=commit,config_sha256=expected,session=session,output=str(output),log=str(log),
          command=args,free_bytes=shutil.disk_usage(parent).free,gpu=gpu,
          scope='Launch receipt only; pipeline process and T0/M0 still require observation')
receipt.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
