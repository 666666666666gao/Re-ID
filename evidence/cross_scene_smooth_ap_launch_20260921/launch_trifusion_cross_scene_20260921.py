from pathlib import Path
import json,hashlib,subprocess,shlex,datetime,shutil

repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
commit='d35864d6411591e05c8ac3e5164ebae48063ad99'
config=repo/'configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json'
expected='e5326b52ebb12dced24ebfac788db2e2bb5bdca0b50c6c4dec64e596f1af05a1'
parent=Path('/root/trifusion-storage/artifacts')
output=parent/'msvr310_cross_scene_smooth_ap_v1_seed42_d35864d'
log=parent/'msvr310_cross_scene_smooth_ap_v1_seed42_d35864d_wrapper.log'
receipt=parent/'msvr310_cross_scene_smooth_ap_v1_seed42_d35864d_launch.json'
session='tri_cross_scene_d35864d'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==commit
assert hashlib.sha256(config.read_bytes()).hexdigest()==expected
assert not output.exists() and not log.exists() and not receipt.exists()
review=json.loads((repo/'refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict']=='PASS_WITH_LIMITS' and review['reviewed_config_sha256']==expected
assert review['blocking_findings']==[]
audit=json.loads((repo/'refine-logs/msvr310_smooth_ap_objective_gradients_v1/EXPERIMENT_AUDIT_SOURCE.json').read_bytes())
assert audit['closure_status']=='CLOSED_WITH_LIMITS' and audit['deterministic_checks_status']=='pass'
assert shutil.disk_usage(parent).free>=4294967296
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert len(gpu.strip().splitlines())==1 and int(gpu.split(',')[1])<500
args=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.run_msvr_cross_scene_smooth_ap',
      '--config',str(config),'--config-sha256',expected,'--code-commit',commit,'--output-dir',str(output)]
command='cd '+shlex.quote(str(repo))+' && exec '+shlex.join(args)+' > '+shlex.quote(str(log))+' 2>&1'
subprocess.run(['screen','-dmS',session,'bash','-lc',command],check=True)
data=dict(status='SCREEN_LAUNCH_RETURNED',started_at=datetime.datetime.now().astimezone().isoformat(),
          code_commit=commit,config_sha256=expected,session=session,output=str(output),log=str(log),
          command=args,free_bytes=shutil.disk_usage(parent).free,gpu=gpu,
          scope='Launch receipt only; pipeline process and T0/M0 still require observation')
receipt.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
