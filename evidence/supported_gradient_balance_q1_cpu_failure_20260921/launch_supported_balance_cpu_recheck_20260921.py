from pathlib import Path
import json,subprocess,sys
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
out=run/'q1_cpu_sqrt_recheck'
assert not out.exists();out.mkdir()
assert json.loads((run/'pipeline.json').read_text())['status']=='STOPPED_AT_Q1_CPU'
command=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.verify_msvr_supported_gradient_balance','--config',str(repo/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json'),'--run-dir',str(run/'q1'),'--output',str(out/'verification.json')]
worker=out/'worker.py'
worker.write_text('import subprocess,json,time,datetime\nfrom pathlib import Path\np=Path('+repr(str(out))+')\ncommand='+repr(command)+'\nstart=time.time()\nwith (p/"verification.log").open("w") as log:\n r=subprocess.run(command,cwd='+repr(str(repo))+',stdout=log,stderr=subprocess.STDOUT)\n(p/"receipt.json").write_text(json.dumps(dict(exit_code=r.returncode,command=command,elapsed_seconds=time.time()-start,ended_at=datetime.datetime.now().astimezone().isoformat())))\n')
with (out/'worker.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,str(worker)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,output=str(out),original_pipeline_unchanged=True)))
