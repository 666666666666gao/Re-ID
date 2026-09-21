from pathlib import Path
import hashlib,json
tmp=Path('D:/Program Files/UserCache/gb/codex/tmp');repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
m=json.loads((tmp/'trifusion_task_state_integration_publication_20260921.json').read_bytes())
head=m['new_head'];config='configs/MSVR310/TriFusion-supported-task-state-paired-v1.json'
digest=hashlib.sha256((repo/config).read_bytes()).hexdigest()
payload='import os,json,subprocess,time,hashlib,shutil\nfrom pathlib import Path\n'
payload+=f'head={head!r}\nconfig={config!r}\ndigest={digest!r}\n'
payload+='''repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
parent=Path('/root/trifusion-storage/artifacts')
root=parent/('msvr310_supported_task_state_v1_seed42_'+head[:7])
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==head
assert hashlib.sha256((repo/config).read_bytes()).hexdigest()==digest
assert not root.exists()
assert shutil.disk_usage(parent).free>=4294967296
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
command=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.run_msvr_supported_task_state',
    '--config',str(repo/config),'--config-sha256',digest,'--code-commit',head,'--output-dir',str(root)]
environment=dict(os.environ,PYTHONPATH=str(repo/'modeling')+':'+str(repo),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
log=parent/(root.name+'_wrapper.log')
with log.open('x') as stream:
    process=subprocess.Popen(command,cwd=repo,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
print(json.dumps(dict(wrapper_pid=process.pid,observed_return_code=process.poll(),run=str(root),log=str(log),
    code_commit=head,config_sha256=digest,free_bytes=shutil.disk_usage(parent).free,
    pipeline=json.loads((root/'pipeline.json').read_bytes()) if (root/'pipeline.json').exists() else None),indent=2))
'''
(tmp/'launch_task_state_payload_20260921.py').write_text(payload,encoding='utf-8')
print(json.dumps(dict(head=head,config_sha256=digest)))
