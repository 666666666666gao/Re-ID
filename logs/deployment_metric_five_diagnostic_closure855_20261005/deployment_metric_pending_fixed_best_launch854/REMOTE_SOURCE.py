from datetime import datetime
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_pending_fixed_best_launch_20261005_854');output=Path('/data/gaob/Re-ID/Trifusion/results/deployment_metric_fixed_best_pending100_20261005_854')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='4d8bf809f5d2139f3b290a40c5333ce9b5d24dcf'
assert all(sha(root/n)==d for n,d in {'refine-logs/deployment_metric_pending_fixed_best_v1/CONTINUE_RGBNT100.py': 'f3418619e4a0763c5c06a749b78b31ec4429a8b7ff3e268ea88f51c56fd736d0', 'refine-logs/deployment_metric_pending_fixed_best_v1/EXPERIMENT_PLAN.md': 'a474c615b5afe731bfdf0a3dd134edda0edf65ffb8e3155dba0934c4411f1d85'}.items())
assert all(sha(root/n)==d for n,d in {'results/deployment_metric_fixed_best_five_20261005_850/RGBNT201_semantic/DIAGNOSIS.json': '2569dbc407e58f806e6aabb2fda740cffc6bd074dd05e56670cfb27e7ea9900a', 'results/deployment_metric_fixed_best_five_20261005_850/RGBNT201_native/DIAGNOSIS.json': '0aae23547efb2b032284fdcd29ccaa37ce420f2b481a3f1dd85a5e7c9ed34c19', 'results/deployment_metric_fixed_best_five_20261005_850/MSVR310_semantic.log': '0005c97e4806e6b2fc64950966d31b83ef5bbb1299ff464e96696b4d531ab6ca', 'results/deployment_metric_fixed_best_five_20261005_850/campaign.json': '119f8b6b57fe14456bbc4fd239a9c43cb465ecd823cefd750b5bbdd95c07d359', 'logs/deployment_metric_fixed_best_five_launch_20261005_850/EXIT.json': '10058721907ae428a7aea4f95a4165adaa33aa2141dd9a3c62cb9eceace53dbd'}.items())
seal_path=root/'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal_path)=='a4de9faca7f502cb7a4fa01800e437919810d2cfa83bfcf0361047d831a42cb7'
seal=json.loads(seal_path.read_text())
assert len(seal['source_sha256'])==341 and len(seal['artifact_sha256'])==271
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
assert not Path('/proc/444043/stat').exists()
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>2*1024**3
assert not launch.exists() and not output.exists()
launch.mkdir();(launch/'supervisor.py').write_text("from datetime import datetime\nfrom pathlib import Path\nimport json,os,subprocess\nlaunch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_pending_fixed_best_launch_20261005_854')\nwith (launch/'console.log').open('x') as log:\n result=subprocess.run(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_pending_fixed_best_v1/CONTINUE_RGBNT100.py'],cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n")
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
record=dict(status='TWO_NEVER_STARTED100_FIXED_DIAGNOSES_LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),
 pid=process.pid,start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),source_commit='4d8bf809f5d2139f3b290a40c5333ce9b5d24dcf',
 launch_dir=str(launch),output_dir=str(output),command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_pending_fixed_best_v1/CONTINUE_RGBNT100.py'],planned_models=2,
 administrative_source_sha256={'refine-logs/deployment_metric_pending_fixed_best_v1/CONTINUE_RGBNT100.py': 'f3418619e4a0763c5c06a749b78b31ec4429a8b7ff3e268ea88f51c56fd736d0', 'refine-logs/deployment_metric_pending_fixed_best_v1/EXPERIMENT_PLAN.md': 'a474c615b5afe731bfdf0a3dd134edda0edf65ffb8e3155dba0934c4411f1d85'},original_failure_frozen_sha256={'results/deployment_metric_fixed_best_five_20261005_850/RGBNT201_semantic/DIAGNOSIS.json': '2569dbc407e58f806e6aabb2fda740cffc6bd074dd05e56670cfb27e7ea9900a', 'results/deployment_metric_fixed_best_five_20261005_850/RGBNT201_native/DIAGNOSIS.json': '0aae23547efb2b032284fdcd29ccaa37ce420f2b481a3f1dd85a5e7c9ed34c19', 'results/deployment_metric_fixed_best_five_20261005_850/MSVR310_semantic.log': '0005c97e4806e6b2fc64950966d31b83ef5bbb1299ff464e96696b4d531ab6ca', 'results/deployment_metric_fixed_best_five_20261005_850/campaign.json': '119f8b6b57fe14456bbc4fd239a9c43cb465ecd823cefd750b5bbdd95c07d359', 'logs/deployment_metric_fixed_best_five_launch_20261005_850/EXIT.json': '10058721907ae428a7aea4f95a4165adaa33aa2141dd9a3c62cb9eceace53dbd'},
 scientific_source_count=339,original_diagnosis_source_count=341,artifact_count=271,
 gpu_memory_only=devices,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Only first RGBNT100 semantic/native fixed diagnosis. Original201 complete/MSVRfailed unchanged. No active remote text sync, training, retry, new selection or power-temperature action.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
