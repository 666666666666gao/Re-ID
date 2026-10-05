from datetime import datetime
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');output=Path('/data/gaob/Re-ID/Trifusion/results/deployment_metric_fixed_best_five_20261005_850');launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_fixed_best_five_launch_20261005_850')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
seal_path=root/'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal_path)=='a4de9faca7f502cb7a4fa01800e437919810d2cfa83bfcf0361047d831a42cb7'
seal=json.loads(seal_path.read_text())
assert seal['schema']=='trifusion-deployment-metric-role-five-fixed-best-diagnosis-v1' and seal['planned_models']==5 and len(seal['rows'])==8
assert len(seal['source_sha256'])==341 and seal['scientific_source_count']==339
assert all(sha(root/name)==digest for name,digest in seal['source_sha256'].items())
assert all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='7471d607b0d881733b292528aabde76f7b904e1b'
assert json.loads((root/'logs/deployment_metric_storage_launch_20261005_848/EXIT.json').read_text())['exit_code']==0
assert json.loads((root/'logs/deployment_metric_storage_five_report_launch_20261005_850/EXIT.json').read_text())['exit_code']==0
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>2*1024**3
assert not output.exists() and not launch.exists()
launch.mkdir()
(launch/'supervisor.py').write_text("from datetime import datetime\nfrom pathlib import Path\nimport json,os,subprocess\nlaunch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_fixed_best_five_launch_20261005_850')\ncommand=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_fixed_best_diagnosis_v1/DIAGNOSE_FIVE.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_v1_20261005_837', '--seal', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json', '--output-dir', '/data/gaob/Re-ID/Trifusion/results/deployment_metric_fixed_best_five_20261005_850']\nwith (launch/'console.log').open('x') as log:\n result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n")
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
record=dict(status='FIVE_FIXED_BEST_DIAGNOSIS_LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=process.pid,
 start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),launch_dir=str(launch),output_dir=str(output),command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_fixed_best_diagnosis_v1/DIAGNOSE_FIVE.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_v1_20261005_837', '--seal', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json', '--output-dir', '/data/gaob/Re-ID/Trifusion/results/deployment_metric_fixed_best_five_20261005_850'],
 seal_sha256='a4de9faca7f502cb7a4fa01800e437919810d2cfa83bfcf0361047d831a42cb7',source_commit='7471d607b0d881733b292528aabde76f7b904e1b',planned_models=5,scientific_source_count=339,diagnosis_source_count=341,
 gpu_memory_only=devices,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Five fixed accepted bests, original complete eval records, sequential26GPU0/1. Original missing MSVRnative has no replay. No training, optimizer updates, new selection, temperature-power action, retired probe verifier or CPU-report replay.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
