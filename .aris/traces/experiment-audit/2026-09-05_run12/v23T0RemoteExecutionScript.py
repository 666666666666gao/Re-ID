from pathlib import Path
from datetime import datetime
import subprocess,json,hashlib,os,time,xml.etree.ElementTree as ET
r=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
art=Path("/root/autodl-tmp/trifusion-v2/artifacts")
started=datetime.now().astimezone().isoformat();tic=time.time()
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=r,text=True).strip()=="9f4a10b6162b9658ba103cd92466411ebb6ccd8f"
gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
parts=[p.strip() for p in gpu.split(",")]
assert len(parts)==4 and int(parts[2])-int(parts[1])>=22000
paths=["modeling/trifusion/signal_preserving_v23.py","tools/train_signal_preserving_v23.py","tests/test_trifusion_signal_preserving_v23.py","configs/RGBNT201/TriFusion-signal-preserving-v23-spectral-adapter-rtx3090.yml","refine-logs/v23/EXPERIMENT_PLAN.md"]
hashes={x:hashlib.sha256((r/x).read_bytes()).hexdigest() for x in paths}
assert hashes["tools/train_signal_preserving_v23.py"]=="1b18edbd28e335469f6647a7095228e9e03cf8195b2739b4a9c54c62aedec42b"
env=dict(os.environ,PYTHONPATH=str(r)+":"+str(r/"modeling")+":"+str(r/"tests"),CUDA_VISIBLE_DEVICES="0")
log=art/"trifusion_v23_t0_9f4a10b.log"
xml=art/"trifusion_v23_t0_9f4a10b.xml"
assert not log.exists() and not xml.exists()
command=["/root/miniconda3/envs/tri_reid/bin/python","-m","pytest","-q","tests/test_trifusion_signal_preserving_v23.py","--junitxml="+str(xml)]
result=subprocess.run(command,cwd=r,env=env,capture_output=True,text=True)
log.write_bytes((result.stdout+result.stderr).encode())
root=ET.fromstring(xml.read_bytes());suite=root[0]
report={"started_at":started,"finished_at":datetime.now().astimezone().isoformat(),"elapsed_seconds":time.time()-tic,
"execution_commit":"9f4a10b6162b9658ba103cd92466411ebb6ccd8f","command":command,"gpu_before":gpu,
"source_file_sha256":hashes,"returncode":result.returncode,"test_summary":dict(suite.attrib),
"log_path":str(log),"log_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),"junit_path":str(xml),
"junit_sha256":hashlib.sha256(xml.read_bytes()).hexdigest(),"real_dataset_access_count":0,"source_checkpoint_load_count":0,
"real_model_instantiations":0,"synthetic_model_instantiations":6,"standalone_spectral_stage_instantiations":1,
"toy_optimizer_steps":6,"project_optimizer_steps":0,"evaluation_type":"synthetic_cuda_zero_adapter_modality_dispatch_gradient_and_strict_reload_contract"}
report["passed"]=result.returncode==0 and suite.attrib["tests"]=="5" and suite.attrib["failures"]=="0" and suite.attrib["errors"]=="0"
out=art/"trifusion_v23_t0_9f4a10b.json";out.write_bytes((json.dumps(report,indent=2)+"\n").encode())
print(json.dumps(report,indent=2))
print(result.stdout[-5000:])
