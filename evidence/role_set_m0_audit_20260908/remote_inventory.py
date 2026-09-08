"""Read-only remote provenance collection; no imports of project runtime modules."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
files={}; texts={}; bindings=[]; configs={}

def record(path,text=False):
    path=Path(path)
    key=str(path)
    if key not in files:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        files[key]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
    if text:
        texts[key]=path.read_text(encoding='utf-8-sig')
    return files[key]

def strings(value):
    if isinstance(value,dict):
        for v in value.values():yield from strings(v)
    elif isinstance(value,list):
        for v in value:yield from strings(v)
    elif isinstance(value,str):yield value

pending=['configs/MSVR310/TriFusion-role-set-paired-v1.json']
while pending:
    name=pending.pop(0)
    if name in configs:continue
    record(ROOT/name,True)
    config=json.loads(texts[str(ROOT/name)]);configs[name]=config
    for kind in ('project_file_sha256','project_source_file_sha256','fixed_file_sha256','signal_source_file_sha256'):
        if kind not in config:continue
        parent=Path(config['signal_source']) if kind=='signal_source_file_sha256' else ROOT
        for path,digest in config.get(kind,{}).items():
            actual=record(parent/path,not str(path).startswith('/'))
            bindings.append({'config':name,'kind':kind,'path':str(parent/path),'expected_sha256':digest,**actual,'pass':actual['sha256']==digest})
    pending.extend(s for s in strings(config) if s.startswith('configs/') and s.endswith('.json'))

base=configs['configs/MSVR310/Signal-source-oof-v1.json']
config=configs['configs/MSVR310/TriFusion-source-style-paired-v1-r2.json']
for path in (ROOT/base['protocol'],ROOT/config['SOURCE_METADATA']['PATH'],Path(config['BASELINE']['SUMMARY'])):
    record(path,True)
record(base['clip_weight'])
baseline=json.loads(texts[config['BASELINE']['SUMMARY']])
for fold in baseline['folds']:
    record(fold['checkpoint'])
    record(Path(fold['checkpoint']).parent/'retrieval_arrays.pt')

# Closed M0 only. The ongoing q1 directory and q1.log are never opened.
for path in sorted((RUN/'m0').rglob('*')):
    if path.is_file():record(path,path.suffix in ('.json','.jsonl','.log'))
for name in ('t0.json','t0.log','m0.log','m0_cpu.json','m0_cpu.log','pipeline.json'):
    record(RUN/name,True)

protocol=json.loads(texts[str(ROOT/base['protocol'])])
dataset_root=Path(base['dataset_root'])
metadata_names=[str(p.relative_to(dataset_root)) for p in dataset_root.iterdir() if p.is_file() and p.suffix.lower() in ('.txt','.json','.csv','.md')]
for name in metadata_names:record(dataset_root/name,True)
data_presence=[]
for row in protocol['records']:
    data_presence.append({'index':row['index'],'paths':[{ 'path':str(dataset_root/p),'is_file':(dataset_root/p).is_file(),'bytes':(dataset_root/p).stat().st_size} for p in row['paths']]})

pipeline=json.loads(texts[str(RUN/'pipeline.json')])
pids=[pipeline['wrapper_pid']]+[r['original_pid'] for r in pipeline['stages']]
processes={}
for pid in pids:
    p=Path('/proc')/str(pid)
    entry={'exists':p.is_dir()}
    if p.is_dir():
        entry['status_selected']=[s for s in (p/'status').read_text().splitlines() if s.startswith(('Name:','State:','Pid:','PPid:','Threads:'))]
        entry['cwd']=os.readlink(p/'cwd')
    processes[str(pid)]=entry

signal=Path(base['signal_source'])
result={'generated_at':datetime.now(timezone.utc).isoformat(),'uid':os.getuid(),'remote_repository':str(ROOT),'project_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'signal_head':subprocess.check_output(['git','-C',str(signal),'rev-parse','HEAD'],text=True).strip(),'signal_diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(signal),'diff','--binary'])).hexdigest(),'files':files,'texts':texts,'bindings':bindings,'dataset_root_metadata_files':metadata_names,'data_presence':data_presence,'original_processes':processes,'boundaries':{'gpu_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'cpu_thread_environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},'model_instantiations':0,'model_forwards':0,'image_reads':0,'q1_artifact_reads':0}}
print(json.dumps(result,ensure_ascii=False,indent=2))
