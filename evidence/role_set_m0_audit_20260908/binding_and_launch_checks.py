"""Read-only explicit nested binding and launch-commit checks."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
spec=read(ROOT/'configs/MSVR310/TriFusion-role-set-paired-v1.json')
cfg=read(ROOT/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
base=read(ROOT/cfg['BASELINE']['CONFIG'])
summary=read(RUN/'m0/summary.json')
pipeline=read(RUN/'pipeline.json')
assert summary['project_commit']==pipeline['code_commit']=='26c97390704c629237687d263b4381f5584cbe97'
assert summary['config_sha256']==pipeline['config_sha256']==sha(ROOT/'configs/MSVR310/TriFusion-role-set-paired-v1.json')
assert [s['stage'] for s in pipeline['stages'][:4]]==['t0','m0','m0_cpu','q1']
for stage in pipeline['stages'][:3]:assert stage['exit_code']==0
assert pipeline['stages'][3]['started_at']>pipeline['stages'][2]['ended_at']
bindings=[]
for path,digest in [(ROOT/cfg['BASELINE']['CONFIG'],cfg['BASELINE']['CONFIG_SHA256']),(Path(cfg['BASELINE']['SUMMARY']),cfg['BASELINE']['SUMMARY_SHA256']),(ROOT/cfg['DATA']['PROTOCOL'],cfg['DATA']['PROTOCOL_SHA256']),(ROOT/cfg['SOURCE_METADATA']['PATH'],cfg['SOURCE_METADATA']['SHA256']),(Path(base['clip_weight']),base['clip_weight_sha256'])]:
    actual=sha(path);assert actual==digest
    bindings.append(dict(path=str(path),sha256=actual))
launch=[]
for path in ['configs/MSVR310/TriFusion-role-set-paired-v1.json',*spec['project_file_sha256']]:
    committed=subprocess.check_output(['git','-C',str(ROOT),'show',pipeline['code_commit']+':'+path])
    current=(ROOT/path).read_bytes()
    assert current==committed
    launch.append(dict(path=path,sha256=hashlib.sha256(current).hexdigest(),exact_launch_commit_bytes=True))
signal=Path(base['signal_source'])
assert subprocess.check_output(['git','-C',str(signal),'rev-parse','HEAD'],text=True).strip()==base['signal_commit']
assert hashlib.sha256(subprocess.check_output(['git','-C',str(signal),'diff','--binary'])).hexdigest()==base['signal_diff_sha256']
predecessors=[]
paths=['configs/MSVR310/TriFusion-role-set-paired-v1.json','configs/MSVR310/TriFusion-history-gradient-paired-v1.json','configs/MSVR310/TriFusion-fresh-coordinate-paired-v1.json','configs/MSVR310/TriFusion-instance-memory-paired-v1.json']
for path in paths:
    config=read(ROOT/path)
    for bound,digest in config['fixed_file_sha256'].items():
        payload=read(bound)
        assert sha(bound)==digest
        predecessors.append(dict(config=path,path=bound,sha256=digest,status=payload.get('status'),summary_sha256=payload.get('summary_sha256')))
text_paths=['tools/train_msvr310_source_style.py','tools/train_msvr_instance_memory.py','tools/train_msvr310_trifusion_oof.py','tools/train_msvr310_signal_oof.py','tools/build_v12_complete_path_oof_targets.py','tools/run_signal_preserving_v5.py','tools/train_signal_preserving_v17.py','modeling/trifusion/signal_preserving_v8.py','modeling/trifusion/signal_preserving_v8_builder.py','modeling/trifusion/experts/semantic_residual.py','modeling/trifusion/experts/mamba.py','modeling/trifusion/aligned_data.py']
active_source=[]
for path in text_paths:
    current=(ROOT/path).read_bytes()
    committed=subprocess.check_output(['git','-C',str(ROOT),'show',pipeline['code_commit']+':'+path])
    active_source.append(dict(path=path,sha256=hashlib.sha256(current).hexdigest(),launch_commit_raw_bytes_equal=current==committed,launch_commit_ast_equal=ast.dump(ast.parse(current.decode()))==ast.dump(ast.parse(committed.decode()))))
    assert active_source[-1]['launch_commit_ast_equal']
owned=[p for p in (RUN/'m0').rglob('*') if p.is_file()]+[RUN/n for n in ('t0.json','t0.log','m0.log','m0_cpu.json','m0_cpu.log')]
print(json.dumps(dict(status='PASS_EXPLICIT_BINDINGS_AND_LAUNCH_BYTES',nested_bindings=bindings,primary_launch_files=launch,active_dependency_launch_checks=active_source,predecessor_status_bindings=predecessors,closed_stage_receipts=pipeline['stages'][:3],original_wrapper_pid=pipeline['wrapper_pid'],original_q1_pid=pipeline['stages'][3]['original_pid'],gpu_memory_mib_at_launch=int(pipeline['gpu_before'].split(',')[1]),free_bytes_before=pipeline['free_bytes_before'],m0_and_t0_cpu_owned_files=len(owned),m0_and_t0_cpu_owned_bytes=sum(p.stat().st_size for p in owned),q1_artifact_reads=0,model_forwards=0,image_reads=0),indent=2))
