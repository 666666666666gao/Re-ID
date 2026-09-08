"""Read-only remote input/state inventory, executed from stdin with bytecode off."""
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
SOURCE = Path('/root/autodl-tmp/trifusion-v2/artifacts/msvr310_source_relations_v1_seed42_4e57e54')
RESULT = Path('/root/autodl-tmp/trifusion-v2/artifacts/msvr310_role_relation_coverage_v1_seed42_c46be4e')
def emit(kind, **value):
    print(json.dumps(dict(kind=kind, **value), sort_keys=True), flush=True)

def document(path):
    raw = path.read_bytes()
    emit('document', path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
         contents_b64=base64.b64encode(raw).decode('ascii'))
    return json.loads(raw) if path.suffix == '.json' else None

emit('environment', time=datetime.now(timezone.utc).isoformat(), hostname=os.uname().nodename,
     uid=os.getuid(), cwd=os.getcwd(), executable=sys.executable, version=sys.version,
     dont_write_bytecode=sys.dont_write_bytecode, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'))
for relative in ['tools/analyze_msvr_role_relation_coverage.py','tools/run_msvr_role_relation_coverage.py',
                 'configs/MSVR310/Role-relation-coverage-v1.json',
                 'refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_PLAN.md',
                 'refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_TRACKER.md',
                 'protocols/msvr310_train_oof_v1.json','tools/diagnose_msvr_source_relations.py',
                 'tools/msvr_source_relation_math.py','configs/MSVR310/Source-relation-census-v1.json',
                 'tools/build_msvr310_train_oof_protocol.py','evidence/vehicle_query_protocol_labels_20260905.json',
                 'tools/audit_vehicle_query_protocol_labels.py']:
    document(PROJECT / relative)
for name in ['summary.json','cpu_verification.json','pipeline.json']:
    document(SOURCE/name)
new_pipeline = document(RESULT/'pipeline.json')
document(RESULT/'analysis.log')
document(RESULT/'analysis/summary.json')
for p in sorted(RESULT.rglob('*')):
    if p.is_file():
        raw = p.read_bytes()
        emit('result_inventory', path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
for name in ['wrapper_pid','analysis_pid']:
    pid = new_pipeline[name]
    emit('process', role=name, pid=pid, present=Path('/proc',str(pid)).exists())
old_spec = json.loads((PROJECT/'configs/MSVR310/Source-relation-census-v1.json').read_bytes())
document(Path(old_spec['q1_summary']))
for name in ['original_config']:
    document(PROJECT/old_spec[name])
source_summary=json.loads((SOURCE/'summary.json').read_bytes())
for condition in source_summary['conditions']:
    document(SOURCE/condition['directory']/'receipt.json')
    document(SOURCE/condition['directory']/'inputs.jsonl')
for args in [['rev-parse','HEAD'],['status','--porcelain','--untracked-files=no']]:
    process = subprocess.run(['git',*args],cwd=PROJECT,capture_output=True,text=True,check=False)
    emit('git', args=args, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
for commit, paths in [(new_pipeline['code_commit'],['tools/analyze_msvr_role_relation_coverage.py',
                              'tools/run_msvr_role_relation_coverage.py','configs/MSVR310/Role-relation-coverage-v1.json',
                              'refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_PLAN.md']),
                      (source_summary['code_commit'],['tools/diagnose_msvr_source_relations.py',
                              'tools/msvr_source_relation_math.py','configs/MSVR310/Source-relation-census-v1.json'])]:
    for name in paths:
        process=subprocess.run(['git','show',f'{commit}:{name}'],cwd=PROJECT,capture_output=True,check=False)
        emit('commit_blob', commit=commit, path=name, returncode=process.returncode,
             bytes=len(process.stdout), sha256=hashlib.sha256(process.stdout).hexdigest(),
             working_sha256=hashlib.sha256((PROJECT/name).read_bytes()).hexdigest(),
             stderr=process.stderr.decode('utf-8'))
