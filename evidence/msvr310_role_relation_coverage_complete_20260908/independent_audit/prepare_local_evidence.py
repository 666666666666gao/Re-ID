"""Snapshot all direct text inputs and decode the raw remote read-only inventory."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
PROJECT=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
INTAKE=Path('C:/Users/gb/.codex_tmp/msvr_role_relation_coverage_complete_20260908')
request=Path('C:/Users/gb/.codex_tmp/msvr_role_relation_coverage_audit_request_20260908.md')
files=['AGENTS.md','tools/analyze_msvr_role_relation_coverage.py','tools/run_msvr_role_relation_coverage.py',
       'configs/MSVR310/Role-relation-coverage-v1.json',
       'refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_PLAN.md',
       'refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_TRACKER.md',
       'protocols/msvr310_train_oof_v1.json','tools/diagnose_msvr_source_relations.py',
       'tools/msvr_source_relation_math.py','configs/MSVR310/Source-relation-census-v1.json',
       'tools/build_msvr310_train_oof_protocol.py','tools/audit_vehicle_query_protocol_labels.py',
       'evidence/vehicle_query_protocol_labels_20260905.json',
       'evidence/msvr310_source_relation_complete_20260907/summary.json',
       'evidence/msvr310_source_relation_complete_20260907/cpu_verification.json',
       'evidence/msvr310_source_relation_complete_20260907/pipeline.json']
inputs=[]
def save_input(path,relative):
    raw=path.read_bytes()
    dest=OUT/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(raw)
    inputs.append(dict(path=str(path),snapshot=str(dest.relative_to(OUT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))

save_input(request,'reviewer_request.md')
for name in files:
    save_input(PROJECT/name,'local_snapshots/project/'+name)
for name in ['intake_manifest.json','pipeline.json','analysis.log','analysis/summary.json','analysis/all_query_relations.jsonl']:
    save_input(INTAKE/name,'local_snapshots/intake/'+name)
for name in ['experiment-audit/SKILL.md','shared-references/local-codex-policy.md',
             'shared-references/reviewer-independence.md','shared-references/experiment-integrity.md',
             'shared-references/review-tracing.md']:
    save_input(Path('C:/Users/gb/.agents/skills')/name,'policy_snapshots/'+name)
inventory=[]
remote_inputs=[]
prefixes={'/root/autodl-tmp/trifusion-v2/TriFusion-ReID/':'project/',
          '/root/autodl-tmp/trifusion-v2/artifacts/msvr310_source_relations_v1_seed42_4e57e54/':'source/',
          '/root/autodl-tmp/trifusion-v2/artifacts/msvr310_role_relation_coverage_v1_seed42_c46be4e/':'result/',
          '/root/autodl-tmp/trifusion-v2/artifacts/':'artifacts/'}
for line in (OUT/'inspect_remote_inputs.stdout.jsonl').read_text(encoding='utf-8').splitlines():
    row=json.loads(line)
    if row['kind']=='document':
        raw=base64.b64decode(row.pop('contents_b64'))
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        prefix=next(p for p in prefixes if row['path'].startswith(p))
        relative='remote_snapshots/'+prefixes[prefix]+row['path'][len(prefix):]
        dest=OUT/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
        row['snapshot']=relative
        remote_inputs.append(row)
    inventory.append(row)
(OUT/'local_input_hashes.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
(OUT/'remote_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
(OUT/'remote_text_input_hashes.json').write_text(json.dumps(remote_inputs,indent=2)+'\n',encoding='utf-8')
trace=OUT/'trace'
trace.mkdir(exist_ok=True)
routing=dict(requested_model='gpt-6-astra',requested_reasoning_effort='max',requested_fork_turns='none',
             canonical_task_id='/root/audit_msvr_role_relation_coverage',agent_id=None,
             backend_attestation=None,review_independence='same-family',acceptance_status='provisional')
(trace/'run.meta.json').write_text(json.dumps(dict(skill='experiment-audit',started_at=datetime.now(timezone.utc).isoformat(),
    executor='codex',project_dir=str(PROJECT),**routing),indent=2)+'\n',encoding='utf-8')
(trace/'001-role-relation-coverage.request.json').write_text(json.dumps(dict(call_number=1,purpose='fresh-read-only-experiment-audit',
    prompt=request.read_text(encoding='utf-8'),source_request_path=str(request),**routing),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(local_inputs=len(inputs),remote_text_inputs=len(remote_inputs),
    process_and_git=[r for r in inventory if r['kind'] in ['environment','process','git','commit_blob']],
    result_inventory=[r for r in inventory if r['kind']=='result_inventory']),indent=2))
