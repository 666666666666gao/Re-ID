"""Bind the reviewer's reports and exact evidence; writes audit artifacts only."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
BASE=ROOT/'refine-logs/patch_memory_roles_v1'
E=BASE/'integrity_678_evidence'
NOW=datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
arrays=read(E/'integrity678_cpu_evidence.json')
supp=read(E/'integrity678_supplement_v2.json')
identity=read(E/'integrity678_identity_checks.json')
factories=[read(E/'integrity678_factory'/f'{ds}.json') for ds in ('RGBNT201','RGBNT100','MSVR310')]
jobs=read(E/'integrity678_factory/jobs.json')
assert len(arrays['checks'])==241 and len(supp['checks'])==226
assert all(c['status']=='PASS' for c in arrays['checks']+supp['checks'])
assert all(j['exit_code']==0 for j in jobs)
assert all(f['cuda_initialized'] is False and f['neural_forward_executed'] is False and len(f['endpoints'])==2 for f in factories)
assert all(x['status']=='PASS' for x in identity['rows'])
inputs={**arrays['hashes'],**supp['hashes']}
for f in factories:
    for row in f['endpoints']:
        inputs[row['mamba_source']]={'sha256':row['mamba_source_sha256'],'bytes':None}
local_checks=[]
for folder in ('patch_memory_complete_intake_20261001','patch_memory_recovery_intake_20261001_0038','patch_memory_recovery_intake_20261001_0111','patch_memory_100_diagnosis_intake_20261001'):
    base=ROOT/'logs'/folder;j=read(base/'INTAKE.json')
    refs=j['files'] if 'files' in j else j['file_sha256']
    for rel,meta in refs.items():
        expected=meta['sha256'] if isinstance(meta,dict) else meta
        p=base/rel
        local_checks.append({'path':str(p.relative_to(ROOT)),'actual_sha256':sha(p),'expected_sha256':expected,'status':'PASS' if sha(p)==expected else 'FAIL'})
manifest=read(ROOT/'logs/patch_memory_complete_intake_20261001/manifest.json')
local_source_counts={'present':0,'unavailable_local_only':[],'mismatched':[]}
for rel,expected in manifest['source_sha256'].items():
    p=ROOT/rel
    if p.exists():
        local_source_counts['present']+=1
        if sha(p)!=expected:local_source_counts['mismatched'].append(rel)
    else:local_source_counts['unavailable_local_only'].append(rel)
local={'generated_at':NOW,'archive_checks':local_checks,'runtime_source_mirror':local_source_counts,
       'boundary':'Remote artifacts are authoritative. Locally absent protocol/comparator files were audited remotely, not silently passed locally.'}
(E/'LOCAL_BINDINGS.json').write_text(json.dumps(local,indent=2)+'\n',encoding='utf-8')
assert all(x['status']=='PASS' for x in local_checks)
line_endings=read(E/'LOCAL_SOURCE_LINE_ENDINGS.json')
assert len(line_endings['rows'])==27 and all(x['only_CRLF_difference'] for x in line_endings['rows'])
for rel in ('results/PATCH_MEMORY_PARTIAL_2026-10-01.md','results/PATCH_MEMORY_ARCHIVED_TRAJECTORY_2026-10-01.md','results/PATCH_MEMORY_ARCHIVED_TRAJECTORY_2026-10-01.json','tools/report_patch_memory_complete.py','results/patch_memory_complete_20261001/SUMMARY.json','refine-logs/patch_memory_roles_v1/EXPERIMENT_PLAN.md','refine-logs/patch_memory_roles_v1/RECOVERY_PLAN_20261001.md','refine-logs/patch_memory_roles_v1/SLOT_DIAGNOSTIC_PLAN_20261001.md'):
    p=ROOT/rel;inputs[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
for rel in ('experiment-audit/SKILL.md','shared-references/local-codex-policy.md','shared-references/reviewer-independence.md','shared-references/experiment-integrity.md','shared-references/review-tracing.md'):
    p=Path('C:/Users/gb/.codex/skills')/rel;inputs[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
evidence={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in E.rglob('*') if p.is_file()}
md=BASE/'INTEGRITY_AUDIT_678_20261001.md'
trace=ROOT/'.aris/traces/experiment-audit/2026-10-01_patch_memory_678'
checks={
 'A_gt_provenance':{'status':'PASS','evaluation_type':'real_gt','details':'Independent on-disk split path set, filename identity/camera/scene, label-map and train/test identity-disjointness verification. No generated targets. Download-source authenticity was not re-audited.','evidence':['tools/build_official_three_dataset_protocols.py:32','tools/official_three_dataset_data.py:8','tools/run_correspondence_context_identity.py:207','integrity_678_evidence/integrity678_cpu_arrays.py:72']},
 'B_score_normalization':{'status':'PASS','details':'L2 feature normalization followed by ordinary distance ranking/AP/CMC and percentage conversion; no self-normalized retrieval score. Exact same-identity AND camera/scene exclusions.','evidence':['tools/run_official_three_dataset_roles.py:230','comparators/Signal-cd1b0a6/utils/metrics.py:68','comparators/Signal-cd1b0a6/utils/metrics.py:137']},
 'C_result_existence':{'status':'PASS','details':'Six authoritative endpoints; 210 source and three baseline hashes; 18 full matrix independent rankings/72 metric checks; all archive, payload, history and stdout bindings match.','evidence':['logs/patch_memory_roles_recovery_20261001/accepted_matrix.json:5','integrity_678_evidence/integrity678_cpu_arrays.py:154','integrity_678_evidence/integrity678_supplement_v2.py:24']},
 'D_called_evaluation':{'status':'PASS','details':'Inspected actual entry monkeypatch/install, training evaluator and final same-checkpoint evaluation; no unused helper promoted as a result.','evidence':['tools/run_patch_memory_roles.py:72','tools/run_correspondence_context_identity.py:137','tools/run_correspondence_context_identity.py:175','tools/run_correspondence_context_identity.py:218']},
 'E_scope':{'status':'WARN','details':'Three datasets, two modes, one development seed, 300 role-stage epochs plus prior dataset baseline training; official data used for per-epoch and prior method selection. No untouched test or multi-seed claim.','evidence':['refine-logs/patch_memory_roles_v1/EXPERIMENT_PLAN.md:17','refine-logs/patch_memory_roles_v1/EXPERIMENT_PLAN.md:24','refine-logs/patch_memory_roles_v1/EXPERIMENT_PLAN.md:44','tools/run_correspondence_context_identity.py:140']},
 'F_evaluation_type':{'status':'PASS','classification':{'formal_retrieval':'real_gt','slot_statistics':'self_supervised_proxy (GT-free internal descriptive diagnostic)','M0_and_CPU_factory':'engineering_validation; not a retrieval evaluation'},'evidence':['tools/run_correspondence_context_identity.py:207','tools/diagnose_patch_memory_slots.py:30']},
 'production_factory_and_initialization':{'status':'PASS','details':'Actual production Mamba class instantiated on CPU, exact six initial hashes and matched counts; twelve 144-key strict loads. Device moves redirected in the audit process only. No forward or CUDA initialization.','evidence':['modeling/trifusion/experts/mamba.py:29','modeling/trifusion/patch_memory_roles.py:10','tools/run_correspondence_roles.py:32','integrity_678_evidence/integrity678_factory/jobs.json:1']},
 'loss_accounting':{'status':'PASS','maximum_scalar_error':max(r['loss_max_error'] for r in arrays['endpoints']),'details':'20,416 training steps reconcile across all 300 epoch receipts; both RGBNT100 runs have zero positive Triplet steps; no scalar-to-gradient-share inference.','evidence':['tools/run_correspondence_context_identity.py:110','integrity_678_evidence/integrity678_cpu_evidence.json:2746']},
 'recovery_provenance':{'status':'WARN','details':'Original parent FAILED and failed attempts preserved; two full50 training/checkpoint hashes preserved with new evaluate exit 0; four fresh runs. Historical two reused train OS exits remain unavailable.','evidence':['refine-logs/patch_memory_roles_v1/RECOVER_PANEL_20261001.py:29','integrity_678_evidence/integrity678_supplement_v2.json:1880']},
 'diagnostic_scope':{'status':'WARN','details':'Identity bootstrap arithmetic independently matches, but is fixed-model resampling. Slot receipts/coverage/source/hook-parity and means verified, not neurally replayed. Old two slot GPU exits unavailable; RGBNT100 retained wait exits are zero. Shared-global is adapter-dependent and not a separately trained control.','evidence':['tools/diagnose_patch_memory_slots.py:79','logs/patch_memory_100_diagnosis_20261001/campaign.json:11','modeling/trifusion/correspondence_roles.py:64']},
 'registered_gate_reporting':{'status':'PASS','scientific_gate':'FAIL','details':'Correct original rule gives FAIL. A hypothesis/gain failure is not itself an integrity failure.','evidence':['refine-logs/patch_memory_roles_v1/EXPERIMENT_PLAN.md:36','integrity_678_evidence/integrity678_cpu_evidence.json:3186']},
 'local_source_mirror':{'status':'WARN','details':'103 of 210 manifest files are present locally: 76 raw hashes match, 27 differ only by CRLF/LF line endings; 107 are unavailable locally. All 210 remote primary files match exactly. No source bytes changed; normalized equality is not a raw-hash PASS.','evidence':['integrity_678_evidence/LOCAL_BINDINGS.json:1','integrity_678_evidence/LOCAL_SOURCE_LINE_ENDINGS.json:1']}}
unavailable=[{'check':'fresh_neural_gpu_or_cpu_forward','status':'UNAVAILABLE','reason':'Reviewer was authorized CPU evidence calculations and strict loads, not GPU inference; no neural forward executed.'},
 {'check':'fresh_real_batch_gradient_and_optimizer_replay','status':'UNAVAILABLE','reason':'Historical M0 receipts/source verified; no reviewer backward/optimizer execution.'},
 {'check':'two_reused_original_train_OS_exit_codes','status':'UNAVAILABLE','reason':'Original child workers stopped before recording terminal train statuses; source full50 receipts/logs/checkpoints retained. New evaluation exits do not replace missing old train exits.'},
 {'check':'original_RGBNT201_MSVR310_slot_GPU_OS_exit_codes','status':'UNAVAILABLE','reason':'No retained wait-parent exit receipts in reviewed artifacts; COMPLETE output is not an OS exit receipt.'},
 {'check':'untouched_test_and_multiseed_generalization','status':'UNAVAILABLE','reason':'One seed and official-set selection only.'},
 {'check':'fresh_download_origin_and_image_content_authenticity_audit','status':'UNAVAILABLE','reason':'Verified installed dataset paths and metadata; did not reacquire datasets or authenticate every image payload.'}]
audit={'audit_skill':'experiment-audit','date':'2026-10-01','generated_at':NOW,'verdict':'WARN','overall_verdict':'WARN','integrity_status':'warn','reason_code':'official_set_postselection_single_seed_and_replay_limits',
       'summary':'Six-end evidence and deterministic checks pass; original advancement gate fails. Scope/replay limits retained.',
       'agent_id':'/root/patch_memory_panel_integrity_678','opaque_agent_id':'unavailable','verdict_id':'/root/patch_memory_panel_integrity_678:2026-10-01:678',
       'reviewer_backend':'codex','reviewer_model':'gpt-6-astra','reviewer_reasoning':'max','reviewer_family':'openai','executor_model':'not_exposed_to_reviewer','executor_family':'openai',
       'model_attribution_source':'Parent explicitly confirmed actual spawn configuration: fork_turns=none, model=gpt-6-astra, reasoning_effort=max; canonical task confirmed by list_agents. Opaque agent id is not exposed and remains unavailable.',
       'review_independence':'same-family','acceptance_status':'provisional','review_context':'Fresh delegated artifact review; SOUL/USER/recent daily notes read per workspace AGENTS. Conclusions grounded in direct primary reads and own deterministic checks, not executor summaries.',
       'source_root_remote':'/data/gaob/Re-ID/Trifusion','local_root':str(ROOT),'checks':checks,'unavailable_checks':unavailable,
       'deterministic_execution':{'acceptance_status':'accepted','review_independence':'deterministic','primary_checks':241,'supplemental_checks':226,'local_archive_checks':len(local_checks),'identity_diagnostic_checks':3,'factory_jobs':jobs,'strict_loads':12,
          'full_distance_matrices':18,'retrieval_metrics':72,'max_ranking_error_pp':max(m['max_metric_error_pp'] for e in arrays['endpoints'] for m in e['ranking'].values()),
          'optimizer_training_steps_accepted':sum(r['steps'] for r in arrays['endpoints']),'accepted_M0_steps':48,'failed_attempt_logged_steps':289,'superseded_M0_steps':16,
          'accepted_training_plus_epoch_eval_seconds':sum(r['training_seconds'] for r in arrays['endpoints']),
          'cost_boundary':'Sum of accepted training receipt intervals; excludes M0, strict evaluation, failed attempts, source baselines and diagnostics. It is not campaign wall time.'},
       'paired_results':arrays['pairwise'],'registered_advancement_gate':'FAIL','factory_evidence':factories,
       'claims':[{'id':'matched_full_vs_local_negative_result','impact':'supported'},{'id':'generalizable_gain_or_SOTA','impact':'unsupported'},{'id':'unique_slot_redundancy_cause','impact':'unsupported'},{'id':'descriptive_attention_similarity','impact':'supported_with_scope_qualifier'}],
       'audited_input_hashes':{p:'sha256:'+meta['sha256'] for p,meta in inputs.items()},'audited_input_file_sizes':{p:meta['bytes'] for p,meta in inputs.items()},
       'evidence_artifacts':evidence,'report_markdown':str(md.relative_to(ROOT)),'report_markdown_sha256':sha(md),'trace_path':str(trace.relative_to(ROOT)),
       'helper_failures_preserved':[read(E/'integrity678_supplement_v1_failure.json'),read(E/'finalize_integrity678_v1_failure.json')],
       'actions':'Retain failed gate and postselection/one-seed/diagnostic qualifiers. Do not label CPU loads as neural replay or same-family review as cross-family acceptance.'}
jsonpath=BASE/'INTEGRITY_AUDIT_678_20261001.json'
assert not jsonpath.exists()
jsonpath.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
trace.mkdir(parents=True,exist_ok=False)
prompt=Path('C:/Users/gb/.codex_tmp/integrity678_review_request.txt').read_text(encoding='utf-8')
meta={'skill':'experiment-audit','run_id':'2026-10-01_patch_memory_678','started_at':arrays['started_at'],'completed_at':NOW,'executor':'codex','reviewer_model':'gpt-6-astra','reviewer_reasoning':'max','reviewer_family':'openai','review_independence':'same-family','acceptance_status':'provisional','project_dir':str(ROOT)}
(trace/'run.meta.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
(trace/'001-integrity-review.request.json').write_text(json.dumps({'call_number':1,'purpose':'patch-memory-panel-integrity-678','tool':'collaboration.spawn_agent','fork_turns':'none','model':'gpt-6-astra','reasoning_effort':'max','agent_id':audit['agent_id'],'opaque_agent_id':'unavailable','prompt':prompt,'checklist_source':'C:/Users/gb/.codex/skills/experiment-audit/SKILL.md'},indent=2)+'\n',encoding='utf-8')
(trace/'001-integrity-review.response.md').write_bytes(md.read_bytes())
(trace/'001-integrity-review.meta.json').write_text(json.dumps({'call_number':1,'purpose':'patch-memory-panel-integrity-678','timestamp':NOW,'agent_id':audit['agent_id'],'model':'gpt-6-astra','reasoning_effort':'max','reviewer_family':'openai','review_independence':'same-family','acceptance_status':'provisional','status':'ok','verdict':'WARN','response_sha256':sha(md),'bound_json_sha256':sha(jsonpath)},indent=2)+'\n',encoding='utf-8')
events=ROOT/'.aris/meta/events.jsonl';events.parent.mkdir(parents=True,exist_ok=True)
with events.open('a',encoding='utf-8') as f:f.write(json.dumps({'event':'review_trace','skill':'experiment-audit','purpose':'patch-memory-panel-integrity-678','agent_id':audit['agent_id'],'trace_path':str(trace.relative_to(ROOT)),'status':'ok','at':NOW})+'\n')
print(json.dumps({'markdown':str(md),'json':str(jsonpath),'markdown_sha256':sha(md),'json_sha256':sha(jsonpath),'local_archive_checks':len(local_checks),'remote_input_bindings':len(inputs),'local_source_present':local_source_counts['present']}))
