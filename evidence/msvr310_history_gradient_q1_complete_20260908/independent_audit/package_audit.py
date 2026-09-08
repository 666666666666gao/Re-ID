"""Package this reviewer's completed read-only audit; writes only beside this file."""
import hashlib,json,re,sys
from datetime import datetime,timezone
from pathlib import Path
from inspect_inputs import OUT,REPO,record

def obj(path):return json.loads(Path(path).read_bytes())
def sha(data):return hashlib.sha256(data).hexdigest()
def put(path,value):
 path=Path(path).resolve();assert path.is_relative_to(OUT)
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def ref(path,line):return f'{REPO.as_posix()}/{path}:{line}'
def outref(path,line):return f'{OUT.as_posix()}/{path}:{line}'

replay=obj(OUT/'remote_replay.stdout.json');contracts=obj(OUT/'remote_contract_checks.stdout.json')
local=obj(OUT/'local_validation.json');provenance=obj(OUT/'provenance_check.json');inventory=obj(OUT/'remote_inventory.stdout.json')
assert replay['status']=='PASS_INDEPENDENT_SAVED_ARTIFACT_REPLAY'
assert contracts['status']=='PASS_INDEPENDENT_CONTRACT_SAMPLER_T0_M0_CHECKS'
assert local['status']=='PASS_INDEPENDENT_ALL_LOCAL_TEXT_CSV_VALIDATION'
assert replay['scientific_status']=='Q1_FAIL'
assert replay['model_forwards']==replay['optimizer_updates']==replay['official_image_reads']==0
assert not any(replay['paired']['checks'].values())
assert all(not any(x['checks'].values()) for x in replay['endpoints'].values())

trace=OUT/'trace';trace.mkdir(exist_ok=True)
parent_trace=REPO/'.aris/traces/experiment-audit/2026-09-08_history_gradient_q1'
for name in ('001-complete-q1.meta.json','001-complete-q1.request.json'):
 data=record(parent_trace/name);(trace/('original_'+name)).write_bytes(data)
request=Path('C:/Users/gb/.codex_tmp/history_gradient_q1_audit_request_draft_20260908.md')
request_data=record(request)
(trace/'reviewer_prompt_from_draft.md').write_text('## Reviewer prompt'+request_data.decode('utf-8-sig').split('## Reviewer prompt',1)[1],encoding='utf-8')
for path in [Path('C:/Users/gb/.agents/skills/experiment-audit/SKILL.md')]+[Path('C:/Users/gb/.agents/skills/shared-references')/name for name in ('local-codex-policy.md','reviewer-independence.md','experiment-integrity.md','review-tracing.md')]:
 record(path)

# Preserve the exact already-inspected status documents, without silently replacing
# the audit-time hash should the executor have since published a correction.
old_registry=obj(OUT/'audited_input_hashes.json');snapshots=[]
for name in ('AGENTS.md','refine-logs/msvr310_history_gradient_v1/EXPERIMENT_TRACKER.md','refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md','refine-logs/msvr310_history_gradient_v1/POST_Q1_RESEARCH_CANDIDATES.md','docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md','results/MSVR310_HISTORY_GRADIENT_V1_M0_2026-09-08.md'):
 p=(REPO/name).resolve();data=p.read_bytes();old=old_registry[str(p)]['sha256'];current=sha(data)
 item=dict(path=p.as_posix(),original_audited_sha256=old,current_sha256=current,unchanged_since_read=old==current)
 if old==current:
  target=OUT/'audited_document_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);item['snapshot']=target.as_posix()
 snapshots.append(item)
put(OUT/'document_snapshot_bindings.json',snapshots)

local_hashes=obj(OUT/'audited_input_hashes.json');all_hashes={}
def merge(rows,kind):
 for path,info in rows.items():
  key=path.replace('\\','/');new=dict(info);new['coverage']=kind
  assert re.fullmatch('[0-9a-f]{64}',new['sha256'])
  if key in all_hashes:
   assert all_hashes[key]['sha256']==new['sha256'],key
   all_hashes[key]['coverage']=sorted(set(all_hashes[key]['coverage']+[kind]))
  else:new['coverage']=[kind];all_hashes[key]=new
merge(local_hashes,'local_read_or_audit_artifact_input')
merge(replay['audited_input_hashes'],'independent_saved_array_checkpoint_protocol_replay_input')
merge(contracts['audited_input_hashes'],'independent_contract_sampler_queue_input')
merge(inventory['read_hashes'],'live_remote_inventory_read')
# Byte-preserved remote source snapshots are already covered as local read hashes;
# the original path map retains the original remote source names and source digest.

checks={
 'gt_provenance':dict(section='A',status='pass',details='Real filename-derived labels; full identity/path separation and MSVR scene/time masks reconstructed. Official image zero-access remains code/receipt evidence, not OS attestation.',evidence=[ref('tools/build_msvr310_train_oof_protocol.py',12),ref('tools/build_msvr310_train_oof_protocol.py',58),ref('tools/train_msvr310_signal_oof.py',223),outref('remote_replay.stdout.json',4)]),
 'score_normalization':dict(section='B',status='pass',details='Uniform representation normalization and squared Euclidean retrieval; all saved distances/ranks, AP/CMC, identities and gates independently reconstructed. No model-dependent metric denominator.',evidence=[ref('tools/train_msvr310_trifusion_oof.py',196),outref('remote_replay.stdout.json',1313)]),
 'result_existence':dict(section='C',status='warn',numerical_result_status='pass',details='29/29 intake and live remote bytes agree; six endpoints and all stages complete. Audited latest tracker/handoff/AGENTS still show intermediate execution.',evidence=[outref('intake_check.json',1),outref('provenance_check.json',2),outref('local_validation.json',2),ref('refine-logs/msvr310_history_gradient_v1/EXPERIMENT_TRACKER.md',30),ref('docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',5993)]),
 'dead_code':dict(section='D',status='pass',details='Reported evaluator outputs traced to actual executed pipeline call sites; actual pinned Signal sampler/evaluator distinguished from unused alternatives.',evidence=[ref('tools/run_msvr_history_gradient.py',26),ref('tools/train_msvr_history_gradient.py',303),ref('tools/run_signal_baseline_dev.py',57)]),
 'scope':dict(section='E',status='warn',details='One configuration, seed42, reused internal train-split identities, three folds and two endpoints. No untouched-test/multi-seed/causal mechanism/novelty conclusion.',evidence=[ref('configs/MSVR310/TriFusion-history-gradient-paired-v1.json',3),ref('refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md',30),ref('refine-logs/msvr310_history_gradient_v1/POST_Q1_RESEARCH_CANDIDATES.md',38)]),
 'eval_type':dict(section='F',status='pass',primary_type='real_gt',qualifier='internal train-split identity OOF, reused for research',m0='source-only real-label capacity/overfit engineering diagnostics',t0='synthetic math and deterministic queue unit checks; not performance',source_probe='fixed-model source diagnostic; queue recency is not parameter staleness',official_test=False,evidence=[ref('tools/train_msvr_history_gradient.py',317),ref('tools/train_msvr_history_gradient.py',333),ref('tools/probe_msvr_history_candidate_gradients.py',229)])
}
limitations=[
 'No independent full model forward/backward/training replay; saved arrays and recorded scalars were independently recomputed within their available scope.',
 'Historical leaf gradient norms/support were reconstructed from saved distance geometry, not parameter-space vector directions.',
 'Runtime bitwise re-encoding, RNG/buffer preservation and final applied parameter gradients remain runtime witnesses plus inspected code; full per-step parameter arrays were not available.',
 'Six M0 direct-graph checks cover the first single-history-group witness only; no independent full multi-group model equivalence proof.',
 'Checkpoint states and aliases were reconstructed, but source baseline training, original initialization and final image-level extraction were not rerun.',
 'Recorded official-image zero-read counters are declarative; no OS file-open attestation for original training or external historical activity.',
 'Single seed and reused internal identities; bootstrap is conditional whole-identity resampling with query-count weighting, not multi-seed or untouched-test uncertainty.',
 'Compute counts are record-forward/VJP instrumentation and runtime timers/allocator peaks, not full FLOPs or a cheapest detached-history baseline benchmark.',
 'Core effective source and declared SHA bindings were audited; complete third-party binary environment and all unused transitive code were not independently certified.',
 'Requested reviewer routing is not independently verified backend model identity; review is same-family provisional.'
]
q1training=[x for x in replay['training'] if x['mode']=='q1']
costs={}
for end in ('control','history_gradient'):
 rows=[x for x in q1training if x['endpoint']==end]
 costs[end]={key:sum(x[key] for x in rows) for key in ('updates','historical_updates','candidate_record_exposures','fresh_record_forwards','vjp_record_forwards','zero_upstream_group_skips','training_seconds')}
 costs[end]['peak_allocated_mib']=max(x['peak_allocated_mib'] for x in rows)
 costs[end]['peak_reserved_mib']=max(x['peak_reserved_mib'] for x in rows)
report_data=(OUT/'reviewer_full_response.md').read_bytes()
assert len(report_data)>20000
now=datetime.now(timezone.utc).isoformat()
audit=dict(
 schema='independent-experiment-audit-v1',audit_skill='experiment-audit',verdict='WARN',overall_verdict='warn',integrity_status='warn',deterministic_checks_status='PASS',scientific_qualification='Q1_FAIL',
 reason_code='evidentiary_and_scope_limits_with_stale_live_status',reason_codes=['stale_latest_tracker_handoff_status','runtime_gradient_witnesses_not_full_model_reproduction','single_seed_reused_internal_q1_scope','same_family_provisional_review'],
 summary='Complete raw Q1 evidence and all deterministic numerical checks agree. Scientific gates fail. Integrity warnings preserve stale live-status observation and limits of runtime gradient/access/reproducibility evidence.',
 date='2026-09-08',generated_at=now,agent_id='/root/audit_msvr_history_gradient_q1',verdict_id='/root/audit_msvr_history_gradient_q1',
 executor_model=None,executor_family='openai',reviewer_model='gpt-6-astra',reviewer_reasoning='max',reviewer_family='openai',reviewer_backend='codex',
 model_attribution='requested routing in retained native spawn evidence; backend identity not independently attested',backend_identity_verified=False,model_routing=dict(requested_model='gpt-6-astra',requested_reasoning_effort='max',fork_turns='none',fork_turns_evidence='parent-dispatched native spawn routing; exact retained request'),
 review_independence='same-family',acceptance_status='provisional',trace_path=trace.as_posix(),full_response=(OUT/'reviewer_full_response.md').as_posix(),full_response_sha256=sha(report_data),
 read_only=dict(scientific_inputs_modified=False,remote_files_written=False,packages_installed=False,processes_restarted=False,model_forwards=0,optimizer_updates=0,official_image_reads=0,private_helper_content_retained=False),
 checks=checks,checklist_mapping={str(i):f'reviewer_full_response.md section {i}' for i in range(1,8)},
 deterministic_coverage=dict(intake_files=29,intake_bytes=72274170,live_remote_intake_matches=29,config_source_fixed_hash_bindings=contracts['hash_bound_checks'],registered_source_batches=780,t0_queue_steps=780,m0_updates=248,q1_updates=1560,total_training_steps=1808,training_distance_elements=replay['training_distance_elements'],ranking_distance_elements=replay['ranking_distance_elements'],query_output_csv_rows=3000,identity_output_csv_rows=300,epoch_log_events=120,endpoint_aggregates=6,epoch_aggregates=120,numeric_aggregate_fields_compared=local['numeric_fields_compared'],max_weighted_scalar_error=replay['max_weighted_scalar_error'],max_fused_triplet_error=replay['max_fused_triplet_error'],max_leaf_norm_distance_geometry_error=replay['max_leaf_norm_distance_geometry_error'],mixed_relevance_tie_queries=sum(x['mixed_relevance_tie_queries'] for x in replay['retrieval'])),
 protocol=replay['protocol'],endpoint_metrics_and_gates=replay['endpoints'],paired_metrics_gates_and_changes=replay['paired'],costs=costs,m0_overfit=contracts['overfit'],training_replay_records=replay['training'],checkpoint_replay=replay['checkpoints'],retrieval_replay_summary=replay['retrieval'],
 execution_provenance={key:provenance[key] for key in ('q1_project_commit','execution_to_q1_commit_diff','actual_signal_head','actual_signal_diff_sha256','execution_commit_changed_paths','local_lf_mismatches')},
 runtime_code_commit=provenance['pipeline']['code_commit'],live_remote_head_at_audit=inventory['head'],live_remote_status_at_audit=inventory['status'],terminal_pipeline_status=provenance['pipeline']['status'],live_original_process_presence=provenance['live_remote_process_presence'],local_watcher_pid31404_present_at_audit=False,
 verification_limits=limitations,
 required_corrections=[
  'Update latest live tracker/handoff/AGENTS to COMPLETE_VERIFIED_Q1_FAIL, 6/6 endpoints, retaining dated history and linking terminal/audit hashes.',
  'Preserve integrity WARN/deterministic PASS separately from scientific Q1_FAIL and retain same-family/provisional requested-routing attribution.',
  'Describe independent saved-array/scalar/norm replay separately from runtime full-model gradient witnesses and single-group M0 direct checks.',
  'Report all failed gate groups, frozen-Signal comparison, fold/role/R1 changes and instrumented costs with seed42/internal-reuse scope.',
  'Keep pending research candidates and unverified novelty claims pending; stronger claims require new evidence.'
 ],
 claims=[dict(id='C1',claim='Complete internally consistent saved Q1 experiment',impact='supported_with_stated_reproduction_limits'),dict(id='C2',claim='Registered scientific qualification passed',impact='unsupported_Q1_FAIL_all_five_paired_and_endpoint_gates_false'),dict(id='C3',claim='Candidate plus 0.672435pp fused mAP versus instrumented control on internal seed42 Q1',impact='supported_with_R1_fold_role_and_Signal_context'),dict(id='C4',claim='Independent full model backward or image-level training/extraction reproduced',impact='unsupported'),dict(id='C5',claim='Formal-test/multi-seed/causal mechanism or future-candidate success',impact='unsupported')],
 audited_input_hashes={k:'sha256:'+v['sha256'] for k,v in sorted(all_hashes.items())},audited_input_details=dict(sorted(all_hashes.items())),
 document_snapshot_bindings=snapshots,
 deterministic_artifacts=[(OUT/n).as_posix() for n in ('intake_check.json','provenance_check.json','remote_inventory.stdout.json','remote_sources.stdout.json','remote_contract_checks.stdout.json','remote_replay.stdout.json','local_validation.json','independent_training_aggregates.json','independent_all3000_query_output_changes.csv','independent_all300_identity_output_changes.csv')],
 environment=dict(local_python=sys.executable,remote_python='/root/miniconda3/envs/tri_reid/bin/python',torch=replay['torch_version'],numpy=replay['numpy_version'],replay_elapsed_seconds=replay['elapsed_seconds'])
)
put(OUT/'audit.json',audit)
(trace/'001-complete-q1.response.md').write_bytes(report_data)
put(trace/'001-complete-q1.meta.json',dict(status='completed',agent_id=audit['agent_id'],requested_model='gpt-6-astra',requested_reasoning_effort='max',fork_turns='none',backend_identity_verified=False,review_independence='same-family',acceptance_status='provisional',completed_at=now,verdict='WARN',scientific_qualification='Q1_FAIL',full_response_sha256=sha(report_data),audit_json_sha256=sha((OUT/'audit.json').read_bytes())))
manifest={}
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name!='audit_artifact_manifest.json' and '__pycache__' not in p.parts:
  data=p.read_bytes();manifest[p.relative_to(OUT).as_posix()]=dict(bytes=len(data),sha256=sha(data))
put(OUT/'audit_artifact_manifest.json',manifest)
final=obj(OUT/'audit.json');assert final['full_response_sha256']==sha((OUT/'reviewer_full_response.md').read_bytes())
assert (trace/'001-complete-q1.response.md').read_bytes()==report_data
assert all(final['deterministic_coverage'][k]==0 for k in ('mixed_relevance_tie_queries',))
print(json.dumps(dict(verdict=final['verdict'],scientific_qualification=final['scientific_qualification'],review_independence=final['review_independence'],acceptance_status=final['acceptance_status'],audited_input_paths=len(all_hashes),artifact_count=len(manifest),report_bytes=len(report_data),report_sha256=sha(report_data),audit_json_sha256=sha((OUT/'audit.json').read_bytes()),document_snapshots_unchanged=all(x['unchanged_since_read'] for x in snapshots)),indent=2))
