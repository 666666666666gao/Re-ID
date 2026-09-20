from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil

OUT=Path(__file__).resolve().parent
REPO=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
e=read(OUT/'AUDIT_EVIDENCE.json');inputs=read(OUT/'audited_input_hashes.json')
agent='/root/audit_cross_scene_smooth_ap_q1_20260921'
generated=datetime.now(timezone.utc).isoformat()
trace=OUT/'trace';trace.mkdir()
request=REPO/'.aris/traces/experiment-audit/2026-09-21_cross_scene_q1/001-q1-review.request.json'
meta=REPO/'.aris/traces/experiment-audit/2026-09-21_cross_scene_q1/001-q1-review.meta.json'
assert read(request)['model']=='gpt-6-astra' and read(request)['reasoning_effort']=='max' and read(request)['fork_turns']=='none'
assert read(meta)['agent_id']==agent and read(meta)['backend_model_attestation'] is None
shutil.copyfile(request,trace/'001-q1-review.request.json')
shutil.copyfile(meta,trace/'parent_invocation.meta.json')
final=OUT/'final_response.md'
text=final.read_text(encoding='utf-8').replace('</D:/','<D:/')
final.write_text(text,encoding='utf-8',newline='\n')
checks={
    'gt_provenance':dict(status='pass',details='All dataset filename labels, complete identity folds, galleries, legal queries, 1950 prerequisite B0 steps and 1560 Q1 source steps verified. No prediction-derived GT.',
        evidence=['tools/build_msvr310_train_oof_protocol.py:12-123','tools/train_msvr310_signal_oof.py:68-99','tools/train_msvr310_trifusion_oof.py:27-55','AUDIT_EVIDENCE.json:33']),
    'score_normalization':dict(status='pass',details='Raw AP/CMC, exact stored feature-to-distance reconstruction and all step Smooth-AP/hard/eligible-anchor arithmetic pass. No prediction-statistic score normalization.',
        evidence=['tools/msvr_cross_scene_smooth_ap.py:11-62','tools/train_msvr310_trifusion_oof.py:196-241','AUDIT_EVIDENCE.json:201','AUDIT_EVIDENCE.json:215']),
    'result_existence':dict(status='warn',details='All 119 original run files unchanged; checkpoint states, results, claims and receipts pass. Mutable tracker and AGENTS still show the earlier running state.',
        evidence=['tools/run_msvr_cross_scene_smooth_ap.py:59-70','refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_TRACKER.md:10-13','AGENTS.md:3-5','AUDIT_EVIDENCE.json:68','AUDIT_EVIDENCE.json:982']),
    'dead_code':dict(status='warn',details='Relevant objectives and evaluation are live. Q1 current/history gradient assertions are saved runtime witnesses; direct comparisons are M0-only, and 203/203 is cumulative coverage.',
        evidence=['tools/train_msvr_cross_scene_smooth_ap.py:129-185','tools/train_msvr_cross_scene_smooth_ap.py:190-264','tools/train_msvr_cross_scene_smooth_ap.py:273-285']),
    'scope':dict(status='warn',details='Complete registered folds/endpoints/queries/identities/outputs, but one seed, reused internal development folds, non-bitwise common-objective warmup and no independent all-step model-gradient replay.',
        evidence=['refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_PLAN.md:24-32','modeling/trifusion/signal_preserving_v13.py:253-279','AUDIT_EVIDENCE.json:587','AUDIT_EVIDENCE.json:658']),
    'eval_type':'real_gt',
    'eval_type_details':dict(status='pass',classification='real_gt',scope='Training-internal identity OOF retrieval; T0 synthetic numerical fixtures separately simulation_only; source objective is a real-label training surrogate, not official mAP.')}
report=dict(audit_skill='experiment-audit',verdict='WARN',overall_verdict='warn',integrity_status='warn',
    closure_status='CLOSED_WITH_LIMITS',deterministic_checks_status='pass',engineering_status='pass',
    scientific_qualification='FAIL',scientific_status='Q1_FAIL',
    reason_code='runtime_witness_limits_single_seed_adaptive_development_and_stale_status_docs',
    summary='Complete fixed Q1 artifacts and independent arithmetic pass. Paired fused gain 0.5671825364 pp and identity bootstrap lower -0.0772393104 pp fail unchanged qualification gates. Runtime and scope limitations retained.',
    date='2026-09-21',generated_at=generated,auditor='gpt-6-astra-max',
    agent_id=agent,verdict_id=agent,executor_model='codex-gpt-6-astra',executor_family='openai',
    reviewer_model='gpt-6-astra',reviewer_family='openai',reviewer_reasoning='max',fork_turns='none',
    review_independence='same-family',acceptance_status='provisional',backend_model_attestation=None,
    model_attribution_basis='Saved native spawn request and metadata; backend identity not independently attested.',
    trace_path=str(trace),parent_trace_path=str(request.parent),
    native_invocation_hashes={str(request):sha(request),str(meta):sha(meta)},
    audited_input_hashes={p:'sha256:'+h for p,h in inputs.items()},audited_input_count=len(inputs),
    checks=checks,scope_counts=e['totals'],maximum_errors=e['maximum_errors'],
    science=dict(paired_gains_pp=e['paired_gains_pp'],paired_fold_gains_pp=e['paired_fold_gains_pp'],
        paired_bootstrap_lower_pp=e['paired_bootstrap_lower_pp'],paired_checks=e['paired_checks'],
        signal_checks=e['endpoint_metrics']['cross_scene']['checks'],
        signal_bootstrap_lower_pp=e['endpoint_metrics']['cross_scene']['bootstrap_lower_pp'],
        next_phase_qualified=False,thresholds_changed=False),
    claims=[
        dict(id='complete_fixed_q1_execution',impact='supported'),
        dict(id='all_saved_distance_mask_objective_ranking_arithmetic',impact='supported'),
        dict(id='stored_terminal_checkpoint_and_frozen_state_content',impact='supported'),
        dict(id='observed_matched_gain',impact='supported_with_single_seed_and_development_scope'),
        dict(id='all_step_parameter_gradient_reconstruction',impact='unsupported'),
        dict(id='every_step_203_nonzero_gradient_tensors',impact='unsupported'),
        dict(id='bitwise_paired_training_trajectory',impact='unsupported_measured_warmup_differences'),
        dict(id='all_role_improvement',impact='unsupported'),
        dict(id='official_SOTA_or_multi_seed_success',impact='unsupported'),
        dict(id='Q1_promotion',impact='unsupported_original_gates_fail')],
    limitations=[
        'Original model forwards, logits, residual distances and all-step parameter-gradient vectors were not regenerated and are not fully recoverable from saved evidence.',
        'Ten of fourteen loss components are saved scalar ledger inputs; cumulative gradient coverage and RNG/buffer assertions remain runtime witnesses.',
        'Q1 contains zero direct full-gradient comparison record-forwards. Prior M0 single-history-group witnesses do not establish Q1 all-step reconstruction.',
        'Stored feature-to-checkpoint provenance is a runtime binding; this audit independently recomputed distances/ranks from features and checkpoint contents without new model forwards.',
        'Common-hard warmup differs numerically despite exact initial/input bindings; no unverified backend cause is asserted.',
        'One seed and repeatedly used internal development folds; identity bootstrap is conditional on these saved models, not seed variability or adaptive-selection correction.',
        'Dataset filenames and historical installation/label receipts were inspected; image pixels and official test directories were not read.',
        'Same-family fresh review is provisional; backend model attestation is unavailable.'],
    action_items=[
        'Archive the complete immutable Q1_FAIL outcome with this audit, all CSV rows, original failed attempts and hash receipts.',
        'Update mutable tracker/AGENTS/handoff after recording audit closure; preserve frozen plan/configuration bytes.',
        'Keep cumulative-versus-every-step gradient, runtime-versus-reconstruction, unequal active-loss definitions and non-bitwise warmup qualifications visible.',
        'Do not promote, tune/retrain the failed registered family, run official tests or launch pre-success ablations on the basis of this result.'],
    integrity_blockers=[],scientific_promotion_blockers=[k for k,v in e['paired_checks'].items() if not v]+['signal:'+k for k,v in e['endpoint_metrics']['cross_scene']['checks'].items() if not v],
    helper_incidents=read(OUT/'AUDIT_ATTEMPTS.json'),
    artifacts=dict(report_md='EXPERIMENT_AUDIT_Q1.md',evidence='AUDIT_EVIDENCE.json',inputs='audited_input_hashes.json',
        attempts='AUDIT_ATTEMPTS.json',final_response='final_response.md'),
    no_training=True,no_optimizer_updates=True,no_official_test_reads=True,no_binary_downloads=True,
    no_project_or_original_evidence_edits=True,no_environment_installs=True)
write(OUT/'EXPERIMENT_AUDIT_Q1.json',report)
write(trace/'run.meta.json',dict(skill='experiment-audit',run_id='2026-09-21_cross_scene_q1',
    executor='codex',executor_model='gpt-6-astra',executor_family='openai',review_independence='same-family',acceptance_status='provisional',
    project_dir=str(REPO),generated_at=generated))
shutil.copyfile(final,trace/'001-q1-review.response.md')
write(trace/'001-q1-review.meta.json',dict(call_number=1,purpose='complete-q1-terminal-review',timestamp=generated,
    agent_id=agent,model='gpt-6-astra',reasoning_effort='max',reviewer_family='openai',review_independence='same-family',
    acceptance_status='provisional',backend_model_attestation=None,status='ok',verdict='WARN'))
assert final.read_bytes()==(trace/'001-q1-review.response.md').read_bytes()
assert len(report['audited_input_hashes'])==448
assert not any(report['science']['paired_checks'][k] for k in ('fused_gain_at_least_1pp','all_role_gains_nonnegative','paired_identity_bootstrap_lower_positive'))
assert sum(report['science']['paired_checks'].values())==2 and sum(report['science']['signal_checks'].values())==1
manifest=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name not in ('manifest.json','AUDIT_RECEIPT.json') and not p.name.startswith('13_write_metadata_and_seal.'):
        manifest.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
write(OUT/'manifest.json',dict(status='SEALED_AUDIT_ARTIFACT_MANIFEST',files=manifest))
write(OUT/'AUDIT_RECEIPT.json',dict(status='COMPLETE_WARN_CLOSED_WITH_LIMITS',generated_at=generated,agent_id=agent,
    integrity='WARN',deterministic_checks='PASS',scientific_qualification='FAIL',original_gate_changes=0,
    source_input_hashes=len(inputs),original_run_files_unchanged=e['original_artifact_files_unchanged'],
    artifact_count=len(manifest),manifest_sha256=sha(OUT/'manifest.json'),
    report_md_sha256=sha(OUT/'EXPERIMENT_AUDIT_Q1.md'),report_json_sha256=sha(OUT/'EXPERIMENT_AUDIT_Q1.json'),
    final_response_sha256=sha(final),trace_response_sha256=sha(trace/'001-q1-review.response.md'),
    reviewer_model='gpt-6-astra',reviewer_reasoning='max',review_independence='same-family',acceptance_status='provisional',
    backend_model_attestation=None,model_forwards=0,optimizer_updates=0,official_test_reads=0))
print(json.dumps(dict(status='COMPLETE_WARN_CLOSED_WITH_LIMITS',output=str(OUT),manifest_files=len(manifest),
    final_response_sha256=sha(final),receipt_sha256=sha(OUT/'AUDIT_RECEIPT.json'))))
