from pathlib import Path
import json,hashlib,math,datetime,subprocess
out=Path(__file__).parent
repo=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
trace=repo/'.aris/traces/experiment-audit/2026-09-20_smooth_ap_source_coverage'
load=lambda p:json.loads(p.read_bytes())
terminal_dir=out/'remote_artifacts/attempt02_terminal'
terminal=load(terminal_dir/'terminal.json')
assert terminal['exit_code']==0
raw=(terminal_dir/'independent_verify.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest()==terminal['stdout_sha256']
assert (terminal_dir/'independent_verify.stderr').read_bytes()==b''
rows=[json.loads(s) for s in raw.decode().splitlines()]
end=rows[-1]
assert end['kind']=='complete' and end['status']=='PASS_INDEPENDENT_ALL_ARRAYS_AND_ROWS'
assert (end['conditions'],end['arrays'],end['output_filter_conditions'],end['full_rows'],end['candidate_rows'])==(12,60,120,82560,1996800)
metrics=[r for r in rows if r['kind']=='condition_check']
geometry=[r for r in rows if r['kind']=='array_geometry']
memory=[r for r in rows if r['kind']=='memory_pool_inputs']
assert len(metrics)==120 and len(geometry)==60 and len(memory)==6
complete=out/'snapshots/repo/evidence/smooth_ap_source_coverage_complete_20260909'
for r in metrics:
    name=r['output']+('_cross_scene' if r['cross_scene'] else '_all_identity')+'_full.json'
    assert hashlib.sha256((complete/r['condition']/name).read_bytes()).hexdigest()==r['full_file_sha256']
assert all(r['full_rows']==82560 for r in [end])
analysis=load(out/'local_analysis_verification.json')
assert analysis['status']=='PASS_COMPLETE_LOCAL_ANALYSIS_AND_TEXT_INTAKE'
provenance=load(out/'provenance_verification.json')
bindings=[json.loads(s) for s in (out/'remote_source_binding.jsonl').read_text().splitlines()]
assert next(r for r in bindings if r['kind']=='recursive_config_binding')['all_match']
signal=next(r for r in bindings if r['kind']=='signal_source_binding')
assert signal['head']==signal['expected_head'] and signal['diff_sha256']==signal['expected_diff_sha256']
assert all(r['actual_sha256']==r['expected_sha256'] for r in signal['checks'])
assert all(r['matches'] for r in bindings if r['kind']=='checkpoint_identity')
stats=dict(complete=end,terminal=terminal,protocol=next(r for r in rows if r['kind']=='protocol'),
    memory_inputs=memory,semantic_fixtures=next(r for r in rows if r['kind']=='semantic_fixtures'),
    direct_formula_full_ap_changed_rows=sum(r['direct_formula_full_ap_changed_queries'] for r in metrics),
    direct_formula_max_ap_delta=max(r['direct_formula_max_ap_delta'] for r in metrics),
    exact_nonself_adjacent_distance_ties=sum(r['exact_nonself_adjacent_distance_ties'] for r in geometry),
    cross_identity_adjacent_ties=sum(r['cross_identity_ties'] for r in geometry),
    direct_vs_registered_order_changed_queries=sum(r['direct_vs_registered_order_changed_queries'] for r in geometry),
    max_feature_unit_norm_error=max(r['norm_max_error'] for r in geometry),
    source_feature_array_bytes=sum((next(c for c in load(complete/'summary.json')['conditions'] if c['directory']==r['condition']))['files'][r['output']+'.npy']['bytes'] for r in geometry),
    remote_full_json_matches_local_intake=120,recursive_project_bindings=99,signal_pinned_source_files=len(signal['checks']),checkpoint_file_hashes=6)
(out/'independent_verification_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
agent='/root/audit_smooth_coverage_resume_20260920'
checks={
 'gt_provenance':dict(status='PASS',details='All 1032 train records match the pinned dataset filename label manifest; 155 identities; source partitions match the registered label-only rule. No model outputs supply identity or scene labels.',evidence=['snapshots/ground_truth/tools/audit_vehicle_query_protocol_labels.py:13-18','snapshots/ground_truth/tools/build_msvr310_train_oof_protocol.py:13-53','provenance_verification.json']),
 'score_normalization':dict(status='PASS',details='Feature L2 normalization defines geometry; AP uses legal positive count, and mAP uses eligible query/exposure denominators. No score is divided by the model own maximum, minimum or mean.',evidence=['snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:38-58','snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:127-131','snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:175-200']),
 'result_existence':dict(status='PASS',details='All 60 arrays, 82560 full rows, 1996800 candidate rows, 120 condition summaries, 40 aggregates, 20 paired summaries and their 180 CSV rows verified. 159 text files match the historical intake manifest; six checkpoints match pinned whole-file hashes.',evidence=['independent_verification_summary.json','local_analysis_verification.json','remote_source_binding.jsonl']),
 'dead_code':dict(status='PASS',details='Source loader -> exact_signal_forward -> output_mapping emits five arrays; analyze invokes rank_row for every full record and candidate exposure. Persisted rows and summaries were independently reconstructed.',evidence=['snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:85-155','snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:158-215','remote_artifacts/attempt02_terminal/independent_verify.jsonl']),
 'scope':dict(status='PASS',details='Accurately scoped fixed terminal source diagnostic: one dataset, seed42, six epoch20 checkpoints, two fixed views, full source galleries. Source membership and repeated anchor weighting are disclosed; neither generalization qualification nor full objective-gradient measurement is claimed.',evidence=['snapshots/repo/refine-logs/msvr310_smooth_ap_source_coverage_v1/DIAGNOSTIC_PLAN.md:3-13','snapshots/repo/results/MSVR310_SMOOTH_AP_SOURCE_COVERAGE_2026-09-09.md','local_analysis_verification.json']),
 'eval_type':dict(status='PASS',classification='real_gt',qualifier='source_only_fixed_state_diagnostic_not_heldout_or_official'),
 'runtime_assurance':dict(status='WARN',details='No images, model forwards or checkpoint tensor-to-feature replay were authorized. Historical state invariance, pixel generation and no-gradient/no-official-read assertions are supported by pinned code and receipts, not independently re-enacted runtime telemetry. Large arrays/checkpoints remain remote; semantic review is same-family/provisional.')
}
limits=[
 'This audit reads saved arrays, records, source and whole checkpoint bytes; it does not independently regenerate features from source image pixels or instantiate/reload the model.',
 'Historical input pixel equality, model-state invariance, absent gradients, zero optimizer updates and zero heldout/official image reads have consistent runtime receipts and code paths; no independent historical syscall/GPU telemetry exists here.',
 'The diagnostic pools use one fixed representation per unique record; original training used repeated sampled positions, changing augmentations/dropout and historical views. Diagnostic AP is not a reconstruction of the dynamic Smooth-AP objective.',
 'All metrics use registered Float64 algebraic squared distances and global-record stable tie order. This does not replace the separately registered FP32 Q1 protocol.',
 'Only one dataset and seed42 are covered. Source records recur in two source folds; member rows and anchor exposures are not independent subjects or new images.',
 'Binary model/NPY/candidate artifacts remain remote; local report, source snapshots, complete row hash receipts and full independent logs are preserved. No portable image-to-feature reproduction bundle is claimed.',
 'The requested reviewer model/effort is gpt-6-astra/max, but only the canonical task name is tool-visible. No backend UUID or independently verified model identity is claimed. Same-family review remains provisional.',
 'Earlier sealed Q1/M0 studies and the separate active objective-gradient experiment were not re-audited; their qualification or promotion status is unchanged.'
]
claims=[
 dict(id='C1',claim='Complete registered source coverage and saved-array ranking arithmetic',impact='supported',evidence='12/12 conditions, 60 arrays, 120 output/filter cells, 82560 full and 1996800 candidate rows.'),
 dict(id='C2',claim='Smooth-AP endpoint improves these fixed source-view full-gallery rankings',impact='supported_with_scope',evidence='All 16 role/fused aggregate AP gains are positive and four baseline gains exactly zero; per-member regressions are retained.'),
 dict(id='C3',claim='Candidate versus full gallery AP gap under common eligible repeated-anchor weighting',impact='supported_with_scope',evidence='Matched fixed feature table and independently reconstructed unique source pools; cannot compare this denominator with full-source member-weighted AP.'),
 dict(id='C4',claim='Dynamic training candidate weighting, score calibration failure, parameter-gradient effect, or unknown-identity generalization',impact='not_supported_and_not_claimed',evidence='Explicitly excluded by registered plan and result report.'),
 dict(id='C5',claim='Historical feature extraction state/pixel invariance independently reproduced in this review',impact='not_claimed',evidence='Pinned source and receipts only; no model/image replay.'),
]
audit=dict(audit_skill='experiment-audit',verdict='WARN',overall_verdict='warn',integrity_status='warn',closure_status='CLOSED_WITH_LIMITS',reason_code='runtime_generation_not_independently_reconstructed',
 summary='Complete saved-array and row arithmetic passes; source-only claims are supported within the frozen protocol. Historical model/pixel generation and same-family provenance limits remain.',
 deterministic_checks_status='pass',engineering_status='PASS_WITH_LIMITS',scientific_qualification='SOURCE_ONLY_DIAGNOSTIC; original Q1_FAIL unchanged',
 agent_id=agent,verdict_id=agent,agent_uuid=None,executor_model=None,executor_family='openai',reviewer_model='gpt-6-astra',reviewer_model_requested='gpt-6-astra',reviewer_reasoning='max',fork_turns='none',
 backend_identity_independently_verified=False,identity_evidence='Parent requested model/effort/fork; collaboration.list_agents returned canonical task name only.',
 reviewer_family='openai',review_independence='same-family',acceptance_status='provisional',trace_path=str(trace),generated_at=now,date='2026-09-20',
 checks=checks,claims=claims,limitations=limits,verification_scope=stats,
 attempt_history=[dict(attempt='R1',exit_code=1,reason='NumPy int64 protocol summary JSON serialization; no array checks completed',receipt='remote_artifacts/failed_attempt01/terminal.json'),dict(attempt='R2',exit_code=0,change='Only three protocol summary counts cast to int',receipt='remote_artifacts/attempt02_terminal/terminal.json')],
 required_scientific_corrections=[],required_bookkeeping=['Replace audit-pending pointers with this CLOSED_WITH_LIMITS report when integrating the audit; preserve the original immutable plan and historical progress records.'],
 prohibited_actions_taken=False,auditor_model_forwards=0,auditor_optimizer_updates=0,auditor_image_reads=0)
hashes={}
for p in sorted((out/'snapshots').rglob('*')):
    if p.is_file():hashes[str(p.relative_to(out))]='sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted((out/'remote_artifacts').rglob('*')):
    if p.is_file():hashes[str(p.relative_to(out))]='sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('local_input_inventory.json','dependency_inventory.json','remote_source_binding.jsonl','provenance_verification.json','local_analysis_verification.json','independent_verification_summary.json','independent_remote_cpu_verify.py','independent_remote_cpu_verify_r2.py','auditor_script_r2_change.json','remote_receipt_discovery.jsonl','initial_environment_errors.md'):
    hashes[name]='sha256:'+hashlib.sha256((out/name).read_bytes()).hexdigest()
audit['audited_input_hashes']=hashes
(out/'EXPERIMENT_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
aggregates=analysis['all40_aggregates'];pairs=analysis['all20_source_pairs']
def agg(endpoint,view,output,cross):return next(r for r in aggregates if (r['endpoint'],r['view'],r['output'],r['cross_scene'])==(endpoint,view,output,cross))
fused_table='\n'.join('| '+view+' / '+('cross-scene' if cross else 'all-identity')+f" | {agg('control',view,'fused',cross)['full_source_map']:.6f} | {agg('smooth_ap',view,'fused',cross)['full_source_map']:.6f} | {agg('control',view,'fused',cross)['full_eligible_members']} |" for view in ('clean','augmented') for cross in (False,True))
paired_table='\n'.join(f"| {r['view']} | {r['output']} | {'cross-scene' if r['cross_scene'] else 'all-identity'} | {r['ap_gain_pp']:+.6f} | {r['better']} | {r['worse']} |" for r in pairs)
candidate_table='\n'.join(f"| {endpoint} | {'cross-scene' if cross else 'all-identity'} | {agg(endpoint,'augmented','fused',cross)['common_candidate_map']:.6f} | {agg(endpoint,'augmented','fused',cross)['common_full_map']:.6f} | {agg(endpoint,'augmented','fused',cross)['common_eligible_exposures']} |" for endpoint in ('control','smooth_ap') for cross in (False,True))
checks_md='\n\n'.join(f"### {letter}. {label}: {checks[key]['status']}\n\n{checks[key]['details']}\n\nEvidence: "+'; '.join('`'+p+'`' for p in checks[key]['evidence']) for letter,label,key in [('A','Ground truth provenance','gt_provenance'),('B','Score normalization','score_normalization'),('C','Result existence and arithmetic','result_existence'),('D','Executed metric path','dead_code'),('E','Scope assessment','scope')])
report=f'''# Experiment audit: MSVR310 fixed-source Smooth-AP coverage

Date: 2026-09-20. Overall verdict: **WARN / CLOSED_WITH_LIMITS**. Deterministic saved-array and result checks: **PASS**. Engineering: **PASS_WITH_LIMITS**. Review independence: **same-family**; acceptance: **provisional**.

The registered source diagnostic is complete and its source-ranking/candidate-coverage claims match all saved results. No scientific code, metric, denominator, masking or reported-number correction is required. WARN records the actual provenance limits below; it is not a failure of the reconstructed ranking arithmetic and does not promote the sealed Q1 result.

Requested reviewer: `gpt-6-astra`, effort `max`, fresh task (`fork_turns=none`). Actual tool-visible identity: `{agent}`. No UUID or independently verified backend identity is available. The prior auditor left no verdict or complete independent receipt. Its script was inspected and reused as an incomplete independent input, not as an accepted conclusion.

## Completed scope and receipts

- Fixed six epoch20 checkpoints, 3 folds x control/Smooth-AP x clean/one seed42 augmentation: **12 conditions**, original **8,256 source triplet-record forwards**, **0 optimizer updates**.
- Source galleries contain **672 / 683 / 709** records, with **103 / 103 / 104** identities and **390 / 393 / 417** cross-scene-eligible records. The unique dataset has **1,032 train records / 155 identities**. Each record occurs in two source folds.
- Independently checked **60 Float32 feature arrays / 41,280 array rows**, **82,560 complete-gallery query rows**, **1,996,800 candidate/full paired anchor exposures**, and **120 output/filter conditions**.
- Independently reconstructed all six **260-step memory ledgers**, current-record deduplication, last-occurrence LRU order, age<=8, capacity512, warmup65, 64 positions and eight identities x eight positions per batch. Every selected record belongs to the corresponding source fold; matched endpoints have identical record/pixel/pool ledgers.
- Recomputed all **40 aggregates**, **20 full-source endpoint pairs** and **180 CSV rows**. All **159 original text files / 13,715,377 bytes** match the saved intake manifest; all 120 current remote full-query files also match those local bytes.
- Current bytes match **99 recursive project/config bindings**, **{len(signal['checks'])} pinned Signal source files**, the pinned Signal commit and diff hash, and **six whole-checkpoint hashes**. Five local-vs-remote source differences are exclusively CRLF vs LF with identical ASTs; both versions are preserved.
- R2 CPU checker `{terminal['child_pid']}` (wrapper `{terminal['wrapper_pid']}`) exited **0** at `{terminal['ended_at']}`; elapsed **{end['elapsed_seconds']:.3f} s**, peak RSS **{end['max_rss_kib']} KiB**. It imported no Torch, initialized no CUDA, read no images and performed no model forwards or updates. Only audit scripts/logs/receipts were written to the separate remote audit directory.
- Maximum AP reconstruction error **{end['max_ap_error']:.17g}**; maximum direct-difference vs registered algebraic distance error **{end['max_distance_error']:.17g}**; maximum aggregate rounding difference **{analysis['max_numeric_error']:.17g}**.

Complete evidence: `remote_artifacts/attempt02_terminal/independent_verify.jsonl`, `terminal.json`, `intake.json`; `independent_verification_summary.json`; `local_analysis_verification.json`; `provenance_verification.json`; `remote_source_binding.jsonl`. Input inventories and all SHA256 values are in `EXPERIMENT_AUDIT.json`, `local_input_inventory.json` and `dependency_inventory.json`.

## Checks A-F

{checks_md}

### F. Evaluation type: PASS — real_gt, source-only diagnostic

Identity and scene labels come from dataset filename metadata, consistent with the pinned MSVR310 loader (`snapshots/signal_msvr310.py:73-97`) and the archived label manifest. The source embeddings are model outputs; they are not ground truth. This diagnostic uses the source complement of each fold, even though the reused registry is named OOF. It is not a heldout/official evaluation. The custom metric preserves the upstream same-ID AND same-scene exclusion and AP denominator (`snapshots/signal_metrics.py:65-107`), while adding separately registered all-identity/source-self and stable Float64 rules.

## Metric semantics and weighting

`rank_row` excludes the anchor's global record, never merely one duplicated batch position. Candidate pools are the sorted union of current and memory record indices, deduplicated once. In cross-scene mode, same-ID/same-scene records are removed rather than relabeled as negatives. Every different-identity gallery record remains a distractor, including single-scene identities with no valid query positive. Ineligible queries have AP/Rank1 `null`; their records remain in other queries' galleries. AP is the mean of precision at every legal positive's rank; Rank1 is the first remaining hit. `inverted_positives` counts positive positions with at least one strictly closer negative, not the number of all inversion pairs.

All ranks use the registered Float64 algebraic squared-distance matrix and a stable ascending global-record tie break. Independent direct-difference geometry was checked for every array. Across array-query rows, **{stats['exact_nonself_adjacent_distance_ties']}** exact adjacent non-self distance ties were observed, including **{stats['cross_identity_adjacent_ties']}** cross-identity adjacent ties. Direct-formula order changed in **{stats['direct_vs_registered_order_changed_queries']}** array-query rows; complete-gallery AP changed in **{stats['direct_formula_full_ap_changed_rows']}** output/filter/query rows, with maximum AP delta **{stats['direct_formula_max_ap_delta']:.17g}**. This sensitivity check does not redefine the registered ordering.

Full-source mAP weights each eligible source member once within each fold, then pools AP values without mixing embedding coordinate systems. Candidate-vs-full mAP weights the original repeated anchor positions and uses only positions eligible in both pools. Each endpoint/view has 49,920 positions: 49,872 common all-identity positions (48 unique-pool missing-positive cases) or 20,496 common cross-scene positions (168 full-positive/pool-missing cases). Those 48 cases do not mean the original K8 batches lacked training positives: training retained distinct sampled positions and random views. Cross-table differences are not candidate effects because the weighting differs.

## Reported results, fully retained

Member-weighted fused full-source mAP:

| View / positive rule | Control | Smooth-AP | Eligible source members per endpoint |
|---|---:|---:|---:|
{fused_table}

Augmented fused candidate/full mAP on common eligible repeated positions:

| Endpoint | Positive rule | Candidate | Full source | Common exposures |
|---|---|---:|---:|---:|
{candidate_table}

All 20 source endpoint comparisons (AP gain in percentage points; no branch/view selection). Improved/worsened members require AP delta greater than +1e-12 / less than -1e-12; other members are unchanged at this tolerance:

| View | Output | Positive rule | AP gain | Improved members | Worsened members |
|---|---|---|---:|---:|---:|
{paired_table}

The highlighted augmented cross-scene fused gain is **+1.228427 pp**, 95.444495 -> 96.672922, with Rank1 96.583333 -> 97.916667. Its 1,200 source members contain 251 improvements, 70 regressions and 879 unchanged AP values at the stated tolerance; strictly inverted positive positions decrease 1,214 -> 1,002. The original report retains these adverse rows and explicitly denies a generalization or calibration-failure conclusion. All 16 non-baseline aggregate AP gains are positive, while all four baseline comparisons are exactly unchanged; this remains one fixed source study, not independent replication.

## Actual execution chain and provenance limits

The inspected chain is coverage `contract/extract` -> Smooth-AP `context` -> role-set -> history-gradient -> fresh-coordinate -> instance-memory -> source-style configuration context -> Signal configuration. The coverage extractor actually uses the ordinary role `build_model/reload_model`, not the optional V27 source-style wrapper. Source records use `records_for(..., True)`; `source_loader` creates a sequential complete source loader; `SharedGeometryTripletTransform` applies common triplet geometry and independent erasing for the augmented view. Seed42 is reset for each endpoint/view. Signal's `train_collate_fn` preserves filenames and camera/scene metadata. `_training_batch` passes images, all-present modality masks and camera IDs to the model, not identity GT for ranking generation.

The actual inference route is `exact_signal_forward` -> `torch.func.functional_call` under `no_grad` -> V8 hierarchical frozen Signal + three role residual paths -> output mapping -> per-output feature L2 normalization. The temporary detached SIM dispatch view does not update its registered parameter. Strict reload binds fold/source IDs/config SHA/role state/frozen baseline aliases, and extraction asserts full model state SHA unchanged, absent gradients and paired pixel/baseline equality. Those historical assertions are consistent with every receipt and the inspected source; they were **not independently re-executed** in this CPU-only audit. Architecture/source snapshots are preserved, including the external Signal loader/model files; compiled extensions and GPU kernels were not independently inspected or replayed.

The original math/extract/analyze pipeline and executor verifier have complete exit0 receipts. The extractor's 8,256 forwards are original runtime cost, not new work by this reviewer. The saved arrays consume **{stats['source_feature_array_bytes']:,} bytes including NPY headers**. The original models, arrays and large candidate records remain on the server. The historical result report's gradient-preflight progress paragraph is outside this audit's subject and is not treated as a current statement about the separate running gradient task.

{chr(10).join('- '+x for x in limits)}

## Attempts, corrections and claim impact

R1 copied the prior auditor's unfinished checker exactly (SHA `a2d580137b449ebe0f0d5d28bd3640b87080a7e538a3c08c066c0d721edddb69`). It exited1 before any array check because a protocol count was a NumPy int64 that JSON could not serialize. The complete original script, one-line stdout, traceback and terminal receipt are preserved in `remote_artifacts/failed_attempt01`. After confirming both processes had ended, R2 changed only three summary counts to Python int (SHA `{terminal['script_sha256']}`), preserving all scientific assertions and arithmetic. R2 then completed the full requested verification once. Local runtime discovery/path-search errors are recorded in `initial_environment_errors.md`; no environment, experiment source or scientific result was changed.

No scientific correction is required. When integrating this report, replace audit-pending status pointers with **CLOSED_WITH_LIMITS / deterministic PASS / same-family provisional** and link this report, retaining the immutable pre-run plan and historical progress entries. Do not describe this as cross-family acceptance, official performance, generalization qualification, a replay of training's dynamic objective, or parameter-gradient verification. Do not reopen sealed Q1/M0 or authorize training based on this diagnostic.

No commit, push, master-document edit, model/GPU forward, new training, image read or weight deletion was performed by this audit. The existing `.aris/meta/events.jsonl` was already modified by other work and was left untouched; the review event is saved in this audit's private trace directory instead.
'''
(out/'EXPERIMENT_AUDIT.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(status='REPORTS_WRITTEN',verdict=audit['verdict'],closure_status=audit['closure_status'],report=str(out/'EXPERIMENT_AUDIT.md'),json=str(out/'EXPERIMENT_AUDIT.json'),independent_summary=stats),ensure_ascii=False,indent=2))
