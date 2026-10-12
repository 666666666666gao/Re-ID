"""Write the scoped audit using actual deterministic execution receipts."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
checks = read(HERE / 'CHECKS.json')
chain = read(HERE / 'CHECKS_CHAIN.json')
runs = [read(HERE / f'RUN_{i:02}.json') for i in (1, 2)]
failed_writer = read(HERE / 'WRITE_AUDIT_EXECUTION.json')
assert failed_writer['exit_code'] == 1
assert hashlib.sha256((HERE / 'write_audit_FAILED_01.py').read_bytes()).hexdigest() == failed_writer['script_sha256']
assert checks['status'] == chain['status'] == 'PASS'
assert [r['exit_code'] for r in runs] == [0, 0]
hashes = {**checks['reviewed_hashes'], **chain['reviewed_hashes']}
stability = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()}
assert all(stability.values())
(HERE / 'CHECKS_FINAL.json').write_text(json.dumps(dict(status='PASS', unchanged_reviewed_files=len(stability),
    hashes=hashes, scope='End-of-review local input byte stability only.'), indent=2) + '\n', encoding='utf-8')

def evidence(relative, line):
    return str(BASE / relative) + ':' + str(line)


audit_checks = [
    dict(id='A_ground_truth', status='WARN_PROVENANCE_LIMIT', classification='real_gt_source_path',
         details='The actual invoked evaluator takes identity/camera/scene labels from protocol records. RGBNT201/RGBNT100 remove same-identity same-camera gallery items; MSVR310 removes same-identity same-scene items. AP averages precision at all remaining true-match positions; CMC uses first true-match rank. The pipeline calls the author-source evaluator and asserts agreement within 1e-5. Saved-query counts are 836, 591, 1715 and source-declared full-gallery counts are 836, 1055, 8575. No model-derived GT is used in this path. This auditor did not receive/re-authenticate the raw benchmark annotations, protocol row files or remote distance tensors; real-GT provenance is source-level, not independent raw-data recertification.',
         evidence=[evidence('sealed_source_snapshot911/tools/run_correspondence_roles.py', 79), evidence('sealed_source_snapshot911/tools/run_correspondence_roles.py', 88), evidence('sealed_source_snapshot911/tools/official_three_dataset_data.py', 8), evidence('sealed_source_snapshot911/tools/run_official_three_dataset_roles.py', 58), evidence('sealed_source_snapshot911/tools/train_rgbnt100_signal_oof.py', 253), evidence('sealed_source_snapshot911/tools/train_msvr310_signal_oof.py', 223), evidence('sealed_source_snapshot911/tools/report_prepool_dense_correspondence.py', 18)]),
    dict(id='B_score_normalization', status='PASS_SOURCE_AND_SAVED_QUERY_ARITHMETIC',
         details='No metric is divided by the model output maximum/minimum/mean. AP uses the number of GT positives; macro mAP/CMC divide by legal query count and multiply by 100. Feature L2 normalization belongs to the distance definition, not score inflation. Recomputed all 12 saved-query pair metrics, AP sign counts at the existing 1e-8 threshold, per-identity deltas, repair/harm counts, four AP contribution groups and worst ceil(5%*N) tails. Official float32 CMC differs from integer-count percentages by at most 2.9541660637733003e-06 percentage points, inside the existing source tolerance of 1e-5; this is not a relaxed threshold.',
         evidence=[evidence('sealed_source_snapshot911/tools/run_official_three_dataset_roles.py', 230), evidence('sealed_source_snapshot911/tools/train_rgbnt100_signal_oof.py', 263), evidence('sealed_source_snapshot911/tools/train_msvr310_signal_oof.py', 233), evidence('sealed_source_snapshot911/tools/report_prepool_dense_correspondence.py', 36), evidence('review/remaining911/evidence_snapshot/production/tools/report_region_use_endpoint.py', 74)]),
    dict(id='C_results_and_aggregation', status='PASS_LOCAL_BYTES_AND_RECORDS',
         details='All three manifests match all 59 extracted files (30+11+18); sixth ZIP matches its 2,554,519-byte metadata hash and all 18 members plus manifest. All six summaries agree with original training and strict metric receipts. Each contains epochs1..50, seed42, and the max-mAP/latest-tie selected epoch (8/8, 38/38, 26/26) with that same epoch CMC. All 12,968 step records align with batch-order records; epoch counts/loss means/auxiliary traces are reproduced. Aggregate input hashes, six row projections, all 3,142 uniform–ambiguity aligned query records and all 15 proposed report table rows agree. Total formal epochs=300; original M0 updates are not included in this formal-update total.',
         evidence=[evidence('results908/INTAKE_MANIFEST_908.json', 3), evidence('results911/INTAKE_MANIFEST_911.json', 3), evidence('results912/INTAKE_MANIFEST_912.json', 3), evidence('ALL_SIX_RESULTS_912.json', 4), evidence('ALL_SIX_RESULTS_912.json', 570), evidence('ALL_SIX_RESULTS_912.json', 28955), evidence('SIX_RESULTS_REPORT_913.md', 5), evidence('SIX_RESULTS_REPORT_913.md', 14)]),
    dict(id='D_executed_path_and_exits', status='PASS_SOURCE_AND_CAPTURED_RECEIPTS',
         details='Historical original906 top-level RUNNING is stale and remains unchanged; its supervisor EXIT1 is retained, with exactly five COMPLETE jobs, four successful stages each, and the sixth originally PENDING with zero stages. Those five job results equal their strict receipts; the four earlier908 completed job objects are unchanged. Original remaining911 EXIT1 stopped before evaluate after successful prepare/M0/train. New strict campaign preserves those entire three stage objects and appends only evaluate; all four stage exit records are0. Its command equals original train command with only --mode changed to evaluate, retaining original output/initializer. Captured old/new checkpoint pins and first-strict metrics match original epoch26 selection. Source performs one run_logged evaluate call and no prepare/M0/train replay. New supervisor EXIT0 is 2026-10-13T03:42:03.740534+08:00; stage evaluate is03:40:01.688954 to03:41:54.317090. Exactly one CPU reporter subprocess call is supported by exclusive creation guards, source, one COMPLETE/EXIT0 receipt and one stdout record; five earlier reports are reused. This is acceptance of captured records plus source, not a live remote process attestation.',
         evidence=[evidence('results911/logs/region_use_v1_20261012_906/campaign.json', 3), evidence('results911/logs/region_use_v1_20261012_906/campaign.json', 938), evidence('results911/logs/region_use_resume906/EXIT.json', 12), evidence('TERMINAL_STORAGE_912.json', 14), evidence('TERMINAL_STORAGE_912.json', 613), evidence('TERMINAL_STORAGE_912.json', 714), evidence('remaining_strict912/tools/finish_region_use_evaluation.py', 30), evidence('remaining_strict912/tools/finish_region_use_evaluation.py', 54), evidence('sealed_source_snapshot911/tools/queue_native_research.py', 62), evidence('sealed_source_snapshot911/tools/run_region_use_probe.py', 122), evidence('sealed_source_snapshot911/tools/run_foundation_recipe.py', 287), evidence('results912/logs/region_use_remaining_strict_20261013_912/region_use_remaining_20261013_911/campaign.json', 112), evidence('results912/logs/region_use_remaining_strict_admin912/EXIT.json', 14), evidence('collect_remaining_strict_terminal912.ps1', 20), evidence('collect_remaining_strict_terminal912.ps1', 29), evidence('results912/logs/region_use_remaining_strict_admin912/CPU_REPORT_912.json', 20)]),
    dict(id='E_claim_scope', status='WARN_SINGLE_SEED_DEVELOPMENT_ONLY',
         details='The proposed numerical conclusions are supported as descriptive saved-record comparisons. Ambiguity-minus-uniform mAP/R1 is RGBNT201 -0.158148/+0.119615 (6 repairs,5 new errors), MSVR310 +0.003983/+0 (0/0), RGBNT100 +0.060152/+0.233239 (15/11). No consistent three-dataset deployed-vector gain, significance, unique mechanism, stronger teacher, >Signal goal completion, or full-flow multiseed claim follows. Report explicitly marks ACTIVE_UNMET, consumed development benchmarks and unexecuted same-best decomposition. Independent-global differences do not identify harm to the current same-model global branch. Source assurance is reused only within its prior bounded limitations.',
         evidence=[evidence('ALL_SIX_RESULTS_912.json', 570), evidence('SIX_RESULTS_REPORT_913.md', 31), evidence('SIX_RESULTS_REPORT_913.md', 33), evidence('SIX_RESULTS_REPORT_913.md', 39), evidence('review/remaining_strict912/SOURCE_REVIEW.json', 156)]),
    dict(id='F_evaluation_class', status='WARN_RAW_DATA_NOT_REOBSERVED', evaluation_type='real_gt',
         details='Classification real_gt applies to the inspected scoring/dataflow. Saved per-query AP/ranks and receipts are sufficient for arithmetic and alignment checks, but cannot independently regenerate gallery rankings or authenticate dataset identity labels without the remote raw distance/annotation artifacts. No neural model, torch module or production module was imported/executed by the auditor.',
         evidence=[evidence('sealed_source_snapshot911/tools/run_correspondence_roles.py', 88), evidence('sealed_source_snapshot911/tools/report_prepool_dense_correspondence.py', 25), evidence('results912/INTAKE_MANIFEST_912.json', 23), evidence('results908/INTAKE_MANIFEST_908.json', 36)]),
    dict(id='G_cleanup_and_source_chain', status='WARN_RECEIPT_SCOPE_ONLY',
         details='The cleanup receipt and extracted copy are identical; it records one closed/unprotected historical H2a100 best, a named local archive, protected counts50/6/8 and current_best_deleted0. All six current-best pins match the accepted summaries; retired path is outside the14 additional pins. This does not independently prove remote deletion, existence/content of all archives, or current protected raw binary equality. Controller hash and seal match prior source review and current launch/campaign pins; twelve necessary inherited source file hashes match captured map. All463 source bodies and original50 binaries were deliberately not re-audited. The distinct strict-distance and training-best-distance file hashes do not establish byte identity; only selected-epoch/metric equality is claimed.',
         evidence=[evidence('CLEANUP_912.json', 4), evidence('CLEANUP_912.json', 20), evidence('CLEANUP_912.json', 23), evidence('CLEANUP_912.json', 26), evidence('CLEANUP_912.json', 42), evidence('SIX_RESULTS_REPORT_913.md', 37), evidence('results912/trained-model/region_use_remaining_20261013_911_full_ambiguity_RGBNT100/official_metrics.json', 32), evidence('review/remaining_strict912/SOURCE_REVIEW.json', 5)])
]
warnings = [
    'Actual service model and reasoning effort are UNATTESTED. gpt-6-astra/max are requested routing metadata only; assurance is same-family/provisional.',
    'Historical five original runs were read from saved texts; no original run was newly observed, rerun, or recertified remotely.',
    'Raw remote checkpoints, distances, benchmark annotations/protocol rows, current GPU placement and process absence were not independently observed. Receipt-consistent facts are explicitly separated from those unverified runtime/binary facts.',
    'Complete legal-query counts are verified from saved records; full-gallery contents and GT identities are source/receipt claims, not newly authenticated arrays.',
    'Cleanup/protection and local historical archive intactness are bounded receipt claims only. In particular report line37 must not be presented as a new auditor inspection of all historical archives.',
    'Publication whitespace-failure preservation is stated in report line37 but the original publication failure artifact is outside this supplied result audit. No fabricated verification of that failure is made.',
    'One seed42 per condition with benchmark use in development/epoch selection supports descriptive comparisons only. No stronger teacher, unique failure cause, significance, multiseed or goal-completion promotion.',
]
qualifiers = [
    'Attach the WARN/same-family/provisional audit qualification to publication; describe runtime/GPU/cleanup/protection claims as accepted captured receipts, not independently re-observed raw facts.',
    'For SIX_RESULTS_REPORT_913.md:37, narrow "Local historical archives remain intact" to the named archive/cleanup receipt, or retain an explicit adjacent qualification that broad archive intactness was not independently checked.',
]
actual_checks = dict(status='PASS', main_checks=checks['check_count'], historical_chain_checks=chain['checks'],
    total_primary_checks=checks['check_count'] + chain['checks'], final_unchanged_hashes=len(stability),
    failures=checks['failures'] + chain['failures'], manifests=dict(files=59, groups=[30, 11, 18]),
    saved_pair_query_records=12568, uniform_ambiguity_query_records=3142, strict_summaries=6,
    formal_epochs=300, formal_recorded_updates=12968, proposed_table_rows=15,
    relevant_inherited_source_hashes=12, local_input_hashes=len(hashes), commands_and_exits=runs,
    failed_verifier_invocations=[], tools=[dict(chunk_id='a22c12', exit_code=0), dict(chunk_id='baf0ca', exit_code=0)])
audit = dict(audit_skill='experiment-audit', generated_at=datetime.now().astimezone().isoformat(),
    verdict='WARN', overall_verdict='WARN', integrity_status='warn',
    reason_code='saved_evidence_checks_pass_raw_runtime_provenance_limited',
    summary='29,060 deterministic saved-evidence checks passed with zero mismatches. Six original full50/selected-best records and all reported comparisons are internally consistent. Acceptance is bounded to local bytes, inspected source and captured receipts.',
    agent_id='/root/audit_six_results913', verdict_id='/root/audit_six_results913/result-audit-913',
    reviewer_backend='native-codex', reviewer_requested_model='gpt-6-astra', reviewer_requested_reasoning_effort='max',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', actual_service_model='UNATTESTED', actual_service_effort='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    blockers=[], required_source_or_result_corrections=[], required_publication_qualifiers=qualifiers,
    checks=audit_checks, deterministic_checks=actual_checks, reviewed_hashes=hashes,
    warnings=warnings, bounds=checks['details']['bounds'],
    failed_artifact_writer_invocations=[dict(receipt=failed_writer, source_snapshot=str(HERE / 'write_audit_FAILED_01.py'), error='SyntaxError: unterminated string literal at line30; before execution', correction='Private writer string quoting only; verifier scripts/source/results unchanged', tool_chunk_id='8d33ac')],
    claims=[dict(claim='six original full50/first-strict selected-best numerical records', impact='supported_with_record_provenance_limit'),
            dict(claim='all reported control and uniform/ambiguity numeric contrasts', impact='supported_saved_query_arithmetic'),
            dict(claim='no prepare/M0/full50 replay in strict completion', impact='supported_source_and_captured_stage_objects'),
            dict(claim='new live verification of remote runtime/binaries/cleanup/all historical archives', impact='unsupported_by_this_audit'),
            dict(claim='statistical significance, stronger teacher, causal mechanism, multiple-seed or overall goal success', impact='unsupported_and_not_claimed')],
    artifacts=[str(HERE / name) for name in ('AUDIT.json', 'AUDIT.md', 'CHECKS.json', 'CHECKS_CHAIN.json', 'CHECKS_FINAL.json', 'verify_saved_evidence.py', 'verify_historical_chain.py', 'RUN_01.json', 'RUN_01.log', 'RUN_02.json', 'RUN_02.log')],
    original_input_mutations=0)
(HERE / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

lines = [
    '**Fresh bounded result-integrity audit — WARN**', '',
    'Auditor: native fresh same-family agent `/root/audit_six_results913`; acceptance is **provisional**. Requested model/effort: `gpt-6-astra` / `max`. Actual service model/effort: **UNATTESTED**.', '',
    'There are **zero numeric, source or replay blockers** within the supplied saved-evidence scope. The two actual stdlib verifier invocations returned **EXIT0**, passing **29,060 checks** with no mismatches. This is not a new remote runtime or binary certification. Original report/candidate/scientific result files were not changed.', '',
    f'Evidence paths below are relative to `{BASE}`. Full absolute file:line references and all reviewed SHA256 hashes are in `AUDIT.json`.', '',
    '**Actual deterministic checks**', '',
    '- Three manifests match all **59 extracted files (30+11+18)**. The sixth ZIP matches its **2,554,519-byte** metadata/hash and all18 payload members plus manifest.',
    '- Six full50 seed42 records have the correct mAP-best epoch and the CMC from that same epoch. All **300 epochs /12,968 formal step records** reconcile with batch-order records and auxiliary traces.',
    '- All **12 saved control pairs (12,568 query records)** reproduce AP/CMC, repair/new-error counts, identity deltas, AP contribution groups and worst5% tails. All **3,142 uniform–ambiguity query contrasts** and **15 report table rows** match.',
    '- Historical first-five stage/result receipt links and original failure pins pass34 further checks. Twelve necessary inherited source hashes plus current controller/seal links match; the full463-source audit was not repeated.',
    f'- End-of-review hash stability: **{len(stability)} local reviewed files unchanged**. Official float32 CMC versus integer-query percentages differs by at most **0.0000029541661 percentage points**, within the unchanged existing1e-5 tolerance.', '',
    '| Dataset | Ambiguity−uniform mAP | R1 | Repairs / new errors |',
    '|---|---:|---:|---:|',
    '| RGBNT201 | −0.158148 | +0.119615 | 6 / 5 |',
    '| MSVR310 | +0.003983 | 0 | 0 / 0 |',
    '| RGBNT100 | +0.060152 | +0.233239 | 15 / 11 |', '',
]
for item in audit_checks:
    lines += [f'**{item["id"]}: {item["status"]}**', '', item['details'], '',
              'Evidence: ' + '; '.join('`' + value.replace(str(BASE) + '\\', '') + '`' for value in item['evidence']) + '.', '']
lines += ['**Required publication qualifications; no source/result edits required**', '']
lines += ['- ' + q for q in qualifiers]
lines += ['', '**Remaining provenance limits**', '']
lines += ['- ' + w for w in warnings]
lines += ['', '**Actual invocation record**', '']
for i, run in enumerate(runs, 1):
    lines += [f'{i}. Command: `' + ' '.join('"' + x + '"' for x in run['command']) + f'`. Actual exit: **{run["exit_code"]}**. Start `{run["started_at"]}`; end `{run["completed_at"]}`. Script SHA256 `{run["script_sha256"]}`. Preserved receipt/output: `RUN_{i:02}.json`, `RUN_{i:02}.log`.', '']
lines += [
    'No failed verifier invocation occurred in this review. One private artifact-writer invocation failed before execution with an unterminated-string SyntaxError at line 30 (actual EXIT1; tool chunk 8d33ac). Its original source `write_audit_FAILED_01.py`, output `WRITE_AUDIT.log` and exact command/exit/hash receipt `WRITE_AUDIT_EXECUTION.json` are preserved. Only the private writer string was corrected. Read-only inspection output truncations were not counted as failed verifiers. Original experiment EXIT1 receipts remain unchanged; the historical publication whitespace failure is outside this supplied audit evidence.', '',
    'Artifacts: `AUDIT.json`, `AUDIT.md`, `CHECKS.json`, `CHECKS_CHAIN.json`, `CHECKS_FINAL.json`, `verify_saved_evidence.py`, `verify_historical_chain.py`, and both actual invocation receipts/logs. Zero SSH calls, GPU queries, neural calls, optimizer updates, production imports, source/result mutations or additional agents.', ''
]
(HERE / 'AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(dict(verdict=audit['verdict'], blockers=0, checks=actual_checks['total_primary_checks'],
                     unchanged_hashes=len(stability), actual_exits=[r['exit_code'] for r in runs],
                     audit_json=str(HERE / 'AUDIT.json'), audit_md=str(HERE / 'AUDIT.md'))))
