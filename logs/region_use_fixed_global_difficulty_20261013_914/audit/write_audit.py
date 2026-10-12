"""Write this bounded reviewer verdict and final local-byte receipts only."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

here = Path(__file__).resolve().parent
work = here.parent.parent
checks = json.loads((here / 'CHECKS.json').read_text(encoding='utf-8'))
run = json.loads((here / 'RUN_01.json').read_text(encoding='utf-8'))
assert checks['status'] == 'PASS' and checks['failed_checks'] == 0 and run['exit_code'] == 0
assert hashlib.sha256(Path(run['output_log']).read_bytes()).hexdigest() == run['output_sha256']
assert hashlib.sha256((here / 'verify_fixed_difficulty.py').read_bytes()).hexdigest() == run['script_sha256']
stability = []
for path, digest in checks['reviewed_hashes'].items():
    observed = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    stability.append({'path': path, 'expected_sha256': digest,
                      'observed_sha256': observed, 'unchanged': observed == digest})
assert all(row['unchanged'] for row in stability)

source = work / 'analyze_fixed_global_difficulty914.py'
report = work / 'FIXED_GLOBAL_DIFFICULTY_914.md'
result = work / 'FIXED_GLOBAL_DIFFICULTY_914.json'
aggregate = work / 'ALL_SIX_RESULTS_912.json'
prior = work / 'review/six_results913/AUDIT.json'
findings = [
    {
        'id': 'A_ground_truth_provenance', 'status': 'WARN',
        'details': 'The new analyzer reads saved independent_global per-query identity, AP and first-positive-rank fields. It creates no pseudo-GT. These ranks are GT-dependent post-development descriptors. The prior bounded audit classifies the original scoring path as real_gt but explicitly did not re-authenticate raw benchmark annotations or distances. This review inherits that provenance limit and does not claim a new dataset/raw-ranking certification.',
        'evidence': [f'{source}:32', f'{source}:39', f'{source}:43', f'{prior}:25', f'{prior}:102', f'{report}:3'],
    },
    {
        'id': 'B_score_normalization', 'status': 'PASS',
        'details': 'Within-bin AP differences divide the sum of percentage-point AP changes by the bin count. Contributions to total mAP divide the same numerator by the full dataset query count. There is no prediction maximum/minimum/mean normalization. Independent 50-digit Decimal sums verify both denominators, every mean, and count/N times within-bin delta equals all-query contribution. The four strata form one partition and the two pooled groups form a separate partition; do not add all six rows together.',
        'evidence': [f'{source}:51', f'{source}:58', f'{source}:59', f'{source}:63', f'{source}:64', f'{here / "CHECKS.json"}'],
    },
    {
        'id': 'C_result_existence_and_arithmetic', 'status': 'PASS',
        'details': 'All 11 supplied input artifacts and the request exist and were read as UTF-8. All eight producer input pins match; all six SUMMARY pins match ALL_SIX_RESULTS_912 and the prior audit, and the aggregate hash matches the prior audit. The input-set check plus these 21 hash comparisons comprise 22 pin checks. All 6,284 endpoint query records are valid, with unique contiguous indices; 3,142 dataset-query keys align exactly across conditions. Reconstructed all 12 strata and 6 pooled groups, including identity counts, all AP means/deltas/contributions and every Rank-1 repair/new-error count. All 18 Markdown table rows match exact four-decimal rendering. Recombined totals match the accepted all-six result deltas.',
        'evidence': [f'{result}:4', f'{result}:516', f'{aggregate}:570', f'{aggregate}:28957', f'{prior}:192', f'{report}:7', f'{here / "RUN_01.json"}'],
    },
    {
        'id': 'D_executed_path_and_dead_code', 'status': 'PASS',
        'details': 'Source was read line by line and AST-parsed. Both rank_bin and summarize are called in the result-construction path. The independent verifier imported only standard-library modules and never executed or imported the producer. The verifier actually completed with exit 0. Original producer execution chronology is not separately attested by this bounded audit; support is exact source/result pins plus independent reconstruction.',
        'evidence': [f'{source}:17', f'{source}:47', f'{source}:49', f'{source}:69', f'{source}:71', f'{source}:88', f'{here / "RUN_01.json"}'],
    },
    {
        'id': 'E_scope_and_wording', 'status': 'WARN',
        'details': 'The report correctly says post-development, all original selected-best queries retained, identity counts nonadditive, and within-bin means distinct from global contributions. It excludes train-time margin/weight, GT deployment selection, multiple-seed uncertainty, same-model decomposition, stronger-teacher evidence and a unique failure cause. Fixed independent-global strata describe comparisons among original selected-best models; they cannot attribute a same-model branch effect or establish a training mechanism. Scope remains three datasets and one original seed42 per condition, using development-consumed benchmarks. No wording correction is required.',
        'evidence': [f'{source}:90', f'{source}:91', f'{report}:1', f'{report}:3', f'{aggregate}:28970'],
    },
    {
        'id': 'F_evaluation_type', 'status': 'WARN', 'evaluation_type': 'real_gt',
        'details': 'Saved-query description derived from dataset-GT evaluation records, with inherited source/receipt provenance. Independent-global AP is a comparator, not fabricated ground truth. Dataset annotations, gallery rankings and raw distance tensors were outside this review. Consequently real_gt describes the declared and previously inspected scoring path, not new raw-data authentication.',
        'evidence': [f'{source}:32', f'{source}:39', f'{prior}:25', f'{prior}:102'],
    },
]
limits = [
    'Fresh native same-family review; acceptance provisional. gpt-6-astra/max were requested; actual service model and actual reasoning effort are UNATTESTED.',
    'Only supplied local analyzer, derived JSON/Markdown, accepted all-six aggregate, prior audit record and six saved SUMMARY files were examined. No full earlier 59-file audit replay.',
    'No SSH, network, GPU/NN work, optimizer/training, model or candidate change, producer execution/import, raw-distance or raw-annotation read, or source/result mutation.',
    'GT provenance is inherited from the prior bounded source/receipt audit; no independent authentication of raw benchmark labels, gallery contents, rankings or remote binaries.',
    'One original seed42 per condition; selected-best checkpoints and development-consumed benchmark records. No statistical significance, multiseed uncertainty or untouched-test generalization claim.',
    'Difficulty bins are post-development GT-based fixed independent-global ranks. They are not actual train-time margin/weights, a deployable selector, same-model causal decomposition, stronger-teacher evidence or a unique failure explanation.',
    'Four strata and two pooled groups are alternate partitions of the same queries. Identity counts may overlap across strata and are not additive.',
    'Original official CMC uses float32 percentages. Count-based Rank-1 deltas differ slightly from subtracting those percentages; the existing absolute 1e-5 percentage-point tolerance was retained, not widened.',
]
warnings = [
    'Overall WARN reflects bounded provenance, single-seed/development scope and provisional reviewer assurance; no numerical mismatch or required source/result correction was found.',
    'RGBNT100 global-first-wrong contains 58/1715 queries: the A-minus-U within-bin AP change is about +0.9595 points but its contribution to overall mAP is only about +0.0324 points. This does not establish a deployable hard-query improvement.',
    'Counts, source-inspection facts and saved-file arithmetic are newly checked; historical training/runtime, raw GT and remote-binary facts are not re-observed.',
]
audit = {
    'audit_skill': 'experiment-audit',
    'generated_at': datetime.now().astimezone().isoformat(),
    'verdict': 'WARN', 'overall_verdict': 'WARN', 'integrity_status': 'warn',
    'reason_code': 'bounded_saved_query_provenance_and_scope',
    'summary': '57,422 deterministic checks passed, zero mismatches. Fixed-global strata, pooled groups, all reported arithmetic and the 18-row report are supported as bounded saved-query descriptions.',
    'blockers': [], 'required_corrections': [],
    'agent_id': '/root/audit_fixed_difficulty914',
    'reviewer_backend': 'native Codex agent',
    'reviewer_requested_model': 'gpt-6-astra',
    'reviewer_requested_reasoning_effort': 'max',
    'reviewer_model': 'UNATTESTED', 'reviewer_reasoning': 'UNATTESTED',
    'actual_service_model': 'UNATTESTED', 'actual_service_effort': 'UNATTESTED',
    'reviewer_family': 'openai', 'review_independence': 'same-family',
    'acceptance_status': 'provisional',
    'reviewed_hashes': checks['reviewed_hashes'],
    'actual_checks': {
        'status': checks['status'], 'total_checks': checks['total_checks'],
        'failed_checks': checks['failed_checks'], 'by_category': checks['counts'],
        'aligned_queries': checks['aligned_unique_dataset_query_records'],
        'endpoint_query_records': checks['saved_endpoint_query_records'],
        'strata': 12, 'pooled_groups': 6, 'markdown_table_rows': 18,
        'reconstruction': checks['method'],
        'numeric_tolerances': checks['numeric_tolerances'],
        'max_observed_absolute_errors': checks['max_observed_absolute_errors'],
        'checks_path': str(here / 'CHECKS.json'),
        'command_receipt': run,
    },
    'checks': findings,
    'scope_limits': limits,
    'warnings': warnings,
    'bounded_numeric_totals': checks['recomputed_totals'],
    'failed_invocations': [],
    'failed_invocation_note': 'No failed verifier or writer invocation occurred before this verdict was written. Read-only display calls returned exit 0; output truncation was not recorded as a successful deterministic check. Prior audit failure records were read and left unchanged, not replayed.',
    'bounds': checks['bounds'],
    'trace_path': str(here), 'request_path': str(here / 'REQUEST.md'),
    'artifacts': [str(here / name) for name in (
        'AUDIT.json', 'AUDIT.md', 'CHECKS.json', 'CHECKS_FINAL.json',
        'verify_fixed_difficulty.py', 'RUN_01.json', 'RUN_01.log',
        'write_audit.py', 'WRITE_AUDIT.json', 'WRITE_AUDIT.log')],
}
(here / 'AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
lines = [
    '# Fixed-global difficulty 914: bounded experiment audit', '',
    '**Verdict: WARN. Blockers: 0. Required corrections: none.**', '',
    'All 57,422 deterministic checks passed with zero mismatches. This supports the saved-query description and its report within the limits below.', '',
    'Requested reviewer: `gpt-6-astra`, reasoning `max`. Actual service model and effort: **UNATTESTED**. Native fresh reviewer `/root/audit_fixed_difficulty914`; **same-family / provisional**.', '',
    'The CPU verifier used the supplied Python 3.13 interpreter with `-X utf8 -B`, independent 50-digit Decimal sums and an index-keyed query join. It did not execute/import the producer. Its real command, timestamps, hashes and exit 0 are in `RUN_01.json`; output is in `RUN_01.log`.', '',
    '| Dataset | Aligned queries | Global first wrong | A-U overall mAP points | A-U repairs/new errors |',
    '|---|---:|---:|---:|---:|',
]
for row in checks['recomputed_rows']:
    total = next(t for t in checks['recomputed_totals']
                 if t['dataset'] == row['dataset'] and t['comparison'] == 'ambiguity_minus_uniform')
    lines.append(f"| {row['dataset']} | {row['queries']} | {row['pooled'][1]['queries']} | {float(total['delta_map_points']):+.9f} | {total['rank1_repairs']}/{total['rank1_new_errors']} |")
lines.extend(['', 'The 6,284 endpoint records form 3,142 aligned dataset-query keys. All 12 strata, 6 pooled groups, 162 numeric result fields, 186 exact fields and 18 report rows agree. The new result has eight valid input pins; the six summaries and accepted aggregate also match prior-audit hashes.', '',
              'Within-bin delta = sum of AP changes in percentage points / bin size. All-query contribution = the same sum / total query count. Thus contribution = within-bin delta × bin size / total size. Add the four strata or the two pooled groups, never both partitions together. Rank-1 repair/error counts use the comparator named by each contrast.', '',
              'The maximum derived numeric-field discrepancy is 1.08e-14 points (arithmetic tolerance 1e-10). Integer-count R1 deltas differ from accepted float32 CMC deltas by at most 2.79e-6 points, below the existing 1e-5 bound. No tolerance was widened.', ''])
for finding in findings:
    lines.extend([f"## {finding['id']}: {finding['status']}", '', finding['details'], '',
                  'Evidence: ' + '; '.join(finding['evidence']) + '.', ''])
lines.extend(['## Claim boundaries', ''])
lines.extend('- ' + limit for limit in limits)
lines.extend(['', '## Actual execution and preservation', '',
              '- `CHECKS.json` contains category counts, precise recomputed values, every stratum membership, totals, input hashes and zero failures.',
              '- `RUN_01.json` / `RUN_01.log` preserve the actual deterministic command and exit 0.',
              '- `CHECKS_FINAL.json` records final unchanged hashes for the request and all 11 supplied files plus audit artifact hashes. `WRITE_AUDIT.json` / `.log` record the artifact-writing command and exit.',
              '- No failed invocation occurred in this audit. Prior audit failure artifacts were not modified or rerun.',
              '- Writes are limited to this private review directory; SSH/GPU/NN/optimizer/training/source-result mutation counts are zero.', ''])
(here / 'AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')

parsed = json.loads((here / 'AUDIT.json').read_text(encoding='utf-8'))
required = {'verdict', 'blockers', 'required_corrections', 'reviewed_hashes', 'actual_checks', 'scope_limits'}
assert required.issubset(parsed)
assert parsed['actual_checks']['total_checks'] == 57422
final = {
    'generated_at': datetime.now().astimezone().isoformat(),
    'status': 'PASS', 'input_files_unchanged': len(stability),
    'input_stability_checks': stability,
    'audit_required_keys_present': sorted(required),
    'command_log_and_verifier_hashes_match': True,
    'audit_artifact_hashes': {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest()
        for name in ('AUDIT.json', 'AUDIT.md', 'CHECKS.json', 'RUN_01.json', 'RUN_01.log',
                     'verify_fixed_difficulty.py', 'write_audit.py')
    },
}
(here / 'CHECKS_FINAL.json').write_text(json.dumps(final, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'verdict': parsed['verdict'], 'blockers': parsed['blockers'],
                  'required_corrections': parsed['required_corrections'],
                  'actual_checks': parsed['actual_checks']['total_checks'],
                  'input_files_unchanged': len(stability), 'audit_path': str(here / 'AUDIT.json')}))
