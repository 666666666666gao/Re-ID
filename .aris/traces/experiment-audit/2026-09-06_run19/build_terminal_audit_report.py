#!/usr/bin/env python3
import datetime
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN19 = ROOT / '.aris/traces/experiment-audit/2026-09-06_run19'
RESULT = json.loads((RUN19 / 'terminal_audit_replay_result.json').read_text(encoding='utf-8'))
COMPARISON = json.loads((ROOT / 'evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json').read_text(encoding='utf-8'))
TERMINAL_FILES = json.loads((ROOT / 'evidence/trifusion_msvr310_trifusion_v1_terminal_files_verification_20260906.json').read_text(encoding='utf-8'))
ERROR_CENSUS = json.loads((ROOT / 'evidence/trifusion_msvr310_trifusion_v1_complete_error_census_20260906.json').read_text(encoding='utf-8'))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def file_info(path: Path):
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}

now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
query = RESULT['query_metric_replay']
training = RESULT['training_scalar_replay']
protocol = RESULT['protocol_replay']
receipts = RESULT['receipt_checks']
error = RESULT['error_census_replay']
comparison_replay = query['comparison_replay']
metrics = comparison_replay['metrics']
gains = comparison_replay['gains_over_signal_pp']
folds = query['fold_summaries']
identity_rows = comparison_replay['per_identity']
array_rows = TERMINAL_FILES['array_checks']

replay_artifacts = {
    'script': file_info(RUN19 / 'replay_msvr310_terminal_audit.py'),
    'report_builder': file_info(RUN19 / 'build_terminal_audit_report.py'),
    'stdout': file_info(RUN19 / 'terminal_audit_replay_stdout.txt'),
    'result': file_info(RUN19 / 'terminal_audit_replay_result.json'),
}
for name, info in RESULT['output_artifacts'].items():
    replay_artifacts[name] = info

line_refs = {
    'scene_ap_rank': 'tools/train_msvr310_signal_oof.py:223-238',
    'trifusion_eval': 'tools/train_msvr310_trifusion_oof.py:196-241',
    'comparison_summary': 'tools/train_msvr310_trifusion_oof.py:244-285',
    'run_binding': 'tools/train_msvr310_trifusion_oof.py:288-413',
    'resume_reuse': 'tools/resume_msvr310_trifusion_exact_inference.py:82-109',
    'resume_new_folds': 'tools/resume_msvr310_trifusion_exact_inference.py:111-170',
    'exact_helper': 'tools/msvr310_exact_signal_inference.py:4-22',
    'exact_verifier': 'tools/verify_msvr310_exact_signal_inference.py:41-112',
    'loss_formula': 'tools/run_signal_preserving_v5.py:99-142',
    'lr_formula': 'tools/run_signal_preserving_v5.py:1621-1628',
    'identity_bootstrap': 'modeling/trifusion/signal_preserving_v13.py:253-280',
    'dense_outputs': 'modeling/trifusion/signal_preserving_v8.py:492-516,620-666,669-676',
    'pytorch_linear_algebra': 'evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2111-2118,2157-2166',
    'pytorch_linear': 'evidence/msvr310_pytorch251_Linear.cpp.txt:180-231,820-824',
    'pytorch_mha': 'evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:162-175,621-639,667-678',
    'failure_receipt': 'evidence/trifusion_msvr310_trifusion_v1_comparison_failure_receipt_20260906.json:3-8,52',
    'failed_log': 'evidence/trifusion_msvr310_trifusion_v1_comparison_failed_run_20260906.log:145-166',
    'comparison_complete': 'evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json:1-22,94826-94894,95557-95589',
    'exact_verification_json': 'evidence/trifusion_msvr310_trifusion_v1_exact_signal_inference_verification_20260906.json:3,11-19,73-80',
    'sim_diagnosis_json': 'evidence/trifusion_msvr310_trifusion_v1_sim_operation_parity_diagnosis_20260906.json:76-87,89-143',
    'baseline_parity_json': 'evidence/trifusion_msvr310_trifusion_v1_baseline_parity_diagnosis_20260906.json:1-70,522-535',
    'terminal_files_json': 'evidence/trifusion_msvr310_trifusion_v1_terminal_files_verification_20260906.json:1-60,150-180,360-395',
    'r3_wrapper': 'evidence/trifusion_msvr310_trifusion_v1_resume_r3_wrapper_20260906.py:9-24',
    'r3_launch': 'evidence/trifusion_msvr310_trifusion_v1_resume_r3_launch_20260906.json:1-12',
    'r3_log': 'evidence/trifusion_msvr310_trifusion_v1_resume_r3_run_20260906.log:1-80',
    'source_binding': 'evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json:1-86',
    'signal_b0': 'evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json:4,181765-181793',
    'm0': 'evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:4,31,3480-3492,12810-12812',
    'error_census': 'evidence/trifusion_msvr310_trifusion_v1_complete_error_census_20260906.json:1-67,45981',
    'terminal_request': '.aris/traces/experiment-audit/2026-09-06_run19/001-terminal.request.json:2,7-8',
    'replay_result': '.aris/traces/experiment-audit/2026-09-06_run19/terminal_audit_replay_result.json:1-1274',
}

verdicts = {
    'integrity_qualification': 'PASS_WITH_LIMITS',
    'engineering_qualification': 'PASS',
    'scientific_qualification': 'FAIL',
    'overall': 'Engineering evidence supports the completed original three-role internal comparison, but the scientific support gate fails.',
    'a_f_categories': [
        {'category': 'A', 'name': 'Protocol, labels, and data isolation', 'verdict': 'PASS', 'basis': '600 valid internal queries, 60 query identities, 1032 gallery records, source/heldout disjoint per fold, and no official test/dev access.'},
        {'category': 'B', 'name': 'Metric and score computation', 'verdict': 'PASS', 'basis': 'AP/Rank recomputed from categorical labels after same identity/same scene filtering and complete saved rankings.'},
        {'category': 'C', 'name': 'Independent replay and number/file agreement', 'verdict': 'PASS', 'basis': 'Independent stdlib/NumPy replay matched aggregate/fold metrics, query AP/Rank, bootstrap, loss/epoch/LR, protocol counts, and error census; manifest pre/post hash checks passed.'},
        {'category': 'D', 'name': 'Execution boundary, reuse, and source binding', 'verdict': 'PASS', 'basis': 'Original fold0 failure and R3 continuation are separated; fold0 training/verified features reused, folds1/2 trained once, no metric-selection retraining found in text receipts.'},
        {'category': 'E', 'name': 'Scope and claim qualification', 'verdict': 'WARN', 'basis': 'Evidence is complete for the declared internal OOF comparison, but not an official MSVR310 591/1055 test result; raw tensor/checkpoint binaries are represented by remote receipts only.'},
        {'category': 'F', 'name': 'Scientific support', 'verdict': 'FAIL', 'basis': 'All five fixed scientific conditions are false; fused mAP is 1.011990 pp below Signal baseline and the identity bootstrap lower bound is negative.'},
    ],
}

md = []
md.append('# Experiment Audit: MSVR310 TriFusion Original Three-Role Terminal Comparison')
md.append('')
md.append(f'Generated: {now} (Asia/Shanghai).')
md.append('')
md.append('Verdict: **integrity PASS with limits, engineering PASS, scientific FAIL**. The completed comparison is internally reproducible from the provided text/JSON evidence and timestamped receipts. It does not support the original three-role method scientifically under its own fixed gates.')
md.append('')
md.append('## A-F verdicts')
md.append('')
md.append('| Category | Verdict | Meaning |')
md.append('|---|---:|---|')
for row in verdicts['a_f_categories']:
    md.append(f"| {row['category']} - {row['name']} | {row['verdict']} | {row['basis']} |")
md.append('')
md.append('## Audit boundary')
md.append('')
md.append('- I used the `experiment-audit` checklist directly as the terminal reviewer, but did not delegate further because the task explicitly forbids further delegation.')
md.append(f'- The immutable input manifest hash matched the requested value: `{RESULT["manifest"]["sha256"]}`. Pre and post hash checks covered {RESULT["manifest"]["pre_hashcheck"]["file_count"]} files and {RESULT["manifest"]["pre_hashcheck"]["manifest_bytes_total"]:,} bytes; both reported no mismatches.')
md.append(f'- Execution was local text/JSON/stdlib/NumPy only. Python: `{RESULT["python_executable"]}`. NumPy path: `{RESULT["numpy_path"]}`. NumPy version: `{RESULT["numpy_version"]}`.')
md.append('- I did not use torch, model runtime, tensor/image libraries, remote commands, network, browser, package installation, or further delegation.')
md.append('- Signal B0 and original M0 were treated as separately audited prerequisites. I inspected their commit/config/protocol/checkpoint/initialization bindings only and did not repeat their prior 1950/124-update numerical audits.')
md.append('- The package does not contain raw `.pt`/`.pth` tensor/checkpoint binaries. Stored-array and checkpoint parity beyond text/JSON rankings is therefore receipt-audited against terminal remote hashes and shapes, not locally reloaded from binary tensors.')
md.append('- `.aris/traces/experiment-audit/2026-09-06_run19/dispatch_observation.json` was not present. The request metadata shows requested model/reasoning, but that is not independent backend attestation.')
md.append('')
md.append('## Replay command and artifacts')
md.append('')
md.append('Complete stdout, command line, start/end timestamps, and exit code are preserved in `.aris/traces/experiment-audit/2026-09-06_run19/terminal_audit_replay_stdout.txt`.')
md.append('')
md.append('```powershell')
md.append("& 'C:\\Users\\gb\\AppData\\Roaming\\uv\\python\\cpython-3.13-windows-x86_64-none\\python.exe' '.aris\\traces\\experiment-audit\\2026-09-06_run19\\replay_msvr310_terminal_audit.py' --manifest '.aris\\traces\\experiment-audit\\2026-09-06_run19\\input_manifest.json' --output '.aris\\traces\\experiment-audit\\2026-09-06_run19\\terminal_audit_replay_result.json' --query-output '.aris\\traces\\experiment-audit\\2026-09-06_run19\\terminal_audit_query_outputs.jsonl' --training-output '.aris\\traces\\experiment-audit\\2026-09-06_run19\\terminal_audit_training_steps.jsonl' --epoch-output '.aris\\traces\\experiment-audit\\2026-09-06_run19\\terminal_audit_epoch_means.jsonl' --hash-pre-output '.aris\\traces\\experiment-audit\\2026-09-06_run19\\terminal_audit_hashcheck_pre.jsonl' --hash-post-output '.aris\\traces\\experiment-audit\\2026-09-06_run19\\terminal_audit_hashcheck_post.jsonl'")
md.append('```')
md.append('')
md.append('| Artifact | Bytes | SHA-256 |')
md.append('|---|---:|---|')
for name, info in replay_artifacts.items():
    md.append(f"| `{info['path']}` | {info['bytes']} | `{info['sha256']}` |")
md.append('')
md.append('## Independent metric replay')
md.append('')
md.append(f'- Recomputed query-output evaluations: {query["query_output_evaluations"]} = 600 queries x 5 outputs.')
md.append(f'- Distinct query identities: {query["query_identities"]}.')
md.append(f'- Max aggregate metric absolute difference to comparison summary: {query["max_aggregate_metric_abs_diff_to_summary"]}.')
md.append(f'- Max fold metric absolute difference to fold receipts: {query["max_fold_metric_abs_diff_to_receipts"]}.')
md.append(f'- Max AP absolute difference to receipt query AP lists: {query["max_average_precision_abs_diff_to_receipts"]}.')
md.append(f'- Rank mismatch count upper bound from replayed first positive rank: {query["rank_mismatch_count_upper_bound"]}.')
md.append(f'- Bootstrap lower-bound difference to summary: {query["bootstrap_lower_abs_diff_to_summary"]}.')
md.append('')
md.append('Metric implementation basis: `scene_scores` sorts distances, removes gallery records with same identity and same scene as the query, computes AP from positive positions, and records first positive rank (`tools/train_msvr310_signal_oof.py:223-238`). TriFusion evaluation normalizes features, computes pairwise distances, verifies baseline-only distances against B0, writes complete rankings, and calls the same scene-filtered metric (`tools/train_msvr310_trifusion_oof.py:196-241`).')
md.append('')
md.append('| Output | mAP | Rank-1 | Rank-5 | Rank-10 | Gain vs Signal mAP pp |')
md.append('|---|---:|---:|---:|---:|---:|')
for name in ['baseline_only','fused','cnn','transformer','mamba']:
    m = metrics[name]
    md.append(f"| {name} | {m['mAP']:.12f} | {m['Rank-1']:.12f} | {m['Rank-5']:.12f} | {m['Rank-10']:.12f} | {gains[name]:.12f} |")
md.append('')
md.append('| Fold | Queries | Query identities | Gallery | Signal mAP | Fused mAP | Fused gain pp |')
md.append('|---:|---:|---:|---:|---:|---:|---:|')
for f, fg in zip(folds, comparison_replay['fold_fused_gains_pp']):
    md.append(f"| {f['fold']} | {f['queries']} | {f['query_identities']} | {f['gallery_records']} | {f['metrics']['baseline_only']['mAP']:.12f} | {f['metrics']['fused']['mAP']:.12f} | {fg:.12f} |")
md.append('')
md.append('Query changes versus baseline-only:')
md.append('')
md.append('| Output | AP improved | AP declined | AP unchanged | Rank-1 repaired | Rank-1 new errors |')
md.append('|---|---:|---:|---:|---:|---:|')
for name in ['fused','cnn','transformer','mamba']:
    q = comparison_replay['query_changes'][name]
    md.append(f"| {name} | {q['ap_improved']} | {q['ap_declined']} | {q['ap_unchanged']} | {q['rank1_repaired']} | {q['rank1_new_errors']} |")
md.append('')
bs = comparison_replay['identity_bootstrap']
md.append(f"Identity bootstrap: lower bound {bs['lower_bound_pp']:.12f} pp, seed {bs['seed']}, resamples {bs['resamples']}, clusters {bs['cluster_count']}, percentile {bs['percentile']}, method `{bs['quantile_method']}`. The implementation resamples whole identities while retaining query weights (`modeling/trifusion/signal_preserving_v13.py:253-280`).")
md.append('')
md.append('Scientific checks from independent replay:')
md.append('')
for k, v in comparison_replay['scientific_checks'].items():
    md.append(f'- `{k}`: {v}')
md.append('')
md.append('## Protocol, isolation, and scope')
md.append('')
md.append(f'- Aggregate protocol replay: {protocol["valid_queries"]} valid queries, {protocol["unique_query_identities"]} query identities, {protocol["unique_gallery_records"]} gallery records, {protocol["excluded_query_records_retained_in_gallery"]} excluded query records retained in gallery, {protocol["gallery_only_distractor_identities"]} gallery-only distractor identities.')
md.append(f'- All protocol paths came from the training split: {protocol["all_protocol_paths_from_training_split"]}. Official test paths seen: {protocol["official_test_paths_seen"]}.')
md.append('- Fold isolation replay:')
md.append('')
md.append('| Fold | Source IDs | Source records | Heldout IDs | Gallery | Queries | Query IDs | Excluded query records | Source/Heldout disjoint |')
md.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for f in protocol['fold_summaries']:
    md.append(f"| {f['fold']} | {f['source_identities']} | {f['source_records']} | {f['heldout_identities']} | {f['gallery_records']} | {f['query_rows']} | {f['query_identities']} | {f['excluded_query_records']} | {f['source_heldout_disjoint']} |")
md.append('')
md.append('This supports only the declared internal train-split OOF comparison. It is not the official MSVR310 591-query/1055-gallery evaluation, and the timestamped report itself says the comparison is internal and no RGBNT201 dev/official test access was used (`results/TRIFUSION_MSVR310_ORIGINAL_ROLES_COMPARISON_20260906_085329.md:10-12`).')
md.append('')
md.append('## Training scalar replay')
md.append('')
md.append(f'- Replayed optimizer steps: {training["optimizer_steps"]}; summary optimizer-step agreement: {training["matches_summary_optimizer_steps"]}.')
md.append(f'- Epoch rows replayed: {training["epochs"]}; max epoch mean-loss difference: {training["max_epoch_mean_abs_diff"]}; max LR difference: {training["max_learning_rate_abs_diff"]}.')
md.append(f'- Source record exposures: {training["source_record_exposures"]}; same-identity positive pairs: {training["same_identity_positive_pairs"]}; cross-scene positive pairs: {training["cross_scene_positive_pairs"]}.')
md.append(f'- Max weighted loss recomposition difference from serialized components: {training["max_weighted_loss_abs_diff"]}. The max row was below 1e-6; no count, B64/K8, epoch mean, LR, source-membership, or fold total mismatch was found.')
md.append('')
md.append('Loss composition was independently recomputed from the stored scalar parts using the source formula (`tools/run_signal_preserving_v5.py:99-142`). Learning rate was independently recomputed from warmup/cosine source logic (`tools/run_signal_preserving_v5.py:1621-1628`).')
md.append('')
md.append('| Fold | Optimizer steps | Epochs | Source exposures | Unique source records exposed | Source IDs exposed | Same-ID pairs | Cross-scene pairs |')
md.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
for f in training['fold_summaries']:
    md.append(f"| {f['fold']} | {f['optimizer_steps']} | {f['epochs']} | {f['source_record_exposures']} | {f['unique_source_records_exposed']} | {f['source_identities_exposed']} | {f['same_identity_positive_pairs']} | {f['cross_scene_positive_pairs']} |")
md.append('')
md.append('Full per-step replay is in `terminal_audit_training_steps.jsonl` (780 rows). Full per-epoch mean/LR replay is in `terminal_audit_epoch_means.jsonl` (60 rows).')
md.append('')
md.append('## Fold0 repair, original reuse, and R3 execution')
md.append('')
md.append('The evidence supports the engineering narrative with explicit boundaries:')
md.append('')
md.append('- Original execution `1c444cdf72e13fd041afd0c641dc8f522faa5844` trained fold0 for 260 optimizer steps, stopped at baseline feature parity, wrote no retrieval metric outputs, and did not start folds1/2 (`evidence/trifusion_msvr310_trifusion_v1_comparison_failure_receipt_20260906.json:3-8,52`).')
md.append('- R3 execution `1ff7e2d8ed2ed56668f814e229aca347f57b9b5d` reused fold0 training and verified arrays, trained only folds1/2, and asserted the final 520 new steps, 780 formal optimizer steps, 1032 evaluated gallery records, and 672 new heldout forwards (`tools/resume_msvr310_trifusion_exact_inference.py:82-170`).')
md.append('- The R3 wrapper pinned HEAD, runner SHA, plan SHA, and absence of an existing resume directory before launch (`evidence/trifusion_msvr310_trifusion_v1_resume_r3_wrapper_20260906.py:9-24`). The launch scope says fold0 training/verified360 features were reused and only folds1/2 were trained (`evidence/trifusion_msvr310_trifusion_v1_resume_r3_launch_20260906.json:1-12`). The run log begins with fold0 reused/new_steps=0, then shows fold1 and fold2 epoch logs (`evidence/trifusion_msvr310_trifusion_v1_resume_r3_run_20260906.log:1-80`).')
md.append('- Text receipts in the final summary record fold0 `training_reused_from_original_run=true`, fold0 `new_heldout_record_forwards=0`, fold1/2 `training_reused_from_original_run=false`, and final `new_heldout_record_forwards=672` (`evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json:19-22,23727-23729,55700-55704,87494-87498,95589`).')
md.append('- The exact inference helper temporarily uses a functional dispatch view for the frozen SIM in-projection weight and checks that original parameter object, flags, and grad state remain unchanged (`tools/msvr310_exact_signal_inference.py:4-22`).')
md.append('- The full360 verification reproduced the unrepaired difference, then confirmed repaired baseline features and 210x360 distances bitwise equal to B0 while model state, frozen state, signal state, parameter flags, and original batch sizes stayed unchanged; it performed zero optimizer/backward/checkpoint/ranking operations (`tools/verify_msvr310_exact_signal_inference.py:41-112`; `evidence/trifusion_msvr310_trifusion_v1_exact_signal_inference_verification_20260906.json:3,11-19,73-80`).')
md.append('')
md.append('The SIM dispatch diagnosis supports the mechanism within its fixed-input profiler scope: original/repeat/restored requires-grad states keep bitwise parity, while `freeze_sim_only`, build-before-load, load-final-roles, and original-after-load states change the SIM result by max 1.9073486328125e-06 (`evidence/trifusion_msvr310_trifusion_v1_sim_operation_parity_diagnosis_20260906.json:76-87,89-143`). The captured PyTorch source shows matmul folding decisions depend on `requires_grad` for the smaller operand and alternative bmm paths exist (`evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2111-2118,2157-2166`; `evidence/msvr310_pytorch251_Linear.cpp.txt:180-231`; MHA also gates fast path on `requires_grad`, `evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:162-175,621-639,667-678`).')
md.append('')
md.append('A precise caveat: the baseline-parity diagnosis shows that unrepaired `standalone_after_wrapping`, `hierarchical_baseline`, and `full_model_baseline` are not bitwise equal to stored B0, with max absolute difference 1.9073486328125e-06 (`evidence/trifusion_msvr310_trifusion_v1_baseline_parity_diagnosis_20260906.json:1-70`). The valid engineering claim is that the inference-only functional view restores exact B0 and preserves role outputs/state, not that the naturally frozen full-model path was already exact.')
md.append('')
md.append('## Stored-array and remote-binary receipt checks')
md.append('')
md.append('I could not load binary `.pt`/`.pth` files locally because the immutable package excludes raw tensor/checkpoint binaries. I checked all provided terminal receipt rows against fold receipt hashes, expected shapes, and all text flags. The terminal file verifier itself reports zero local model/tensor runtime in that check and `PASS_WHOLE_FILES_AND_ALL15_ARRAYS` (`evidence/trifusion_msvr310_trifusion_v1_terminal_files_verification_20260906.json:1-8,385-395`).')
md.append('')
md.append('| Fold | Output | Feature shape | Distance shape | Distance recompute receipt | Ranking sort receipt | Max distance diff |')
md.append('|---:|---|---:|---:|---:|---:|---:|')
for r in array_rows:
    md.append(f"| {r['fold']} | {r['output']} | {r['feature_shape']} | {r['distance_shape']} | {r['distance_recomputation_bitwise_equal']} | {r['saved_rankings_equal_full_saved_distance_sort']} | {r['maximum_absolute_distance_difference']} |")
md.append('')
md.append(f"Receipt summary from my replay: array_check_count={receipts['array_receipts']['array_check_count']}, all_flags_true={receipts['array_receipts']['all_flags_true']}, retrieval_array_hashes_match_receipts={receipts['array_receipts']['retrieval_array_hashes_match_receipts']}, checkpoint_hashes_match_receipts={receipts['array_receipts']['checkpoint_hashes_match_receipts']}, local_binary_arrays_available={receipts['array_receipts']['local_binary_arrays_available']}.")
md.append('')
md.append('## Dense outputs versus residual-only outputs')
md.append('')
md.append('The evaluated CNN/Transformer/Mamba outputs are dense extended branches, not standalone residual-only vectors. The V8 source concatenates the baseline embedding with the residual bank and returns expert branch embeddings through `fusion.branch_embeddings[expert]`; retrieval outputs are `baseline_only`, `fused`, or an expert branch (`modeling/trifusion/signal_preserving_v8.py:492-516,620-666,669-676`). The fused output is 7680D and branches are 4608D in the stored-array receipt table above.')
md.append('')
md.append('## B0/M0 and source binding inspection')
md.append('')
md.append('- Signal B0 binding inspected: status `COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION`, project commit `bb01d60b6e1517ee6f5dc9120faefd17d75401e5`, fixed epoch50, 1950 optimizer steps, 1032 heldout image forwards, no official test access, aggregate mAP 53.1293805608712 (`evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json:4,181765-181793`).')
md.append('- Original M0 binding inspected: status `PASS_ENGINEERING_ONLY`, project commit `1c444cdf72e13fd041afd0c641dc8f522faa5844`, discarded M0, source-only training, role weights not loaded, 124 optimizer steps, zero heldout forwards (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:4,31,3480-3492,12810-12812`).')
md.append('- Source binding: the manifest maps five actual remote LF project snapshots and 17 Signal source snapshots; the source binding receipt documents CRLF/LF byte differences but local LF equals remote, remote equals git blob, and AST equality for the five changed remote project files, with zero model calls or optimizer updates in that binding step (`evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json:1-86`).')
md.append('')
md.append('## Error census')
md.append('')
md.append('I independently replayed the post-hoc error census from the query rows and rankings. Counts match, but the census is descriptive only and cannot establish an exclusive cause. The census file itself states the same limit (`evidence/trifusion_msvr310_trifusion_v1_complete_error_census_20260906.json:1-67,45981`).')
md.append('')
md.append('| Output | Queries | Rank-1 repaired | Rank-1 new errors | New errors same camera | New errors same scene | All Rank-1 errors | All Rank-1 errors same camera | All Rank-1 errors same scene |')
md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for name in ['fused','cnn','transformer','mamba']:
    r = error['summary'][name]
    md.append(f"| {name} | {r['all_queries']} | {r['rank1_repaired']} | {r['rank1_new_errors']} | {r['new_errors_same_camera']} | {r['new_errors_same_scene']} | {r['all_rank1_errors']} | {r['all_rank1_errors_same_camera']} | {r['all_rank1_errors_same_scene']} |")
md.append('')
base_err = error['baseline_rank1_errors']
ident_dir = error['identity_directions']
md.append(f"Baseline rank-1 errors: total={base_err['total']}, same_camera={base_err['same_camera']}, same_scene={base_err['same_scene']}. Identity directions: improved={ident_dir['improved']}, declined={ident_dir['declined']}, unchanged={ident_dir['unchanged']}.")
md.append('')
md.append('## All 60 identity results')
md.append('')
md.append('| Identity | Queries | Baseline mAP | Fused mAP | CNN mAP | Transformer mAP | Mamba mAP | Fused delta pp |')
md.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in identity_rows:
    m = r['map_by_output']
    delta = m['fused'] - m['baseline_only']
    md.append(f"| {r['identity']} | {r['query_count']} | {m['baseline_only']:.6f} | {m['fused']:.6f} | {m['cnn']:.6f} | {m['transformer']:.6f} | {m['mamba']:.6f} | {delta:.6f} |")
md.append('')
md.append('## Actual issues and required qualification')
md.append('')
md.append('1. **Scientific support fails.** Do not claim the original three-role method improves MSVR310 under this comparison. Fused mAP is 52.1173901165861 versus Signal/B0 53.1293805608712, a -1.0119904442850967 pp drop; all five predefined scientific checks are false (`tools/train_msvr310_trifusion_oof.py:244-285`; `evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json:94833-94894`).')
md.append('2. **The exact-helper boundary must stay explicit.** The unrepaired frozen full-model/SIM paths are not bitwise exact to B0. The valid statement is: the inference-only functional view restores exact original B0 for feature extraction while preserving original parameter values/flags, role residuals, and training receipts.')
md.append('3. **Remote binary evidence is limited.** The local immutable package has no raw tensor/checkpoint binaries. The 15 stored-array path claims and checkpoint hashes are supported by terminal remote receipts and local text/JSON consistency, not by local binary reload. A stronger future audit package should include local raw binaries or an independently generated text digest sufficient to replay binary-array equality without torch.')
md.append('4. **Backend reviewer attestation is absent.** `001-terminal.request.json` records requested model `gpt-5.5` and reasoning `xhigh`, but `dispatch_observation.json` was absent; that request metadata is not independent backend attestation (`.aris/traces/experiment-audit/2026-09-06_run19/001-terminal.request.json:2,7-8`).')
md.append('')
md.append('## Evidence references')
md.append('')
for k, v in line_refs.items():
    md.append(f'- `{k}`: `{v}`')
md.append('')
md_text = '\n'.join(md) + '\n'

json_report = {
    'generated_at': now,
    'verdicts': verdicts,
    'limits': [
        'GPT-family Type-A local text/JSON audit; no dispatch_observation.json backend attestation present.',
        'No local raw .pt/.pth tensor/checkpoint binaries in immutable package; binary claims are remote receipt limited.',
        'No torch/model/tensor/image runtime used in this audit.',
        'B0/M0 prerequisite numerical audits not repeated; only bindings inspected.',
        'Error census verified descriptively only; no exclusive causal mechanism inferred.',
        'Scope is internal MSVR310 train-split OOF 600/1032, not official 591/1055 test evaluation.',
    ],
    'manifest': RESULT['manifest'],
    'replay_artifacts': replay_artifacts,
    'protocol_replay': protocol,
    'query_metric_replay_summary': {
        'queries': query['queries'],
        'query_identities': query['query_identities'],
        'query_output_evaluations': query['query_output_evaluations'],
        'max_aggregate_metric_abs_diff_to_summary': query['max_aggregate_metric_abs_diff_to_summary'],
        'max_fold_metric_abs_diff_to_receipts': query['max_fold_metric_abs_diff_to_receipts'],
        'max_average_precision_abs_diff_to_receipts': query['max_average_precision_abs_diff_to_receipts'],
        'rank_mismatch_count_upper_bound': query['rank_mismatch_count_upper_bound'],
        'bootstrap_lower_abs_diff_to_summary': query['bootstrap_lower_abs_diff_to_summary'],
        'comparison_replay': comparison_replay,
    },
    'training_scalar_replay': training,
    'error_census_replay': error,
    'receipt_checks': receipts,
    'array_checks': array_rows,
    'line_references': line_refs,
    'actual_issues': [
        'Scientific support fails all fixed gates; original three-role method should not be claimed as supported on this MSVR310 comparison.',
        'Unrepaired frozen full-model/SIM paths are not bitwise exact to B0; exact-helper boundary must remain explicit.',
        'Local package lacks raw tensor/checkpoint binaries; binary array claims are limited to terminal remote receipts.',
        'No dispatch_observation.json was present; requested reviewer metadata is not backend attestation.',
    ],
}

(ROOT / 'EXPERIMENT_AUDIT_MSVR310_TRIFUSION_TERMINAL.md').write_text(md_text, encoding='utf-8', newline='\n')
(ROOT / 'EXPERIMENT_AUDIT_MSVR310_TRIFUSION_TERMINAL.json').write_text(json.dumps(json_report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({
    'md': file_info(ROOT / 'EXPERIMENT_AUDIT_MSVR310_TRIFUSION_TERMINAL.md'),
    'json': file_info(ROOT / 'EXPERIMENT_AUDIT_MSVR310_TRIFUSION_TERMINAL.json'),
}, indent=2, sort_keys=True))
