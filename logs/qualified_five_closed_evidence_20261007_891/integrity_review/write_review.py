"""Write the fresh reviewer report and forensic trace; no primary evidence is edited."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

B = Path(r'C:\Users\gb\.codex_tmp\independent_evidence_draft')
O = B / 'qualified_five_integrity_audit891'
S = B / 'qualified_five_audit_sources891'
F = B / 'qualified_five_full_complete889'
SM = json.loads((S / 'FILE_MAP.json').read_text(encoding='utf-8-sig'))
FM = json.loads((F / 'FILE_MAP.json').read_text(encoding='utf-8-sig'))
MM = json.loads((O / 'm0_primary/checked_m0_terminal_failure887/FILE_MAP.json').read_text())
V = json.loads((O / 'DETERMINISTIC_CHECKS.json').read_text())
H = json.loads((O / 'AUDITED_INPUT_HASHES.json').read_text())
NOW = datetime.now().astimezone().isoformat()
TRACE = O / '.aris/traces/experiment-audit/2026-10-07_five891'
TRACE.mkdir(parents=True, exist_ok=True)


def resolve(original):
    if original in SM:
        return S / SM[original]['local_file']
    if original in FM:
        return F / FM[original]['local_file']
    if original in MM:
        return Path(MM[original]['local_file'])
    return Path(original)


def ref(original, line, end=None):
    return dict(original_path=original, local_path=str(resolve(original)), line=line, end_line=end or line)


def citation(original, line, end=None):
    p = resolve(original)
    span = str(line) if end is None else f'{line}–{end}'
    return f'`{original}:{span}` → [{p.name}:{line}]({p.as_posix()}:{line})'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


for p in [F / 'SOURCE.py', F / 'EXIT.json', F / 'STDERR.txt', S / 'SOURCE.py', S / 'REMOTE.json',
          O / 'EVIDENCE_PRECHECK.json', O / 'review_verify.py', O / 'DETERMINISTIC_CHECKS.json',
          Path(r'C:\Users\gb\.codex\skills\experiment-audit\SKILL.md'),
          Path(r'C:\Users\gb\.codex\skills\shared-references\local-codex-policy.md'),
          Path(r'C:\Users\gb\.codex\skills\shared-references\reviewer-independence.md'),
          Path(r'C:\Users\gb\.codex\skills\shared-references\experiment-integrity.md'),
          Path(r'C:\Users\gb\.codex\skills\shared-references\review-tracing.md')]:
    H[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
write(O / 'AUDITED_INPUT_HASHES.json', H)

CAMPAIGN = 'logs/qualified_five_incremental_full_v1_20261007_888/campaign.json'
SUMMARY = 'results/qualified_five_incremental_full_complete_20261007_888/SUMMARY.json'
REPORT = 'results/qualified_five_incremental_full_complete_20261007_888/REPORT.md'
M0EXIT = 'logs/incremental_role_objective_m0_launch_20261007_886/EXIT.json'
M0LOG = 'logs/incremental_role_objective_m0_launch_20261007_886/controller.log'
M0STEPS = 'trained-model/incremental_role_objective_m0_v1_20261007_886_m0_repair_keep_RGBNT100/training_steps.jsonl'
M0CAMPAIGN = 'logs/incremental_role_objective_m0_v1_20261007_886/campaign.json'

checks = {
    'A': dict(name='Ground truth provenance', status='PASS',
        details='The evaluated targets are protocol identities and camera/scene metadata parsed from dataset filenames. All supplied record labels/environment fields were independently checked against filenames; training identities are disjoint from query/gallery identities. Every query has a legal positive. Full protocol query/gallery rows are extracted without prediction-based filtering. MSVR310 retains 464 gallery-only records from 103 identities. This is manifest/source assurance, not a new physical image-archive census.',
        evidence=[ref('comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py',61,85),
            ref('comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py',63,84),
            ref('comparators/Signal-cd1b0a6/data/datasets/msvr310.py',67,87),
            ref('tools/official_three_dataset_data.py',8,18), ref('tools/run_correspondence_roles.py',66,107),
            ref('comparators/Signal-cd1b0a6/utils/metrics.py',68,108),
            ref('comparators/Signal-cd1b0a6/utils/metrics.py',111,170)]),
    'B': dict(name='Score normalization', status='PASS',
        details='mAP is the average of per-query precision at ground-truth match positions; CMC is an average of first-match indicators. Scores are multiplied by 100, with no metric denominator from the model maximum/minimum/mean. L2 feature normalization is conventional distance preparation. The MD batch distance ratio and detached-global repair/keep reference are training losses only, not reported evaluation ground truth or metric scaling.',
        evidence=[ref('tools/run_official_three_dataset_roles.py',230,238),
            ref('tools/train_msvr310_signal_oof.py',223,238),
            ref('tools/train_rgbnt100_signal_oof.py',253,268),
            ref('modeling/trifusion/incremental_role_objectives.py',18,70),
            ref('tools/run_incremental_role_objective_checked.py',22,44)]),
    'C': dict(name='Result existence and numeric consistency', status='PASS',
        details='All 36 intake hashes, 423 snapshot entries and 45 primary full-run texts match their supplied digests. All 45 full-run text payloads exactly match the original collection transcript. The five 50-epoch records contain 9,839 finite logged updates, contiguous epoch/batch indices and corresponding full batch orders; epoch mean losses, component sums, checkpoint selection, official receipts and 12 complete query tables agree. Remote checkpoint/distance existence and byte hashes are supported by the inspected original collector plus terminal transcript, not a fresh remote or local tensor replay by this reviewer.',
        evidence=[ref(str(F / 'SOURCE.py'),9,24),ref(str(F / 'SOURCE.py'),39,72),
            ref(CAMPAIGN,786,790),ref(SUMMARY,5,7),
            ref('tools/run_foundation_recipe.py',221,258),ref('tools/run_foundation_recipe.py',287,312),
            ref('tools/report_incremental_qualified_five.py',35,72)]),
    'D': dict(name='Actual called evaluation path', status='PASS',
        details='The journal invokes run_incremental_role_objective_checked.py for train and evaluate. Its configured dependency chain reaches run_foundation_recipe.train/evaluate and run_correspondence_roles.official_metrics, which directly calls both independent camera_scores/scene_scores and the pinned upstream eval_func/eval_func_msrv. Historical training loops, OOF evaluation_gallery routines, R1_mAP wrappers, visualization functions and old teacher-target code are present in the sealed snapshot but are not the reported campaign execution path. The 420-entry source seal is not an assertion that 420 files executed.',
        evidence=[ref(CAMPAIGN,11,76),ref('tools/run_incremental_role_objective_checked.py',47,50),
            ref('tools/run_incremental_role_objective.py',84,120),ref('tools/run_global_task_role.py',39,49),
            ref('tools/run_role_input_detach.py',34,41),ref('tools/run_native_research.py',22,25),
            ref('tools/run_native_partitioned.py',43,60),ref('tools/run_independent_native_evidence.py',167,184),
            ref('tools/run_foundation_recipe.py',287,312),ref('tools/run_correspondence_roles.py',79,108)]),
    'E': dict(name='Scope, completion, seeds and selection', status='WARN',
        details='The explicitly qualified five-endpoint record is complete: two objectives on RGBNT201/MSVR310 and only md_batch_ratio on RGBNT100, each seed 42 and 50 epochs. All CMC ranks use the same mAP-best checkpoint, with latest-epoch tie-breaking, and one subsequent fresh-construction strict-reload evaluation. The official split was used every epoch for selection and had prior development use. These are consumed-benchmark single-seed development results, not untouched held-out estimates or training-seed significance. None of the 12 comparisons passes the stated >=0.5 mAP/no-R1-decrease progress rule. RGBNT100 repair_keep remains genuinely missing after auxiliary M0 activity rejection; the original EXIT1 is retained despite stale RUNNING controller state.',
        evidence=[ref('tools/queue_incremental_qualified_five.py',17,17),
            ref('tools/queue_incremental_qualified_five.py',28,53),
            ref('tools/queue_incremental_qualified_five.py',63,82),
            ref('tools/run_foundation_recipe.py',248,255),
            ref('tools/run_foundation_recipe.py',287,310),
            ref('tools/analyze_correspondence_distances.py',57,75),
            ref('tools/report_incremental_qualified_five.py',65,78),ref(REPORT,11,13),
            ref(M0EXIT,1),ref(M0LOG,1,10),ref(M0STEPS,1,8)]),
    'F': dict(name='Evaluation type', status='PASS', classification='real_gt',
        details='All five formal retrieval evaluations are real_gt. The auxiliary detached global reference defines training repair/keep cells; it is not a synthetic evaluation target. No human, simulated, or prediction-derived retrieval ground truth is used by the called evaluator.',
        evidence=[ref('tools/run_correspondence_roles.py',88,102),
            ref('modeling/trifusion/incremental_role_objectives.py',43,70),
            ref('tools/run_incremental_role_objective.py',106,120)])
}
issues = [
    dict(id='W1', severity='warning', blocking=False, category='claim_scope',
        finding='Official-split checkpoint selection and prior development consumption with one training seed.',
        claim_impact='Supports descriptive best-of-50 seed-42 comparisons only; does not support untouched-test, training-seed significance, broad robustness, novel loss or SOTA claims.',
        action='Retain the existing consumed-benchmark and single-seed qualifiers; do not reinterpret identity bootstrap intervals as training-seed inference.',
        evidence=checks['E']['evidence'][3:8]),
    dict(id='W2', severity='warning', blocking=False, category='missing_endpoint',
        finding='RGBNT100 repair_keep has no formal endpoint because the retained checked M0 failed auxiliary activity acceptance.',
        claim_impact='The five qualified runs are complete. A six-endpoint campaign or three-dataset repair_keep claim remains unsupported.',
        action='Keep the missing endpoint and original EXIT1 explicit; do not replace it with M0_PASS, a zero score, or another checkpoint.',
        evidence=[ref(M0EXIT,1),ref(M0LOG,1,10),ref(M0STEPS,1,8),ref(REPORT,13)]),
    dict(id='W3', severity='warning', blocking=False, category='assurance_boundary',
        finding='The canonical evidence_check.py precheck is UNRESOLVED; raw tensors, physical dataset inventory and model runtime were not independently replayed in this read-only local review.',
        claim_impact='Local byte/text arithmetic verification is accepted for its exact deterministic scope. Semantic assurance remains same-family/provisional; remote binary evidence is transcript-backed.',
        action='Preserve this assurance boundary. Do not describe this review as fresh inference, fresh remote SHA verification, canonical-verifier PASS, or runtime-attested cross-family acceptance.',
        evidence=[ref(str(O / 'EVIDENCE_PRECHECK.json'),2,7),ref(str(F / 'SOURCE.py'),55,63)])
]
claims = [
    dict(id='C1', claim='Five qualified seed-42 full50 runs, 250 epochs, 9,839 logged/effective updates and one report', impact='supported_with_scope',
         boundary='Source plus complete step/exit records support actual execution; no new training or independent runtime replay was performed by this audit.'),
    dict(id='C2', claim='Full protocol real-GT retrieval with one mAP-best checkpoint for all CMC ranks', impact='supported',
         boundary='Latest-epoch mAP ties are intentional. Fresh strict reload verifies checkpoint consistency; it does not create independent held-out data.'),
    dict(id='C3', claim='Unchanged model initialization and actual full training order relative to raw semantic control', impact='supported_with_scope',
         boundary='Full path/label/camera order bytes and initializer digests match sealed controls. This is not a separate proof of every stochastic augmentation value.'),
    dict(id='C4', claim='The 12 post-selection paired query diagnoses are numerically consistent', impact='supported_with_scope',
         boundary='All query/AP/first-rank tables, identity groups and means recomputed. Raw distance ranking and NumPy bootstrap draws were not replayed locally.'),
    dict(id='C5', claim='Progress gate, robust effectiveness, multi-seed significance or broad goal completion', impact='unsupported',
         boundary='0 of 12 progress gates pass. One seed, consumed official selection, and missing repair_keep RGBNT100 prevent these claims.'),
    dict(id='C6', claim='Three-dataset repair_keep or six completed formal endpoints', impact='unsupported',
         boundary='RGBNT100 repair_keep is absent after recorded activity-gate failure.'),
    dict(id='C7', claim='Training cost measured by sum(history.seconds)', impact='needs_qualifier',
         boundary='history.seconds ends before each official evaluation. Use receipt wallclock for training plus epoch evaluations and process timing for construction/startup-inclusive cost.'),
    dict(id='C8', claim='All sealed historical code executed or a teacher supplied evaluation targets', impact='unsupported',
         boundary='Historical imported functions are not all called; _build_signal_teacher is the constructor name, not evidence of teacher supervision in this route.')
]

result = dict(audit_skill='experiment-audit', verdict='WARN', overall_verdict='warn', integrity_status='warn',
    reason_code='qualified_record_consistent_claim_and_assurance_boundaries', blocking_count=0, warning_count=len(issues),
    summary='No blocking integrity defect found in the explicitly qualified five-endpoint record. The results and completion evidence are consistent; retain consumed-official/single-seed, missing-endpoint and transcript-backed assurance boundaries.',
    generated_at=NOW, date='2026-10-07', project='TriFusion qualified five incremental objectives, campaign 888',
    agent_id='/root/audit_five891', verdict_id='2026-10-07_five891', reviewer_backend='native Codex agent',
    requested_reviewer_model='gpt-6-astra', requested_reviewer_reasoning_effort='max',
    reviewer_model=None, reviewer_reasoning=None, runtime_attestation='unavailable',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    auditor='Fresh same-family Codex reviewer; requested gpt-6-astra/max; actual model/effort unattested',
    checks=checks, issues=issues, claims=claims, evaluation_type='real_gt',
    deterministic_verification=dict(status='PASS', acceptance_status='accepted', review_independence='deterministic',
        scope='Only exact local bytes, supplied text payloads, manifest metadata and recomputed text arithmetic.',
        artifact=str(O / 'DETERMINISTIC_CHECKS.json'), counts=V['byte_verification']),
    canonical_evidence_check=dict(status='UNRESOLVED', artifact=str(O / 'EVIDENCE_PRECHECK.json')),
    audited_input_hashes={name:'sha256:'+digest for name,digest in H.items()},
    audited_hash_manifest=str(O / 'AUDITED_INPUT_HASHES.json'),
    hash_manifest_sha256=hashlib.sha256((O / 'AUDITED_INPUT_HASHES.json').read_bytes()).hexdigest(),
    primary_results=V['rows'], paired_comparisons=V['pairs'], missing_formal_endpoint=dict(dataset='RGBNT100',objective='repair_keep'),
    trace_path=str(TRACE), limitations=V['limits'])

lines = [
    '# Experiment integrity audit — qualified five endpoints', '',
    '**Verdict: WARN. Integrity status: warn. Blocking findings: 0 for the explicitly qualified five-endpoint record.**', '',
    'The primary artifacts support five completed seed-42 full50 runs, 250 epochs, 9,839 logged optimizer updates and one completed report. No fabricated retrieval target, own-output metric normalization, phantom current result, rank-wise checkpoint splicing or hidden sixth formal endpoint was found. All 12 declared progress gates evaluate to false. The warnings below preserve claim and assurance limits already relevant to this record.', '',
    f'Date: 2026-10-07. Reviewer: fresh native Codex agent `/root/audit_five891`; requested model `gpt-6-astra`, effort `max`. Runtime model/effort attestation is unavailable. `review_independence=same-family`; `acceptance_status=provisional`. This is a semantic review of primary artifacts, not cross-family acceptance.', '',
    '## Evidence and deterministic checks', '',
    'The independent standard-library verifier read all 36 intake artifacts, hashed all 423 snapshot entries and 45 full-run text files, and matched the 45 text payloads to the supplied original collection transcript. It parsed every full training step and batch-order row, every epoch, all protocol records, and all 12 complete per-query tables. It also checked exact embedded bytes for 48 M0-matrix and 51 failure-transcript primary texts. The campaign seals 420 source entries; that number includes historical code and does not mean 420 files executed.', '',
    f'Accepted deterministic output: [DETERMINISTIC_CHECKS.json]({(O / "DETERMINISTIC_CHECKS.json").as_posix()}). Actual audited SHA-256 values: [AUDITED_INPUT_HASHES.json]({(O / "AUDITED_INPUT_HASHES.json").as_posix()}). The attached verification script is reviewer-authored, does not import the experiment package and performs no neural execution.', '',
    'Physical remote checkpoint/distance existence is supported by the original collector’s byte assertions and its terminal transcript; this reviewer did not issue SSH or independently open the remote binaries. Local numeric checks are therefore a separate, narrower assurance layer. The canonical `evidence_check.py` precheck remains UNRESOLVED; it is not relabeled PASS.', '',
    citation(str(F / 'SOURCE.py'),9,24) + '; ' + citation(str(F / 'SOURCE.py'),55,72) + '; ' + citation(str(O / 'EVIDENCE_PRECHECK.json'),2,7) + '.', '',
    '## A. Ground truth provenance — PASS', '',
    checks['A']['details'], '',
    'The loaders read train labels only for training, and original identities for query/gallery. The evaluator obtains q/g identities and camera/scene arrays from the protocol, not model predictions. Camera datasets remove only same-identity/same-camera items; MSVR310 removes only same-identity/same-scene items. Different identities remain distractors. Query/gallery order is not sampled or shuffled in evaluation.', '',
    '| Dataset | Train records / IDs | Query records / IDs | Gallery records / IDs | Gallery-only records / IDs | Legal positive range |',
    '|---|---:|---:|---:|---:|---:|'
]
for p in V['protocols']:
    lines.append(f"| {p['dataset']} | {p['counts']['train']} / {p['train_identities']} | {p['counts']['query']} / {p['query_identities']} | {p['counts']['gallery']} / {p['gallery_identities']} | {p['gallery_only_records']} / {len(p['gallery_only_identities'])} | {p['legal_positive_count_range'][0]}–{p['legal_positive_count_range'][1]} |")
lines += ['', citation('tools/official_three_dataset_data.py',8,18) + '; ' + citation('tools/run_correspondence_roles.py',66,107) + '.', '',
    'Filename parsing: ' + citation('comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py',61,85) + '; ' + citation('comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py',63,84) + '; ' + citation('comparators/Signal-cd1b0a6/data/datasets/msvr310.py',67,87) + '.', '',
    'Author metric filters: ' + citation('comparators/Signal-cd1b0a6/utils/metrics.py',68,108) + '; ' + citation('comparators/Signal-cd1b0a6/utils/metrics.py',111,170) + '.', '',
    'Protocol count locations: ' + citation('logs/training_feature_scale_protocols_20261002/RGBNT201.json',78309,78314) + '; ' + citation('logs/training_feature_scale_protocols_20261002/MSVR310.json',38538,38543) + '; ' + citation('logs/training_feature_scale_protocols_20261002/RGBNT100.json',218978,218983) + '.', '',
    '## B. Score normalization — PASS', '', checks['B']['details'], '',
    citation('tools/train_msvr310_signal_oof.py',223,238) + '; ' + citation('tools/train_rgbnt100_signal_oof.py',253,268) + '; ' + citation('tools/run_official_three_dataset_roles.py',230,238) + '; ' + citation('modeling/trifusion/incremental_role_objectives.py',18,70) + '.', '',
    '## C. Result existence, arithmetic and timing — PASS within stated evidence access', '', checks['C']['details'], '',
    '| Dataset | Objective | mAP-best epoch | mAP | R1 | R5 | R10 | Updates | E50 mAP |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in V['rows']:
    met = r['metrics']
    lines.append(f"| {r['dataset']} | {r['objective']} | {r['best_epoch']} | {met['mAP']:.6f} | {met['Rank-1']:.6f} | {met['Rank-5']:.6f} | {met['Rank-10']:.6f} | {r['formal_steps']} | {r['last_epoch_metrics']['mAP']:.6f} |")
lines += ['', 'Each row’s four retrieval metrics matches its single mAP-best history row and its first subsequent strict evaluation, with a maximum allowed parity error of 1e-5 percentage points. The actual selected epochs are 8, 8, 38, 38 and 5. Ties select the latest epoch because training saves on `>=` and evaluation maximizes `(mAP, epoch)`.', '',
    citation('tools/run_foundation_recipe.py',248,255) + '; ' + citation('tools/run_foundation_recipe.py',287,312) + '.', '']
for r in V['rows']:
    orig = next(k for k,v in FM.items() if str(F/v['local_file']) == r['official_file'])
    lines.append(f"- {r['dataset']} {r['objective']}: " + citation(orig,8) + '; ' + citation(orig,37,47) + '.')
lines += ['', 'All five update streams have finite values, contiguous epoch/batch pairs, correct epoch mean losses and full matching order streams. Each path’s logged label/camera also matches its training protocol. The formal order SHA equals the existing raw-semantic control seal on all three datasets; the two paired objectives have byte-identical order files on RGBNT201 and MSVR310. Initialization bindings retain identical visual/camera/shared/full-model starting-state digests and optimizer configuration versus raw semantic, except the explicitly changed objective/entry metadata. Full runs construct fresh public initialization and load no M0 probe.', '',
    citation('tools/run_clean_clip_joint.py',37,107) + '; ' + citation('tools/run_foundation_recipe.py',103,107) + '; ' + citation('tools/run_foundation_recipe.py',189,244) + '; ' + citation('refine-logs/incremental_role_objective_v1/INPUT_SEAL.json',450,473) + '.', '',
    'Training cost must distinguish three clocks. `history.seconds` stops before per-epoch official evaluation. Receipt wallclock includes training plus epoch evaluation/checkpoint/receipt work and excludes model construction and final strict evaluation. The parent train-process interval additionally includes startup/construction. These values are measured on the shared execution environment, not an isolated hardware benchmark.', '',
    '| Dataset / objective | Sum history.seconds | Training receipt wall (s) | Train process (s) | Final strict eval (s) |',
    '|---|---:|---:|---:|---:|']
for r in V['rows']:
    lines.append(f"| {r['dataset']} / {r['objective']} | {r['train_loop_seconds']:.3f} | {r['training_receipt_wall_seconds']:.3f} | {r['train_process_seconds']:.3f} | {r['strict_evaluation_seconds']:.3f} |")
lines += ['', citation('tools/run_foundation_recipe.py',200,203) + '; ' + citation('tools/run_foundation_recipe.py',246,257) + '; ' + citation('tools/report_incremental_qualified_five.py',39,45) + '.', '',
    'Concrete timing and complete-stream references:', '']
for r in V['rows']:
    orig = next(k for k,v in FM.items() if str(F/v['local_file']) == r['training_file'])
    start,end = (90,700) if r['dataset']=='RGBNT201' else (98,708)
    step_orig = orig.removesuffix('training.json')+'training_steps.jsonl'
    order_orig = orig.removesuffix('training.json')+'training_batch_order.jsonl'
    lines.append(f"- {r['dataset']} {r['objective']}: " + citation(orig,start) + '; ' + citation(orig,end) + '; ' + citation(step_orig,1,r['formal_steps']) + '; ' + citation(order_orig,1,r['formal_steps']) + '.')
lines += ['', 'The 12 query tables cover 836 RGBNT201, 591 MSVR310 or 1,715 RGBNT100 queries each. This audit recomputed mAP, R1/R5/R10, all deltas, rank-1 repairs/harms, query and identity improvement counts, and identity macro means from the saved tables. Tiny CMC differences between upstream float32 averages and exact query-count percentages stay below 1e-5; they are not missing queries. None passes the declared progress rule.', '',
    citation('tools/report_incremental_qualified_five.py',49,72) + '; ' + citation(SUMMARY,3429) + '; ' + citation('tools/analyze_correspondence_distances.py',28,75) + '.', '',
    '## D. Executed evaluation path — PASS', '', checks['D']['details'], '',
    'Actual route: checked entry → incremental objective configuration → global-task-role / role-input-detach wrappers → native-research / partitioned wrappers → independent-native trainer → foundation train/evaluate → correspondence official_metrics → full-protocol extraction and distance_matrix → independent scorer plus pinned author metric. No result in this campaign depends on calling the historical standalone OOF or visualization entry points.', '',
    citation('tools/run_incremental_role_objective_checked.py',47,50) + '; ' + citation('tools/run_incremental_role_objective.py',84,120) + '; ' + citation('tools/run_global_task_role.py',39,49) + '; ' + citation('tools/run_role_input_detach.py',34,41) + '; ' + citation('tools/run_native_research.py',22,25) + '; ' + citation('tools/run_native_partitioned.py',43,60) + '; ' + citation('tools/run_independent_native_evidence.py',167,184) + '.', '',
    'The source contains an `_eval_loader` that constructs doubled records as a historical template. The called RGBNT201 loader takes only that template’s transform and constructs a new single-record-list dataset. Full extracted shapes are asserted, and full query counts were verified; this is not duplicate query evaluation.', '',
    citation('tools/build_v12_complete_path_oof_targets.py',362,381) + '; ' + citation('tools/train_signal_preserving_v18.py',33,41) + '; ' + citation('tools/run_correspondence_roles.py',66,76) + '.', '',
    'The function named `_build_signal_teacher` only constructs `make_frame`; the current initializer verifies public visual tensors and no ReID checkpoint loading. Historical teacher-target, PlainFoundation and old correspondence training functions must not be described as executed by these five jobs.', '',
    citation('tools/build_v12_complete_path_oof_targets.py',244,258) + '; ' + citation('tools/run_clean_clip_joint.py',37,107) + '.', '',
    '## E. Scope, selection and failure retention — WARN', '', checks['E']['details'], '',
    'The full parent exits 0, all five jobs each contain one train followed by one evaluate invocation, and report_invocations is 1 with report_exit_code 0. “First independent strict evaluation” is supported as a fresh process/model construction and strict reload after training; the evaluated benchmark is the same one used for selection.', '',
    citation(CAMPAIGN,11,76) + '; ' + citation(CAMPAIGN,786,790) + '; ' + citation('logs/qualified_five_incremental_full_launch_20261007_888/EXIT.json',1) + '; ' + citation(str(B/'qualified_five_full_observer888/COMPLETE.json'),1) + '.', '',
    'The rejected RGBNT100 repair_keep M0 genuinely performed eight production optimizer updates and passed reload, while both role-2 query/key isolated auxiliary gradient sums were exactly zero and unused=False. The six-job activity gate therefore failed. Its original campaign still says RUNNING because the controller threw before a terminal state update; original EXIT1 and the assertion traceback are the terminal authority. Five accepted jobs are explicitly selected by the new launcher. No sixth formal score is substituted.', '',
    citation(M0CAMPAIGN,2) + '; ' + citation(M0EXIT,1) + '; ' + citation(M0LOG,1,10) + '; ' + citation(M0STEPS,1,8) + '; ' + citation('tools/queue_incremental_qualified_five.py',28,53) + '; ' + citation(REPORT,13) + '.', '',
    'The bootstrap resamples 30, 52 or 50 fixed-model identities and estimates an identity-macro AP difference. It is not the query-weighted mAP estimator, a distribution over training seeds, or a correction for repeated benchmark selection. Even the two reported intervals that exclude zero cannot establish training-run significance. The report explicitly retains this boundary.', '',
    citation('tools/analyze_correspondence_distances.py',57,75) + '; ' + citation('tools/report_incremental_qualified_five.py',50,52) + '; ' + citation(REPORT,11) + '.', '',
    '## F. Evaluation type — PASS / real_gt', '', checks['F']['details'], '',
    citation('tools/run_correspondence_roles.py',88,107) + '; ' + citation('modeling/trifusion/incremental_role_objectives.py',43,70) + '.', '',
    '## Issues and claim impact', '']
for issue in issues:
    lines += [f"- **{issue['id']} — {issue['finding']}** {issue['claim_impact']} {issue['action']}"]
lines += ['', 'No correction to the five reported formal metrics is required by the inspected evidence. Zero blocking findings means the explicitly bounded record can be used as a provisional development result; it does not mean the missing experiment, broad project goal, unseen-data validation or stronger statistical claims are complete.', '',
    '| Claim | Impact | Boundary |', '|---|---|---|']
for claim in claims:
    lines.append(f"| {claim['claim']} | {claim['impact']} | {claim['boundary']} |")
lines += ['', '## Trace and access boundary', '',
    f'Full task prompt, substantive raw reviewer response and metadata are retained under [{TRACE.name}]({TRACE.as_posix()}). The substantive response in the trace is byte-identical to this report. No SSH, GPU query, neural execution, package installation, source/Git change, or primary-evidence alteration was performed. Only this audit directory contains reviewer outputs and exact M0 text extracts.', '']

response = '\n'.join(lines)
(O / 'EXPERIMENT_AUDIT.md').write_text(response, encoding='utf-8')
write(O / 'EXPERIMENT_AUDIT.json', result)

prompt = """Apply C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and its local Codex policy. Fresh same-family provisional integrity review, read-only. Do not spawn agents, run SSH/GPU/NN, install packages, change source/Git or alter existing evidence. Only write your review outputs and trace under C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_integrity_audit891. No result interpretation from executor is provided: inspect primary artifacts yourself.

Artifact PATHS:
- C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_integrity_audit891/AUDIT_INPUT_PATHS.json (36 actual source/protocol/primary paths and hashes; read all relevant listed artifacts, line by line for evaluation code)
- C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/FILE_MAP.json (exact executed source snapshot mapping; follow all imported execution/evaluation dependencies here rather than using potentially CRLF/missing local working-tree source)
- C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_full_complete889/FILE_MAP.json (all primary full experiment text paths; use mapped training_steps/batch_order when checking actual updates/order)
- C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_full_complete889/REMOTE.json (original collection and physical SHA verification transcript; huge, parse selected keys rather than dumping it)
- C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_integrity_audit891/EVIDENCE_PRECHECK.json
- C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_full_observer888/COMPLETE.json
- C:/Users/gb/.codex_tmp/independent_evidence_draft/checked_m0_matrix887/REMOTE.json
- C:/Users/gb/.codex_tmp/independent_evidence_draft/checked_m0_terminal_failure887/REMOTE.json

Be adversarial; trust no author/executor summary. Audit A-F: dataset ground-truth provenance/official camera-scene filters/full query-gallery and distractors, no prediction-derived GT or own-max score normalization, physical result existence and numeric consistency, actual called evaluation path versus dead historical code, completion/scope/seed/best-selection versus claims, evaluation type. Check unchanged initialization/source and actual full training orders, single selected mAP-best for all CMC, first independent strict eval, exact five-job completion and missing formal endpoint boundary, original failure retention, report count, training wallclock versus history.seconds. Do not upgrade fixed-model identity bootstrap into training-seed significance. Deliberately distinguish deterministic byte verification from semantic assurance. Source snapshot includes old code not executed in this campaign: trace actual imports and report scope accurately rather than declaring every historical file executed.

Write EXPERIMENT_AUDIT.md and EXPERIMENT_AUDIT.json with exact mapped original-path/local-file:line citations, A-F statuses, overall verdict, integrity_status, blocking_count and issues, claim-impact boundaries, actual audited input SHA, review_independence=same-family, acceptance_status=provisional, requested reviewer model=gpt-6-astra/effort=max; runtime attestation unavailable, do not invent it. Preserve full prompt/raw response trace under .aris/traces/experiment-audit/2026-10-07_five891/ within your output directory. Return concise verdict and artifact paths."""
followup = """Working local Python: use `uv run --offline python -X utf8 <local script>` (or pipe a PowerShell here-string to `uv run --offline python -`). Avoid generic python / E:/Scripts. No installation needed. New separate saved-text analysis is available at C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_training_text_analysis891/SUMMARY.json and REPORT.md if useful; inspect its primary hashes and complete step records yourself. It only derives fields from existing logs and does not change the original inputs or your verdict."""
write(TRACE/'run.meta.json', dict(skill='experiment-audit', run_id='2026-10-07_five891',
    started_at='2026-10-07', completed_at=NOW, start_clock_precision='date_only_not_runtime_attested',
    executor='codex', executor_model=None, executor_family='openai', project_dir=str(O),
    requested_reviewer_model='gpt-6-astra', requested_reasoning_effort='max', runtime_attestation='unavailable',
    review_independence='same-family', acceptance_status='provisional'))
write(TRACE/'001-integrity-review.request.json', dict(call_number=1, purpose='fresh-primary-artifact-integrity-review',
    tool='spawn_agent', model='gpt-6-astra', reasoning_effort='max', model_fields_semantics='requested only; not runtime attestation',
    files_referenced=[str(O/'AUDIT_INPUT_PATHS.json'),str(S/'FILE_MAP.json'),str(F/'FILE_MAP.json'),str(F/'REMOTE.json'),
        str(O/'EVIDENCE_PRECHECK.json'),str(B/'qualified_five_full_observer888/COMPLETE.json'),
        str(B/'checked_m0_matrix887/REMOTE.json'),str(B/'checked_m0_terminal_failure887/REMOTE.json')],
    prompt=prompt, during_review_executor_message=followup,
    during_review_executor_message_usage='Used only supplied working Python invocation. New executor analysis was not read or used as evidence.'))
(TRACE/'001-integrity-review.response.md').write_text(response, encoding='utf-8')
write(TRACE/'001-integrity-review.meta.json', dict(call_number=1,purpose='fresh-primary-artifact-integrity-review',
    timestamp=NOW,agent_id='/root/audit_five891',requested_model='gpt-6-astra',requested_reasoning_effort='max',
    model=None,reasoning_effort=None,runtime_attestation='unavailable',reviewer_family='openai',
    review_independence='same-family',acceptance_status='provisional',status='ok',verdict='WARN',blocking_count=0,
    response_sha256=hashlib.sha256((TRACE/'001-integrity-review.response.md').read_bytes()).hexdigest()))
events = O/'.aris/meta'
events.mkdir(parents=True,exist_ok=True)
with (events/'events.jsonl').open('w',encoding='utf-8') as stream:
    stream.write(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='fresh-primary-artifact-integrity-review',
        agent_id='/root/audit_five891',trace_path=str(TRACE),status='ok',timestamp=NOW))+'\n')
print(json.dumps(dict(verdict='WARN',integrity_status='warn',blocking_count=0,
    report=str(O/'EXPERIMENT_AUDIT.md'),json=str(O/'EXPERIMENT_AUDIT.json'),trace=str(TRACE)),indent=2))
