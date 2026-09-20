import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root=Path(__file__).resolve().parent
def read(name):return json.loads((root/name).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
initial=read('intake_01.stdout')
static=read('static_dependencies.stdout')
static_receipt=read('static_bindings_receipt.json')
provenance=read('provenance.stdout')
gt=read('ground_truth_disk.stdout')
terminal=read('terminal_intake.stdout')
numeric=read('independent_m0.stdout')
docs=read('final_document_state.stdout')
summary=json.loads(terminal['files']['m0/summary.json']['text'])
assert numeric['status']=='PASS_INDEPENDENT_COMPLETE_M0_CPU_RECONSTRUCTION'
assert provenance['status']=='PASS_STATIC_PROVENANCE_AND_REGISTERED_TEXT_REPLAY'
assert gt['status']=='PASS_ACTUAL_TRAIN_DIRECTORY_LABEL_PROVENANCE'
assert not static_receipt['binding_mismatches'] and not static_receipt['baseline_mismatches']
for p,r in static['files'].items():
    if 'signal_commit_actual' in r:
        config=json.loads(r['text'])
        assert r['signal_commit_actual']==config['signal_commit']
        assert r['signal_diff_sha256_actual']==config['signal_diff_sha256']
assert not any(p.startswith(('tools/','configs/','modeling/','protocols/')) for p in docs['changed_since_execution'])
assert '| M0 |' in docs['tracker']['text'] and 'PASS_ENGINEERING_ONLY' in docs['tracker']['text']
assert '| M0_CPU |' in docs['tracker']['text'] and '| M0_Audit |' in docs['tracker']['text']
for suffix in ('intake_01.stdout','static_dependencies.stdout','provenance.stdout','terminal_intake.stdout','independent_m0.stdout','ground_truth_disk.stdout','final_document_state.stdout'):
    assert (root/(suffix+'.stderr')).stat().st_size==0

repo='/root/autodl-tmp/trifusion-v2/TriFusion-ReID'
run='/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d'
hashes={repo+'/'+n:r['sha256'] for n,r in initial['files'].items() if r.get('exists')}
hashes.update({n:r['sha256'] for n,r in static['files'].items()})
hashes.update({n:r['sha256'] for n,r in gt['files'].items()})
hashes.update({n:r['sha256'] for n,r in numeric['files'].items()})
hashes[docs['tracker']['path']]=docs['tracker']['sha256']
checks={
 'A_ground_truth_provenance':{
  'status':'PASS','evaluation_type':'real_gt',
  'details':'Dataset filename labels, not model outputs. All 1032 protocol records agree with the frozen label manifest and loader parsing; the current source directory has the exact 3096 modality filenames. Three folds are identity-disjoint; all 1950 recorded B0 source steps and all 248 M0 steps use their registered source indices. Three B0 checkpoint payloads and six M0 payloads bind source/heldout identities exactly.',
  'evidence':['tools/audit_vehicle_query_protocol_labels.py:13-18','tools/build_msvr310_train_oof_protocol.py:12-55','tools/build_msvr310_train_oof_protocol.py:57-123','tools/train_msvr310_signal_oof.py:68-99','tools/train_msvr310_trifusion_oof.py:27-55','ground_truth_disk.stdout','provenance.stdout','independent_m0.stdout'],
  'limits':'Image content and historical archive CRC were not rerun; source image filenames were read, not images. The verified source-only training record is a saved execution record, not a retrained baseline.'},
 'B_score_and_reduction':{
  'status':'PASS',
  'details':'All 248 rows independently reconstruct pooled hard loss, standard Smooth-AP, cross-scene Smooth-AP, eligible masks/counts, active selection, three branch current hard losses, and the 14-term weighted scalar ledger. Cross-scene AP removes same-ID/same-scene positions from both rank sums, retains every different-ID negative, excludes each positive self-comparison, and averages only eligible anchors. The overfit denominator is explicitly initial excess loss above an analytic label-smoothing entropy floor; raw initial/final losses are retained. It is an engineering ratio, not normalized retrieval performance.',
  'evidence':['tools/msvr_cross_scene_smooth_ap.py:11-54','tools/msvr_smooth_ap.py:30-62','tools/train_msvr_cross_scene_smooth_ap.py:129-164','tools/run_signal_preserving_v5.py:99-135','tools/run_signal_preserving_v5.py:1580-1617','independent_m0.stdout'],
  'limits':'The ten classification/residual scalar terms have no saved logits/residual distances for fresh per-term reconstruction; all 14 recorded scalars and their weights are independently recombined.'},
 'C_result_files_and_completion':{
  'status':'PASS',
  'details':'Authoritative pipeline records T0, M0 and M0_CPU exit 0; M0 is PASS_ENGINEERING_ONLY with 248 updates. All eight training files, eight complete distance/JSONL pairs, six receipts and six role checkpoints exist and match registered terminal hashes. Six final 472-tensor states reconstruct exactly from 241 B0 aliases plus 231 role tensors. The current tracker records M0 and M0_CPU complete while this independent audit is running. No pending Q1 result is called complete.',
  'evidence':['tools/run_msvr_cross_scene_smooth_ap.py:31-65','tools/train_msvr_cross_scene_smooth_ap.py:322-388','tools/train_msvr310_source_style.py:164-174','terminal_intake.stdout','independent_m0.stdout','final_document_state.stdout'],
  'limits':'The initial 00:32:46 snapshot was RUNNING and is retained. Latest source HEAD is documentation/evidence-only relative to execution d35864d; source/config/protocol paths did not change.'},
 'D_real_call_paths':{
  'status':'WARN',
  'details':'The live trainer invokes the same objective adapter for current graphs (line 129), historical leaf partials (154) and direct full graphs (176); endpoint selection is consistent. Historical candidate VJPs are added before one optimizer step. Complete saved rows show the intended selections and accounting. However the six direct/VJP comparisons, 203/203 cumulative nonzero gradients, frozen-start hashes, RNG/buffer equality and strict reload outputs are runtime witnesses. Independent CPU arithmetic validates their scalar consistency; no saved per-step parameter gradients or inputs exist here for independent reconstruction of the full gradient trajectory.',
  'evidence':['tools/train_msvr_cross_scene_smooth_ap.py:112-264','tools/msvr_freshness_probe.py:43-85','tools/probe_msvr_role_set_gradients.py:29-50','tools/probe_msvr_history_candidate_gradients.py:43-66','independent_m0.stdout'],
  'limits':'No dead objective was found on the M0 path. Q1 retrieval branches were read statically but are outside this performance audit. The one-group witness per capacity endpoint must not be promoted to all-step independent gradient validation.'},
 'E_scope':{
  'status':'PASS',
  'details':'Exactly one dataset and one seed, three folds by two capacity endpoints at 8 updates, plus fold0 two fixed-batch endpoints at 100 updates: 248 total. These are source-only engineering checks. M0 has zero heldout forwards and zero official image reads. The 780-batch T0/support text replay is a separate registered source sequence, not 780 model executions. M0 observed no zero-eligible batches; all nine zero-eligible support rows, including four after warmup, remain in the evidence.',
  'evidence':['refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_PLAN.md:11-24','refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_PLAN.md:34-50','tools/train_msvr_cross_scene_smooth_ap.py:325-381','provenance.stdout','independent_m0.stdout'],
  'limits':'No causal ranking gain, unseen-identity generalization, Q1 qualification, official result, multiple-seed robustness or SOTA claim is supported by M0.'},
 'F_evaluation_type':{
  'status':'PASS',
  'classifications':{'M0_training_masks_and_losses':'real_gt, source-only engineering','T0_saved_source_support_replay':'real_gt, label-only accounting','T0_numerical_fixtures':'simulation_only, synthetic deterministic unit fixtures','checkpoint_and_scalar_checks':'engineering consistency, not retrieval evaluation'},
  'details':'Dataset identities/scenes are the target source. Neither the baseline embedding nor model predictions generate GT. T0 synthetic arrays are explicitly numerical tests; no model-derived proxy is presented as benchmark performance.',
  'evidence':['tools/check_msvr_cross_scene_smooth_ap.py:15-70','tools/check_msvr_cross_scene_smooth_ap_math.py:12-91','tools/train_msvr_cross_scene_smooth_ap.py:108-139','t0.json in terminal_intake.stdout']}
}
limits=numeric['limitations']+[
 'The reviewer is a fresh context in the same model family. Requested route gpt-6-astra/max is recorded; backend model identity is not independently attested.',
 'The official test is not part of this audit. Image-content provenance and system-call-level proof of every historical read are outside the available saved evidence.',
 'Supplemental static text capture normalizes line endings; its stored raw-file SHA values refer to actual remote bytes. The initial intake retains exact decoded bytes for the six new source files.'
]
result=dict(audit_skill='experiment-audit',audit_status='CLOSED_WITH_LIMITS',verdict='WARN',overall_verdict='WARN',integrity_status='warn',
 reason_code='runtime_witness_reconstruction_limits',deterministic_checks_status='pass',m0_engineering_status='PASS',
 summary='Complete independent saved M0 arithmetic and checkpoint evidence pass. Runtime gradient/reload witnesses remain qualified; this is not a retrieval or Q1 scientific result.',
 date='2026-09-21',generated_at=datetime.now(timezone.utc).isoformat(),agent_id='/root/audit_cross_scene_smooth_ap_m0_20260921',
 verdict_id='/root/audit_cross_scene_smooth_ap_m0_20260921:M0:2026-09-21',reviewer_model='gpt-6-astra',reviewer_reasoning='max',
 reviewer_model_identity_basis='requested route; backend identity not independently attested',reviewer_family='openai',executor_family='openai',
 review_independence='same-family',acceptance_status='provisional',requested_fork_turns='none',
 trace_path=None,trace_handoff='Parent /root records the native invocation and complete reviewer response; this auditor does not edit project .aris.',
 repo=repo,run_dir=run,execution_commit=summary['project_commit'],final_observed_head=docs['head'],
 config_sha256=summary['config_sha256'],summary_sha256=numeric['summary_sha256'],m0_cpu_sha256=numeric['m0_cpu_sha256'],
 checks=checks,coverage={
  'registered_steps':248,'checked_steps':numeric['checked_steps'],'current_anchor_exposures':numeric['checked_current_anchor_exposures'],
  'distance_elements':numeric['checked_distance_elements'],'training_endpoints':8,'capacity_checkpoints':6,'baseline_checkpoints':3,
  'baseline_source_training_steps_checked':1950,'protocol_records':1032,'source_image_filenames':3096,'protocol_identities':155,
  'source_support_batches':780,'source_support_anchor_exposures':49920,'recursive_static_unique_files':119,'recursive_static_bindings':156,
  'additional_baseline_file_bindings':6,'direct_single_group_runtime_witnesses':6,
  'm0_zero_eligible_batches':0,'support_zero_eligible_batches_total':9,'support_zero_eligible_batches_post_warmup':4,
  'saved_objective_losses_checked':744,'weighted_component_scalars_checked':248*14,'current_branch_hard_scalars_checked':248*3,
  'independently_regenerated_parameter_gradients':0,'audit_model_forwards':0,'audit_backwards':0,'audit_optimizer_updates':0,'audit_image_content_reads':0},
 max_errors=numeric['max_errors'],runs=numeric['runs'],overfit=numeric['overfit'],direct_runtime_witnesses=numeric['direct_runtime_witnesses'],
 source_support_zero_rows=provenance['zero_eligible_batches'],checkpoints=numeric['checkpoints'],
 blocking_issues=[],warnings=limits,
 action_items=['Archive this report with raw receipts and parent review trace; update M0_Audit tracker to CLOSED_WITH_LIMITS/WARN after recording the verdict.',
               'Keep all six runtime direct/VJP witnesses and the zero-eligible source-support rows; do not relabel them as full independent gradient regeneration.',
               'Assess Q1 only after all registered endpoint, complete-gallery and independent terminal evidence exists; this report gives no Q1 verdict.'],
 claims=[{'id':'M0_248_step_engineering_completion','impact':'supported_with_runtime_witness_limits'},
         {'id':'complete_saved_distance_mask_objective_arithmetic','impact':'supported'},
         {'id':'all_step_parameter_gradient_independent_reconstruction','impact':'unsupported'},
         {'id':'unseen_identity_retrieval_gain_or_Q1_promotion','impact':'unsupported_by_M0'}],
 audited_input_hashes=hashes,raw_receipts=['intake_01.stdout','static_dependencies.stdout','static_bindings_receipt.json','provenance.stdout',
             'terminal_intake.stdout','independent_m0.stdout','ground_truth_disk.stdout','final_document_state.stdout','audit_helper_failure_01.json'],
 helper_incidents=['One local snapshot unpack path-length failure, corrected by flat numbered filenames without any experiment rerun. See audit_helper_failure_01.json.'],
 no_project_environment_process_edits=True,no_binary_downloads=True)
(root/'EXPERIMENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

lines=[
 '# MSVR310 cross-scene Smooth-AP M0 experiment audit',
 '',
 '**Overall verdict: WARN / CLOSED_WITH_LIMITS. Deterministic M0 engineering checks: PASS.**',
 '',
 'The full registered M0 completed, and independent remote CPU arithmetic reproduced every saved distance/mask/objective row and all six final capacity checkpoint states. No blocking arithmetic, provenance, scope, or missing-result defect was found. The warning records the remaining boundary: parameter-gradient trajectories, frozen-start state equality, and strict-reload outputs are runtime witnesses, not a fresh reconstruction of training.',
 '',
 'Fresh reviewer: `/root/audit_cross_scene_smooth_ap_m0_20260921`; requested `gpt-6-astra`, reasoning `max`, `fork_turns=none`. Review independence is **same-family**; acceptance is **provisional**. Backend model identity is not independently attested. The parent records the native invocation and complete response in its review trace.',
 '',
 f'Execution: `{summary["project_commit"]}`. Config SHA-256: `{summary["config_sha256"]}`. Remote run: `{run}`. Last documentation observation: `{docs["checked_at"]}`, HEAD `{docs["head"]}`. Source, config and protocol paths were unchanged from execution; subsequent changes were documentation/evidence.',
 '',
 '| Check | Verdict | Finding |',
 '|---|---|---|',
 '| A. Ground-truth provenance | PASS | Dataset labels and actual source filenames; all three complete source/heldout identity bindings checked. |',
 '| B. Score/reduction arithmetic | PASS | All 248 rows, all three fused objectives, eligible masks, branch hard losses and 14-term ledgers independently recomputed. |',
 '| C. Files and completion | PASS | M0/M0_CPU exited 0; complete 248-step files, six checkpoint payloads and final state hashes verified. |',
 '| D. Real execution paths | WARN | No dead objective found; runtime gradient/reload witnesses cannot be expanded into independent all-step gradient reconstruction. |',
 '| E. Scope | PASS | One dataset, one seed, source-only engineering; no retrieval qualification inferred. |',
 '| F. Evaluation type | PASS | `real_gt` source engineering and label accounting; T0 synthetic numerical fixtures are `simulation_only`. |',
 '',
 '## A. Ground-truth provenance and identity isolation',
 '',checks['A_ground_truth_provenance']['details'],
 '',
 'The GT chain is the source image filename → original MSVR310 loader identity/camera/scene parser → frozen label manifest → train-only protocol → source-index loader → current/history masks. The current directory contains exactly 1,032 filenames in each of vis/ni/th. The source IDs per fold are 103/103/104; heldout IDs are 52/52/51; source records are 672/683/709. Each fold starts from its own fixed epoch-50 Signal checkpoint and fresh seed-42 roles. The three baseline payloads and all 1,950 saved baseline training steps were checked for source isolation. No model output supplies a target label.',
 '',
 'Sources: `tools/audit_vehicle_query_protocol_labels.py:13-18`; `tools/build_msvr310_train_oof_protocol.py:12-55,57-123`; `tools/train_msvr310_signal_oof.py:68-99`; `tools/train_msvr310_trifusion_oof.py:27-55`. Raw independent evidence: `provenance.stdout`, `ground_truth_disk.stdout`, `independent_m0.stdout`. The historical install/CRC receipt was read as provenance, not rerun; no image contents or official-test directories were read.',
 '',
 '## B. Independent score, mask and denominator reconstruction',
 '',
 'The audit used standalone NumPy/stdlib arithmetic with no project objective/verifier imports. PyTorch was used only for CPU checkpoint deserialization. From every saved row it rebuilt FIFO membership, maximum age 8, capacity 512, current-record exclusion, identities/scenes, position self masks, standard positives, cross-scene positives, ignored same-ID/same-scene candidates, retained different-ID negatives, eligible counts, positive ranks and total ranks. It checked hard/standard/cross objectives and the selected loss before/after the two-step M0 warmup.',
 '',
 'Cross-scene loss is `1 - mean(AP over eligible anchors)`. Ineligible AP zeros are placeholders excluded from that mean. There is no prediction-max/min normalization. Each positive excludes its own comparison; same-scene different-identity negatives remain. Score conversion is fixed `1 - d^2/2`, tau 0.01, weight 1. The full intervention changes both relation masks and the eligible-anchor denominator, so its future effect cannot be attributed only to deleting easy positives.',
 '',
 '| Independent quantity | Maximum absolute error | Contract tolerance |',
 '|---|---:|---:|',
 f'| Standard per-anchor AP | {numeric["max_errors"]["standard_anchor_ap"]:.12g} | 2e-6 |',
 f'| Cross-scene per-anchor AP | {numeric["max_errors"]["cross_anchor_ap"]:.12g} | 2e-6 |',
 f'| Cross-scene scalar objective | {numeric["max_errors"]["cross_objective"]:.12g} | 2e-6 |',
 f'| Pooled hard objective | {numeric["max_errors"]["hard_objective"]:.12g} | 2e-6 |',
 f'| Three branch current hard losses | {numeric["max_errors"]["branch_current_hard"]:.12g} | 2e-6 |',
 f'| 14-term weighted ledger | {numeric["max_errors"]["fourteen_term_ledger"]:.12g} | 1e-5 |',
 '',
 'All 3,472 component scalars were checked for finite/nonnegative values and correct weighted summation. The fused objectives and three full-branch current hard losses were recomputed from distances. Classification logits and residual distances are not saved, so the ten remaining component scalars are ledger inputs rather than independently regenerated values.',
 '',
 'The fixed-batch overfit gate uses the registered analytic label-smoothing floor 0.5857136327437849. Its denominator is the initial loss minus that analytic floor; this is an explicitly named engineering ratio with raw losses, not a retrieval metric rescaled by its own predictions.',
 '',
 '| Endpoint | Initial loss | Final loss | Corrected excess-loss ratio |',
 '|---|---:|---:|---:|',
 *[f'| {r["endpoint"]} | {r["initial_loss"]:.15g} | {r["final_loss"]:.15g} | {r["corrected_loss_ratio"]:.15g} |' for r in numeric['overfit']],
 '',
 'Both ratios are below 0.1. Sources: `tools/msvr_cross_scene_smooth_ap.py:11-54`; `tools/msvr_smooth_ap.py:30-62`; `tools/run_signal_preserving_v5.py:99-135,1580-1617`; `independent_m0_remote.py`; `independent_m0.stdout`.',
 '',
 '## C. Complete terminal evidence',
 '',
 'The wrapper records T0 exit 0 at 00:25:36, M0 exit 0 at 00:36:52, and M0_CPU exit 0 at 00:37:03 on 2026-09-21 UTC+08. M0 status is `PASS_ENGINEERING_ONLY`; the complete project CPU verifier status is `PASS_COMPLETE_CROSS_SCENE_SMOOTH_AP_M0`. Independent checks consumed those artifacts without invoking that verifier. The current tracker agrees with those completed phases and correctly leaves M0_Audit running until this verdict is received.',
 '',
 f'- M0 summary SHA-256: `{numeric["summary_sha256"]}`.',
 f'- M0_CPU SHA-256: `{numeric["m0_cpu_sha256"]}`.',
 '- Six capacity runs × 8 updates + two fixed-batch overfit runs × 100 updates = **248/248** checked updates.',
 '- **15,872** current anchor exposures, **4,945,920** saved distance values, **744** fused-objective scalar definitions, **744** branch current-hard scalar checks.',
 '- Six capacity checkpoint hashes and payloads pass. Each full 472-tensor state reconstructs from 241 exact B0 aliases plus 231 saved role tensors, and its SHA equals the training/final-reload state record.',
 '- Eight training files and complete JSONL/distance pairs, six receipts, epoch means, actual selected objectives, paired sample/pixel hashes and overfit gates all agree. The two overfit paths have no final checkpoint by design; their final model-state record remains runtime evidence.',
 '- Recursive static intake checked 156 registered bindings across 119 files, plus six B0 checkpoint/array bindings. All matched. The initially observed running snapshot is preserved and never treated as terminal.',
 '',
 'The exact eight-run row/distance counts, all six checkpoint SHA values, file receipts and per-step numerical errors are in `independent_m0.stdout`. Source: `tools/train_msvr_cross_scene_smooth_ap.py:322-388`; `tools/train_msvr310_source_style.py:164-174`; `terminal_intake.stdout`; `final_document_state.stdout`.',
 '',
 '## D. Actual call paths and runtime-witness boundary',
 '',checks['D_real_call_paths']['details'],
 '',
 'All six direct comparisons occur at step 4, the first actual history group. Their largest recorded direct/VJP relative L2 error is **2.175816305864426e-5**, below 0.005. All six have finite scalar norm identities and consistent recorded decomposition; 5,760 historical VJP record-forwards are accounted for. This audit did not generate those gradients and does not describe these six witnesses as independently regenerated tensor comparisons. The 203/203 nonzero-gradient claim is cumulative over each run, not a claim that every parameter has a nonzero gradient on every step.',
 '',
 'Checkpoint state reconstruction is independent. Strict reload output equality, original frozen-state equality, runtime RNG/buffer preservation and the gradient path itself retain the stated witness limit. Sources: `tools/train_msvr_cross_scene_smooth_ap.py:129-176,179-264`; `tools/msvr_freshness_probe.py:43-85`; `tools/probe_msvr_role_set_gradients.py:29-50`; `tools/probe_msvr_history_candidate_gradients.py:43-66`.',
 '',
 '## E. Scope and complete zero-row accounting',
 '',
 'This is one-seed, one-dataset source-only engineering evidence. It does not measure heldout retrieval. The registered 780-batch source-support sequence was independently replayed from text labels and FIFO rules: 49,920 repeated anchor exposures. Post-warmup eligible exposures are 5,144 / 5,176 / 5,056 out of 12,480 per fold. All nine zero-eligible support batches are retained; four are after warmup: fold0 step180, fold1 step221, fold2 step133 and step232. The other five are warmup rows. The completed 248-step M0 contains **zero** zero-eligible batches, so its zero-batch graph behavior is supported by the explicitly synthetic T0 test and static call path, not by a claimed observed M0 zero case.',
 '',
 'The two overfit paths repeatedly use one fold0 batch and have no historical candidates because every stored record is in the current batch. Historical behavior is exercised by the six capacity paths. M0 records 0 heldout forwards and 0 official image reads. Candidate AP is not full-gallery mAP, and no performance, scientific promotion, robustness, or SOTA conclusion follows from engineering completion. The original failed Smooth-AP family remains unaltered. Sources: `EXPERIMENT_PLAN.md:11-24,34-50`; `provenance.stdout`; `independent_m0.stdout`.',
 '',
 '## F. Evaluation classification',
 '',
 '- M0 mask/loss supervision: **real_gt**, dataset-provided identities/scenes, source-only engineering.',
 '- Source-support/T0 label replay: **real_gt**, label-only candidate accounting, no model execution.',
 '- T0 finite-difference/tie/zero-eligible tests: **simulation_only**, explicitly synthetic numerical unit fixtures, not retrieval evidence.',
 '- Checkpoint hashes and saved scalar checks: engineering consistency; they are not a separate predictive evaluation.',
 '',
 'No model-generated reference is presented as real GT. Sources: `tools/check_msvr_cross_scene_smooth_ap.py:15-70`; `tools/check_msvr_cross_scene_smooth_ap_math.py:12-91`.',
 '',
 '## Limits, action items and audit execution record',
 '',
 'There are **no blocking issues** within the registered M0 arithmetic/checkpoint evidence. Keep the runtime limits attached to every engineering claim. Once this report and the parent trace are archived, update the M0_Audit tracker to `CLOSED_WITH_LIMITS / WARN`, with deterministic engineering PASS. Q1 requires its own complete endpoint/full-gallery and independent terminal evidence; this report contains no Q1 assessment.',
 '',
 'Audit execution performed zero model forwards, backward calls, optimizer updates and image-content reads. All weight/array reads and numerical work remained on remote CPU; no model was instantiated and no binary was downloaded. The actual 3,096 source filenames were enumerated without reading image contents. No project, master, Git, environment configuration or running process was edited by this reviewer.',
 '',
 'A local supplemental snapshot unpack initially failed on a deeply nested Windows path. The complete remote stdout already existed; flat numbered local filenames solved materialization. `audit_helper_failure_01.json` records the actual failure and correction. No experiment or model computation was repeated. All seven remote receipt stderr captures are empty.',
 '',
 'Core archivable evidence is listed, with sizes and SHA-256 values, in `audit_evidence_manifest.json`. Every entry is a text receipt, report, or auditor-written Python script. It excludes the private transport/credential helpers and all binary artifacts. The raw intake JSON files contain the source snapshots; included unpack helpers can materialize them. `independent_m0.stdout` preserves all 248 independent row results, including unfavorable values; no row was filtered.',
 ''
]
(root/'EXPERIMENT_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')

entries=[]
for path in sorted(root.iterdir()):
    if not path.is_file() or path.name in ('audit_evidence_manifest.json','REPORT_RECEIPT.json'):
        continue
    assert path.suffix in ('.py','.json','.md','.stdout','.stderr')
    if path.name.endswith('_remote.py'):
        kind='auditor_remote_read_only_payload'
    elif path.suffix=='.py':
        kind='auditor_local_text_materialization_or_report_script'
    elif path.name.startswith('EXPERIMENT_AUDIT'):
        kind='reviewer_verdict'
    elif path.name=='audit_helper_failure_01.json':
        kind='preserved_actual_local_helper_failure'
    else:
        kind='raw_text_capture_or_receipt'
    entries.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),kind=kind))
manifest=dict(schema='experiment-audit-text-evidence-manifest-v1',base_dir=str(root),entries=entries,
              total_files=len(entries),total_bytes=sum(r['bytes'] for r in entries),
              private_transport_helpers_included=False,binary_artifacts_included=False,
              materialized_snapshot_directories_included=False,
              snapshot_note='Raw intake JSON source texts are the archived originals; duplicate materialized snapshots are derivable with included local scripts.')
(root/'audit_evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (root/'EXPERIMENT_AUDIT.md',root/'EXPERIMENT_AUDIT.json',root/'audit_evidence_manifest.json')}
(root/'REPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(report_receipt=receipt,manifest_files=manifest['total_files'],manifest_bytes=manifest['total_bytes'],
                     verdict=result['verdict'],deterministic_checks_status=result['deterministic_checks_status'],blocking_issues=result['blocking_issues']),indent=2))
