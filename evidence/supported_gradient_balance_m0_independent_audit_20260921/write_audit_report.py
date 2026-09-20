"""Assemble the review from retained primary evidence and independent outputs."""
from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).parent
def load(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def sha(name): return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
proof=load('independent_recompute_02.json')
scope=load('audit_scope_census.json')
claims=load('local_bindings_claims.json')
correction=load('claim_correction_check.json')
R='snapshots/repo/'
X='remote_text/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'
M='remote_text/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/'
checks={
 'A':dict(name='Ground truth provenance',status='PASS',details='Source supervision uses registered official TRAIN identity/scene/camera metadata. Reconstructed all 1,032 triplet labels from filenames, all three identity-isolated folds and all 780 source batches/candidate masks. No model-generated labels or heldout/official images are used by M0.',evidence=[X+'tools/build_msvr310_train_oof_protocol.py:12-116',R+'tools/train_msvr310_signal_oof.py:68-99',R+'tools/msvr_cross_scene_smooth_ap.py:11-54','independent_recompute_02.py:72-150','independent_recompute_02.json:8-110']),
 'B':dict(name='Score normalization',status='PASS',details='Feature normalization and score=1-d^2/2 implement the stated cosine-space loss; no claimed retrieval result is divided by model prediction statistics. EMA norm ratios, relative derivative errors and the registered excess-loss ratio are explicitly engineering measures. Overfit denominator uses initial excess loss, with an analytic label-smoothing floor; it is not a retrieval score.',evidence=[X+'tools/msvr_role_set_relations.py:60-64',R+'tools/msvr_cross_scene_smooth_ap.py:26-54',R+'tools/msvr_supported_gradient_balance.py:23-43',X+'tools/run_signal_preserving_v5.py:1580-1618','independent_recompute_02.py:319-328']),
 'C':dict(name='Result existence, numbers and status',status='PASS',details='Complete evidence supports six 8-step capacity endpoints and two 100-step overfits, 248 updates, 90 saved component references, and six final capacity checkpoints. Rehashed all 37 M0 inventory files remotely and 28 local intake texts. All 744 role CSV rows, 90 reference rows and 8 epoch rows match raw records. The overbroad strict-reload wording was corrected and independently reread; original wording remains archived. R1 remains STOPPED_AT_M0 after 3 updates and failed step 4.',evidence=['independent_recompute_02.json:111-124','independent_recompute_02.json:879-1000','local_bindings_claims.json:1-25','claim_correction_check.json:1-11','corrected_claims/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:6752-6758',R+'evidence/supported_gradient_balance_m0_r1_failure_20260921/remote/m0.log:25-38']),
 'D':dict(name='Executed metric and update paths',status='PASS',details='Actual M0 calls the cross-scene objectives, direct current ranking and independent direct auxiliary derivatives, historical candidate VJPs, balanced/control combination, fixed reference checks, unscale and one AdamW step. Saved distance ledgers independently reproduce the claimed losses and support/EMA state. Retrieval functions are intentionally outside M0 and generate no M0 retrieval claim. Runtime gradient witnesses are not represented as independently regenerated gradients.',evidence=[R+'tools/train_msvr_supported_gradient_balance.py:110-299',R+'tools/msvr_supported_gradient_balance.py:46-106',R+'tools/train_msvr_supported_gradient_balance.py:358-442','independent_recompute_02.py:151-318','independent_recompute_02.json:2083-2130']),
 'E':dict(name='Scope and coverage',status='WARN',details='One dataset and one seed (MSVR310, seed42), with 3 folds, 6 short capacity runs and 2 fixed-batch overfits. Actual M0 has 16 warmup steps, 232 supported active steps and 0 active unsupported steps. Full source metadata predicts 4 active zero-support batches, but this is not M0 runtime coverage. 557 supported overfit role-steps have zero ranking norm; those are not unsupported steps. Parameter reference witnesses occur only at step 4 of each capacity endpoint. No scientific efficacy, robustness, benchmark superiority or promotion claim follows.',evidence=['independent_recompute_02.json:8-124','audit_scope_census.json','local_bindings_claims.json:26-209',R+'tools/train_msvr_supported_gradient_balance.py:185-216','independent_recompute_02.json:2107-2113']),
 'F':dict(name='Evaluation type classification',status='PASS',details='Mixed but labeled: real_gt source-training and real_gt metadata census; simulation_only synthetic T0 controller/loss tests; engineering_proxy direct-gradient/reload/frozen-state/overfit diagnostics. The direct reference is an internal derivative-equivalence reference, not ground-truth retrieval performance. M0 includes no new retrieval evaluation.',evidence=[R+'tools/check_msvr_supported_gradient_balance_math.py',X+'tools/check_msvr_cross_scene_smooth_ap_math.py:33-91',R+'tools/train_msvr_supported_gradient_balance.py:185-216',R+'tools/train_msvr_supported_gradient_balance.py:358-442','independent_recompute_02.json:2107-2130'])
}
input_hashes={
 'execution_commit':proof['execution_commit'],
 'registered_config':proof['config_sha256'],
 'm0_summary':proof['summary_sha256'],
 'm0_cpu_receipt':sha('snapshots/intake/m0_cpu.json'),
 'm0_cpu_receipt_local_rendered_copy':sha(M+'m0_cpu.json'),
 'audit_request':sha('001-request.md'),
 'remote_primary_intake':sha('remote_intake_02.json'),
 'independent_recompute_script':sha('independent_recompute_02.py'),
 'independent_recompute_output':sha('independent_recompute_02.json'),
 'corrected_master':correction['after_sha256']}
maxref=max(x['relative_l2_error'] for x in proof['reference_checks'] if x['relative_l2_error'] is not None)
zero_roles=sum(v['supported_zero_rank'] for end in claims['m0_additional_role_coverage'] for v in end['role_cases'].values())
report=dict(audit_skill='experiment-audit',verdict='WARN',overall_verdict='WARN',integrity_status='warn',
 reason_code='engineering_complete_with_runtime_and_scope_limits',
 summary='Complete M0 engineering evidence passes independently feasible deterministic checks. Scope/runtime limits remain; no retrieval or promotion conclusion.',
 deterministic_checks_status='pass',engineering_status='PASS_ENGINEERING_ONLY',closure_status='CLOSED_WITH_LIMITS',
 remaining_engineering_blockers=[],remaining_audit_blockers=[],
 scientific_claim_blockers=['Complete paired Q1 and its registered CPU checks and fresh terminal audit are outside this audit.','M0 cannot establish retrieval superiority, generalization, robustness or benchmark promotion.'],
 agent_id='/root/audit_supported_gradient_balance_m0_20260921',verdict_id='supported_balance_m0_20260921_fresh_audit_final',
 reviewer_session_id='01a0c0e2-1f03-7e23-b627-9bb872213eec',
 executor_family='openai',reviewer_family='openai',reviewer_model='gpt-6-astra',reviewer_reasoning='max',
 requested_reviewer_model='gpt-6-astra',requested_reasoning_effort='max',requested_fork_turns='none',
 reviewer_backend='native Codex agent',backend_attestation='unavailable',
 review_independence='same-family',acceptance_status='provisional',
 attribution_note='Model and reasoning fields record the requested routing; they are not independently attested backend identity. The fresh reviewer derived the verdict from primary artifacts. No additional reviewer agent was spawned.',
 trace_path=str(ROOT),request_path='001-request.md',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),date='2026-09-21',
 execution_commit=proof['execution_commit'],observed_publication_head=scope['observed_head'],audited_input_hashes=input_hashes,
 checks=checks,coverage=dict(dataset='MSVR310 official TRAIN split',seeds=[42],folds=3,
   source_records=1032,source_identities=155,multi_scene_identities=60,single_scene_identities=95,
   protocol_scene_labels=scope['protocol_scenes'],protocol_camera_labels=scope['protocol_cameras'],
   t0_metadata=proof['t0_counts'],m0=proof['m0_counts'],source_exposures=proof['source_exposures'],
   scalar_pair_identities=proof['scalar_pair_identities_checked'],saved_parameter_reference_comparisons=len(proof['reference_checks']),
   saved_reference_maximum_relative_error=maxref,supported_zero_rank_role_steps=zero_roles,
   final_capacity_checkpoints=6,overfit_final_checkpoints=0,baseline_checkpoints=3,
   raw_m0_inventory_files=37,local_raw_intake_files=28,local_raw_intake_bytes=7961437,
   registered_binding_rows=143,unique_registered_paths=125,execution_commit_blob_rows=97,
   numerical_maxima=proof['maxima'],training_costs=scope['recorded_training_costs']),
 deterministic_checks=['Registered source/config/Signal/CLIP/B0/final-checkpoint and intake hashes','All source identity/scene masks, queue age/capacity/current exclusion and deterministic fold assignment','All saved distance elements, standard/cross-scene AP and hard-loss scalars','All 14 weighted loss ledger entries, support counts, EMA and clipping/weight states for 248 updates','8,280 saved scalar norm/cosine identities and all 90 saved reference thresholds','Six capacity checkpoint tensor-content and frozen subset hashes, plus three B0 checkpoint hashes','All descriptive CSV fields/aggregates and corrected reload claim'],
 runtime_only=['Actual original image reads and augmentation pixel equality are code/counter/hash runtime witnesses; images were not reopened.','No saved full per-step parameter gradient vectors or AdamW optimizer states; these were not regenerated.','Historical reencoding bitwise equality, RNG/buffer restoration, classification head preservation and strict output reload are retained runtime assertions.','Raw step embeddings are not saved; four distance-space consistency does not prove a fresh embedding forward.','Both overfits have no terminal checkpoint or strict reload.'],
 no_new_work=dict(gpu_work=0,model_forwards=0,optimizer_updates=0,image_reads=0,q1_result_reads=0,tensor_array_downloads=0,repository_edits_by_reviewer=0),
 reviewer_cpu=dict(seconds=proof['elapsed_seconds'],threads=2,interop_threads=1,nice_increment=10,cuda_initialized=False,
    python=proof['python_version'],torch=proof['torch_version'],numpy=proof['numpy_version']),
 original_r1_failure=dict(status='STOPPED_AT_M0',completed_optimizer_steps=3,failed_step=4,elapsed_seconds=claims['r1_elapsed_seconds'],reference=claims['r1_failure'],execution_snapshot_files=10,archived_raw_files=8),
 resolved_findings=[dict(id='C-01',status='RESOLVED',finding='Original master wording could imply strict reload for all eight runs.',resolution='Only six capacity checkpoint reloads are now stated; the two overfit runs are explicitly excluded.',proof='claim_correction_check.json')],
 claims=[dict(id='M0_ENGINEERING',impact='supported',text='Registered R2 source-only M0 engineering checks and feasible independent saved-evidence recomputation pass.'),
         dict(id='FULL_PARAMETER_GRADIENT_REGENERATION',impact='unsupported',text='Saved scalar/reference checks do not establish independent reexecution of every parameter gradient or AdamW update.'),
         dict(id='REAL_M0_UNSUPPORTED_UPDATE',impact='unsupported',text='There are zero real active unsupported M0 steps.'),
         dict(id='RETRIEVAL_GAIN_OR_PROMOTION',impact='unsupported',text='No new retrieval efficacy or promotion result follows from M0.')],
 text_copy_provenance='text_copy_provenance.json',text_copy_note='The 154 remote_text files are Windows LF-to-CRLF text renderings, with separate local hashes. Exact raw M0 texts remain in snapshots/intake; raw remote CPU receipt SHA is 910c5c8493059cded0adb02bcbdb5ab0b701b40567b645f3a1ca37014feb21b1. All transformation checks pass; original mistaken drafts are retained.',
 failure_ledger='audit_failures.json',artifact_manifest='artifact_manifest.json',receipt='audit_receipt.json',verbatim_response='final_response.md')
(ROOT/'EXPERIMENT_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
table=[]
for row in scope['endpoints']:
    p=next(v for v in proof['training'] if v['endpoint']==row['endpoint'])
    g=p['gate'];ratio='—' if g is None else format(g['loss_ratio'],'.12g')
    refs=[v for v in proof['reference_checks'] if v['endpoint']==row['endpoint']]
    ref='—' if not refs else format(max(v['relative_l2_error'] for v in refs),'.10g')
    table.append(f"| {row['endpoint']} | {row['steps']} | {row['source_unique_records']} / {row['source_unique_identities']} / {len(row['source_scenes'])} | {p['counts']['warmup_steps']} / {p['counts']['supported_steps']} / {p['counts']['active_zero_support_steps']} | {p['counts']['distance_elements']:,} | {ref} | {ratio} | {row['peak_reserved_mib']:g} |")
md=f'''# MSVR310 supported gradient balance R2 — fresh M0 integrity audit

Date: 2026-09-21. Verdict: **WARN**. Integrity: **warn**. Engineering: **PASS_ENGINEERING_ONLY**. Deterministic checks: **pass**. Closure: **CLOSED_WITH_LIMITS**. Remaining M0 engineering/audit blockers: **none**.

The completed M0 supports the registered engineering claim only. This reviewer independently checked primary source, all saved M0 steps and remote CPU-feasible relationships. It does not establish a retrieval gain or promotion, and does not independently regenerate the original parameter gradients. The requested route was fresh-none / gpt-6-astra / max; this is **same-family / provisional**, with backend attestation **unavailable**. The reviewer is `/root/audit_supported_gradient_balance_m0_20260921`; no additional agent was used.

## Evidence and identity

Execution remains `{proof['execution_commit']}`. The observed documentation publication HEAD was `{scope['observed_head']}`; the 97 bound repository blobs still equal the execution commit. Config SHA-256: `{proof['config_sha256']}`. M0 summary SHA-256: `{proof['summary_sha256']}`. Original M0 CPU receipt SHA-256: `{input_hashes['m0_cpu_receipt']}`.

For portable path references below, **R** means `snapshots/repo/`, **X** means `remote_text/root/autodl-tmp/trifusion-v2/TriFusion-ReID/`, and **M** means `remote_text/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/`, all relative to this audit directory. These are retained copies, not unrecorded live files. `snapshot_inventory.json`, `remote_text_inventory.json` and the final `artifact_manifest.json` record the paths and hashes. `001-request.md` is the exact audit request. Source-file line numbers below refer to those copies.

The remote intake validated 143 registered binding rows over 125 unique paths, including the existing config dependency chain, Signal source commit/diff, CLIP weight, source metadata and B0 bindings; 97 repository paths were also compared to the execution Git blobs. The config itself was independently compared to the execution blob. No replacement provenance framework was imposed. The independent CPU script has no project imports, model construction, image access, forward passes or optimizer updates. It completed in {proof['elapsed_seconds']:.6f} s using two CPU threads, one interop thread, nice +10, CUDA disabled and uninitialized. All tensor, checkpoint and array files stayed remote. Local work used text and Python stdlib only. No repository edits or live Q1 metric reads were performed by this reviewer.

**Text-copy provenance correction:** `remote_text/` contains 154 local text renderings, not byte-exact copies: Windows `Path.write_text` converted LF to CRLF. Their line numbers and parsed content are unchanged; all 154 transformations were checked exactly. Raw M0 texts remain byte-exact under `snapshots/intake/`. A draft mislabeled the rendered CPU receipt hash as raw. Live remote byte verification confirms the raw receipt is 16,016 bytes/SHA `910c5c8493059cded0adb02bcbdb5ab0b701b40567b645f3a1ca37014feb21b1`; its rendering is 16,359 bytes/SHA `4f551ff62e64b5352fbd03398b3725c18ecee81b4d0f8ef8efff68013a6a5228`, with exactly 343 inserted carriage returns. `text_copy_provenance.json`, `verify_cpu_receipt_bytes.json` and `draft_failures/` retain the correction. Remote source/config/file bindings were hashed directly on remote bytes, not on these renderings.

## A–F findings

### A. Ground truth provenance — PASS

The ground truth is the registered **official TRAIN split** metadata, not model outputs. I rederived identity/scene/camera from the filenames of all three modalities for every one of the 1,032 triplet records; the metadata contains 155 identities (60 multi-scene, 95 single-scene), 30 scene labels and 8 camera labels. I rebuilt all three stratified identity partitions, source label maps, source/gallery records and the query eligibility masks. Source/heldout identities are disjoint in every fold. This inspects heldout metadata only; no heldout image was opened.

Fold 0/1/2 contain 103/103/104 source identities and 672/683/709 source records, respectively. Their heldout metadata contain 52/52/51 identities, 360/349/323 gallery records and 210/207/183 eligible queries. These counts define the protocol; they are not M0 retrieval evaluations. The full 260-batch source schedule per fold covers every source record. All 780 batches, 49,920 anchor exposures, 177,622 historical candidate occurrences and 107,608 cross-scene-positive positions were independently reconstructed from the immutable text metadata.

The queue has capacity 512, maximum age 8, latest occurrence per record, and excludes current record IDs. A legal positive has the same identity and a different scene; same-identity/same-scene positions are ignored and never relabeled as negatives. Different-identity negatives remain valid. Ineligible anchors contribute no cross-scene AP average, while their columns can remain negatives to eligible anchors. The all-ineligible loss is a connected zero.

Evidence: X `tools/build_msvr310_train_oof_protocol.py:12–116`; R `tools/train_msvr310_signal_oof.py:68–99`; R `tools/msvr_cross_scene_smooth_ap.py:11–54`; R `tools/msvr_instance_memory.py:8–37`; `independent_recompute_02.py:72–150`; `independent_recompute_02.json:8–110`.

### B. Score normalization — PASS

The loss uses normalized feature vectors and `score = 1 - distance² / 2`, with fixed temperature 0.01. This is the stated cosine-space training loss. It is not a normalization of a reported benchmark number by the model's maximum, mean or best output. The standard AP diagnostic, hard-triplet diagnostic and cross-scene AP term are distinguishable in every saved row. All actual loss scalars, including the original other thirteen terms, remain in the ledger.

Three forms of self-referential arithmetic are correctly limited to engineering use: the controller's EMA gradient-norm ratio, relative derivative-reference error, and the registered overfit excess-loss ratio `(last - analytic_floor)/(first - analytic_floor)`. The latter is not an AP or retrieval score. Its analytic floor is {proof['training'][-1]['gate']['minimum_loss']:.16g}, derived from 103-class label smoothing 0.1 and the fixed identity-loss weights; neither the floor nor the 0.1 gate was fitted after seeing results. Both endpoints started with loss 4.122129917144775. Control ended at 0.588193416595459 with ratio 0.0007012137860048812; balanced ended at 0.5882640480995178 with ratio 0.0007211864075456174. Both pass the unchanged engineering gate, and their difference does not establish superiority.

Evidence: X `tools/msvr_role_set_relations.py:60–64`; R `tools/msvr_cross_scene_smooth_ap.py:26–54`; R `tools/msvr_supported_gradient_balance.py:23–43`; X `tools/run_signal_preserving_v5.py:1580–1618`; `independent_recompute_02.py:319–328`.

### C. Result existence, numbers and state — PASS after one wording correction

The six capacity runs and two overfits all exist with complete logs and reports: **248 optimizer updates** in total, not a partial pilot relabeled complete. Each run records cumulative nonzero gradients for all 203 trainable tensors and zero overflow events. The 37 remote M0 inventory files were rehashed, including all eight distance arrays and six final capacity checkpoints. The 28 intake text files total 7,961,437 bytes and match their retained remote hashes. The published descriptive CSVs were compared field by field with the raw records: 744 role-step rows, 90 direct-reference rows and 8 stage-epoch rows. Those 8 rows are the actual M0 stages, not the planned Q1 epochs.

Six capacity checkpoints were loaded on remote CPU. Their role states plus pinned baseline aliases reconstruct the exact saved final state hashes, strict-reload state hashes and frozen subset hashes. Three B0 checkpoint files, source/heldout identity bindings and saved B0 full-state hashes were independently checked; their registered retrieval-array hashes were checked without loading retrieval scores. There are **189 role encoder tensors and 14 neck/head tensors**. The two overfit runs intentionally have no final checkpoint and no strict reload; their final state and freeze claims are saved runtime witnesses only.

Finding **C-01**: the originally published master §41.233:6754 placed “strict reload passed” after language about all eight runs, which could overstate reload coverage. The executor corrected it to explicitly name six capacity checkpoints and explicitly exclude both overfits. I independently reread the current file and verified the exact two documented text replacements against the archived prior bytes. Original and corrected snapshots remain retained. Corrected master SHA-256: `{correction['after_sha256']}`. `claim_correction_check.json` records closure. An accompanying opening-overview correction concerns an earlier sealed experiment; its retrieval result is not reaudited or used as M0 evidence here.

Original R1 remains a failure: 3 completed optimizer updates; step 4 failed before its optimizer update; relative auxiliary-reference error **0.006824872357540746 > 0.005**; exit 1 and `STOPPED_AT_M0`; elapsed 37.34809142164886 s. The stale initial `RUNNING` M0 summary is not evidence of a live or complete R1. Eight raw R1 text artifacts and ten execution snapshots were verified against the retained inventory and R1 Git blobs. R2 did not resume those three updates. Original failed work is not erased by R2 completion.

Evidence: `local_bindings_claims.json:1–25`; `independent_recompute_02.json:111–124,879–1000,2098–2106`; R `tools/train_msvr_supported_gradient_balance.py:395–440`; R `evidence/supported_gradient_balance_m0_r1_failure_20260921/remote/m0.log:25–38`; the same archive's `remote/pipeline.json:2,42–44`; `claim_correction_check.json`; `corrected_claims/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:6752–6758`.

### D. Executed paths and R1/R2 distinction — PASS

The training path actually calls the loss functions and writes their results. It constructs current and historical distances in the same metric coordinates, applies the active cross-scene fused term, and preserves the other thirteen loss terms. In R2 the auxiliary encoder derivative is a **separate direct call over those thirteen terms**, not `current_total - current_rank`. Every endpoint incurs that call. Both endpoints retain the same current-coordinate historical replay and candidate VJP algorithm. The full ranking block is current ranking plus historical ranking; historical rows are candidates only, never anchors.

For supported balanced updates, the actual encoder gradient is the weighted sum of direct full ranking and direct auxiliary gradients. Control, warmup and unsupported updates use the original `current_total + history` path. The scalar verifier respects this finite-precision distinction. The old subtraction-versus-direct and direct-sum-versus-original discrepancies are retained as diagnostics, not forced to zero or repurposed as the passed reference. This preserves the real R1 failure instead of relaxing the 0.005 gate.

At the first historical-group reference of each capacity endpoint (step 4), the code independently encodes the selected historical group and forms current, historical, full-ranking, auxiliary and applied references for all three roles: 6 × 3 × 5 = **90 comparisons**. Auxiliary is called again for its reference; control applied is compared with the original direct total objective, whereas supported balanced applied is compared with the independent weighted direct parts. All actual gradients use the same AMP scale, unscale once, and perform one AdamW step. Reference tensors stay alive until after checks and are then released. The actual neck/head gradients are copied from the original total derivative and checked unchanged. The 90 saved reference comparisons all satisfy the original relative gate 0.005 (zero reference absolute gate 1e-8); maximum observed relative error is **{maxref:.16g}**. These are runtime witnesses whose scalar relationships I checked, not an independent regeneration of the 90 gradient vectors.

I recomputed all **4,945,920 float32 distance elements**, offsets, shapes, finite checks and EOFs in fused/CNN/Transformer/Mamba spaces; complete per-anchor standard/cross-scene AP and hard objectives; 248 × 14 component ledgers; support/EMA/clipping/weight states; and **8,280 norm/cosine/difference scalar identities**, including all 744 saved AdamW parameter-update norm pairs. The final capacity checkpoint parameter norms also agree with the final saved update endpoints within cross-device reduction precision. AdamW delta norms do not decompose the step into a ranking “share”: momentum, variance and weight decay prevent that interpretation.

Maximum observed errors: standard AP 2.2182630954326044e-7; cross-scene AP 2.4798321118790767e-7; hard objective 2.9802322387695312e-8; weighted total loss 5.45134147600379e-7. Current-distance symmetry residual is 5.21540641784668e-7 and fused squared distance versus mean branch squared distance residual is 1.2031691151159762e-6. Those last two are recorded diagnostics, not newly invented pass gates. Raw step embeddings are not saved, so distance consistency is not a fresh encoder-forward proof.

The controller parameters remain EMA 0.9, exponent 0.5, epsilon 1e-12, ratio bounds [0.25,4], and weight sum 2 with individual weights [0.4,1.6]. Warmup leaves EMA untouched; actual supported steps update it even when ranking norm is exactly zero. Unsupported active behavior has code and synthetic coverage, but no actual M0 instance (see E).

History replay restores the original role-entry RNG in a fork and restores buffers; fresh reencoding, source-pixel equality, no change to current gradients during historical VJPs, frozen baseline, head preservation and strict five-output reload checks are evidenced by the saved runtime assertions and their real call sites. No new forwards were authorized, so I did not reexecute these tensor assertions. The retrieval function exists for Q1 and is intentionally bypassed in M0; absence of M0 retrieval output is consistent with the claimed engineering scope, not fabricated invocation.

Evidence: R `tools/train_msvr_supported_gradient_balance.py:44–54,110–154,155–181,185–299,303–347,358–442`; R `tools/msvr_supported_gradient_balance.py:46–106`; R `tools/probe_msvr_role_set_gradients.py:29–50`; R `tools/msvr_freshness_probe.py:43–85`; R `tools/train_msvr310_trifusion_oof.py:27–73,187–241`; X `tools/train_msvr_instance_memory.py:94–111`; `r1_r2.diff:245–320`; `independent_recompute_02.py:151–382`; `independent_recompute_02.json:2083–2130`.

### E. Scope and coverage — WARN

This is **one dataset, one seed (42), two fixed update rules and three predefined identity folds**, with short capacity runs and fixed-batch overfits. M0 observed source records across 30 scene labels, but did not evaluate retrieval across them. No scientific claim of comprehensive effectiveness, robustness, multiple-seed evidence, multi-dataset generalization or official benchmark improvement is supported.

| Run | Updates | Unique records / identities / scene labels | Warmup / supported / active unsupported | Distance elements | Maximum saved reference relative error | Overfit excess-loss ratio | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Actual M0 coverage is **16 warmup + 232 supported active + 0 active unsupported steps**. There are 9,296 eligible anchor exposures (including warmup), 15,872 current record exposures, 30 historical-candidate steps, 3,448 historical candidate occurrences and 90 replay groups. The full source metadata census has 9 zero-eligible batches, including 4 after its 65-step warmup: fold 0 step 180, fold 1 step 221, fold 2 steps 133 and 232. The remaining 5 are warmup batches. These 1-based positions establish future schedule opportunities, not actual M0 or Q1 runtime execution of the branch.

The overfits include **{zero_roles} supported role-step cases with exactly zero ranking norm** (control 279; balanced 278), among 588 supported overfit role-step cases. These remain supported and update the EMA; they cannot be counted as unsupported/no-EMA-update tests. No capacity role has zero ranking norm. Proposed ratios hit the upper bound during overfit only: control CNN/Transformer/Mamba 91/52/62 times, balanced 93/54/62; no lower-bound hits. Control still applies unit weights. These trajectories are overfit diagnostics and provide no basis to tune Q1 coefficients.

Parameter-reference runtime coverage is six capacity steps, not every one of the 248 updates, and no direct historical reference exists in the fixed-batch overfits because they have no history. The original 203/203 condition is cumulative per run, not a claim that every tensor has a nonzero derivative on every step. No step gradient vectors or optimizer states were retained, so full parameter-gradient and AdamW reexecution is unavailable within this evidence. Strict checkpoint content can be independently checked for six capacity endpoints; strict forward-output equality remains a saved runtime witness.

Evidence: `audit_scope_census.json`; `independent_recompute_02.json:8–124,2107–2113`; `local_bindings_claims.json:26–209`; R `tools/train_msvr_supported_gradient_balance.py:185–216,435–440`.

### F. Evaluation type — PASS, mixed types explicitly separated

| Evidence | Classification | Permitted meaning |
|---|---|---|
| Source supervision and source queue/mask census | `real_gt` | Uses registered dataset identity/scene labels; metadata coverage is not retrieval effectiveness. |
| T0 controller/loss/AMP/permutation finite fixtures | `simulation_only` | Mathematical and implementation sanity checks with no model training or dataset image read. |
| Direct derivative references, frozen state, reload and saved scalar identities | `engineering_proxy` (internal model/derivative reference) | Tests algorithm/state consistency; the internal reference is not a benchmark ground truth. |
| Fixed-batch overfit | `real_gt` supervised fitting plus `engineering_proxy` loss-ratio gate | Shows the registered training path can fit the source batch; it is not a generalization score. |
| Retrieval effectiveness or official benchmark | Not performed by M0 | No new such claim is supported. |

Evidence: R `tools/check_msvr_supported_gradient_balance_math.py`; X `tools/check_msvr_cross_scene_smooth_ap_math.py:33–91`; R `tools/train_msvr_supported_gradient_balance.py:185–216,358–442`; `independent_recompute_02.json:2107–2130`.

## Actual work and failed attempts retained

R2 original M0 took 1016.5666994191706 s and its original CPU verifier 9.100542893633246 s. This independent remote CPU pass took {proof['elapsed_seconds']:.6f} s; it is a separate execution, not the original verifier's green flag. All 248 original current-total backward calls are accompanied by 248 current-ranking derivative calls and **248 direct-auxiliary derivative calls**. Six capacity reference steps add 24 direct component calls and six direct-total reference calls. Saved forward counters total 6,272 extra fresh role record forwards (5,760 history refresh + 512 zero-age replay), 5,760 history-VJP record forwards and 384 direct-check record forwards. These counters exclude initialization, unchanged current forward/backward, preflight and strict reload; they are not a wall-clock cost decomposition. Peak reserved memory is 12,186 MiB, below the unchanged 24 GiB engineering limit.

The original R1's three updates and failed fourth step remain separate spent work. The failed fourth step reached its gradient reference check, so counting only completed optimizer updates would understate its cost. No Q1 runtime or metric was used to select or modify this M0 audit.

Auditor failures are retained in `audit_failures.json`, their original scripts, raw outputs and request receipts. An initial transport failed before remote connection because the selected local stdlib runtime had no Paramiko; the retry used the existing offline uv cache. The first independent CPU script demanded exact CPU/GPU double-reduction equality for a final role parameter norm. Its saved failure and follow-up diagnostic show a maximum absolute difference of 5.684341886080802e-14 (relative 2.0207945872648634e-16), with exact checkpoint content hashes already matching. The revised script only applies the already-existing scalar consistency tolerance to this added cross-device comparison. The experimental 0.005/1e-8 reference gates, 0.1 overfit gate and all thresholds in source/config were unchanged. A local inventory-key assumption and inline PowerShell quoting failure were also preserved; they had no effect on remote training or experimental data.

## Closure, blockers and allowed wording

**CLOSED_WITH_LIMITS** is the terminal audit state. No M0 engineering defect or missing required M0 evidence remains within the authorized read-only CPU audit. C-01 is resolved. The A/B/C/D/F statuses are PASS and E is WARN; overall WARN preserves the coverage and assurance limits rather than treating them as missing engineering work.

Supported wording: “For MSVR310 official TRAIN, seed42 and the registered R2 configuration, all six short capacity runs and two fixed-batch overfits passed M0 engineering gates; an independent bounded-CPU review reproduced all saved-distance, mask, loss-ledger and scalar-state checks. Six capacity checkpoints have verified saved content and recorded strict-reload checks.”

Required qualifiers: no real active unsupported M0 update occurred; no saved per-step parameter-gradient/optimizer replay was independently regenerated; runtime tensor assertions remain runtime witnesses; the two overfits have no final checkpoint/reload; same-family reviewer acceptance is provisional and the requested backend is not independently attested.

Unsupported wording: “The audit independently reproduced all training gradients”, “M0 covered real no-support updates”, “all eight runs passed strict reload”, “the candidate improves retrieval”, or any Q1/scientific promotion/official benchmark claim. Completing Q1's original fixed endpoints, full retrieval gallery, registered gates and terminal independent audit remains outside this audit. No new run or threshold change is requested merely to remove these honest scope limits.

Machine-readable verdict: `EXPERIMENT_AUDIT.json`. Full archive manifest: `artifact_manifest.json`. Receipt: `audit_receipt.json`. Verbatim final response: `final_response.md`.
'''
(ROOT/'AUDIT_DETAILS.md').write_text(md,encoding='utf-8')
concise=f'''# Supported gradient balance R2 — fresh M0 audit

**WARN / CLOSED_WITH_LIMITS**. Deterministic checks: **PASS**. Engineering: **PASS_ENGINEERING_ONLY**. Remaining M0 engineering/audit blockers: **none**. Date: 2026-09-21.

Requested reviewer: fresh-none / gpt-6-astra / max. Attribution: **same-family / provisional; backend attestation unavailable**. Reviewer: `/root/audit_supported_gradient_balance_m0_20260921`. The verdict was derived from primary artifacts; no additional agent or new model run was used.

Execution `{proof['execution_commit']}`; config SHA `{proof['config_sha256']}`; M0 summary SHA `{proof['summary_sha256']}`. The observed publication HEAD `dd6a097` changed text evidence, while all 97 execution-bound repository files still matched the execution commit. All 143 registered binding rows over 125 unique paths matched. See `remote_intake_02.json`, `audit_scope_census.json:1–12` and the existing source/config/checkpoint binding chain; no new provenance framework was imposed.

In the references below, **R** = `snapshots/repo/`, **X** = `remote_text/root/autodl-tmp/trifusion-v2/TriFusion-ReID/`, **M** = `remote_text/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/`. All are retained within this audit directory. X/M are local LF-to-CRLF text renderings with their own hashes; raw M0 texts are byte-exact under `snapshots/intake/`. Exact file-level hashes are in `artifact_manifest.json`; full findings and additional line references are in `AUDIT_DETAILS.md` and `EXPERIMENT_AUDIT.json`.

Raw original CPU receipt SHA is **`{input_hashes['m0_cpu_receipt']}`** (16,016 bytes). Its rendered local copy SHA is `4f551ff62e64b5352fbd03398b3725c18ecee81b4d0f8ef8efff68013a6a5228` (16,359 bytes). Independent live remote hashing, byte-exact intake and embedded UTF-8 agree; the difference is exactly 343 inserted CR bytes and parsed JSON is identical. All 154 rendered-copy transformations were checked. The mistaken draft hash label and original drafts are retained as AF6; see `text_copy_provenance.json` and `verify_cpu_receipt_bytes.json`.

## A–F

| Check | Verdict | Finding and exact evidence |
|---|---|---|
| A — Ground truth | PASS | Dataset TRAIN identity/scene/camera labels, no generated labels. Independently reconstructed all 1,032 triplets, 155 identities, three isolated source/heldout folds and all 780 source batches/masks. X `tools/build_msvr310_train_oof_protocol.py:12–116`; R `tools/msvr_cross_scene_smooth_ap.py:11–54`; `independent_recompute_02.py:72–150`; result `independent_recompute_02.json:8–110`. |
| B — Normalization | PASS | `1-d²/2` is the stated normalized-feature loss score. EMA norm ratios, relative gradient error and the registered excess-loss ratio are engineering measures, not retrieval metrics. The analytic smoothing floor and original gates are unchanged. X `tools/msvr_role_set_relations.py:60–64`; X `tools/run_signal_preserving_v5.py:1580–1618`; R `tools/msvr_supported_gradient_balance.py:23–43`; `independent_recompute_02.py:319–328`. |
| C — Results/status | PASS after correction | Six 8-step capacities + two 100-step overfits = 248 updates; all 37 M0 files remotely rehashed; 28 intake texts/7,961,437 bytes matched; every field of 744 role rows, 90 reference rows and 8 epoch rows verified. Six capacity checkpoints, no overfit checkpoints/reload. Original R1 remains failed. `local_bindings_claims.json:1–25`; `independent_recompute_02.json:879–1000`; R R1 archive `remote/m0.log:25–38`; `claim_correction_check.json`. |
| D — Executed paths | PASS | Actual cross-scene loss, direct auxiliary, full current/history ranking, control/balanced combination, references, one unscale and one AdamW update are connected. All saved distances and scalar state relationships independently recomputed. Retrieval is intentionally outside M0. R `tools/train_msvr_supported_gradient_balance.py:110–299,358–442`; R `tools/msvr_supported_gradient_balance.py:46–106`; `independent_recompute_02.py:151–382`. |
| E — Scope | WARN | MSVR310, seed42 only. Actual M0 = 16 warmup + 232 supported + **0 active unsupported** steps; direct references at six capacity step-4 positions only. No independent parameter-gradient/optimizer replay or new retrieval. `independent_recompute_02.json:111–124,2107–2113`; `audit_scope_census.json`; `local_bindings_claims.json:26–209`. |
| F — Evaluation type | PASS | Source training/masks: `real_gt`; synthetic T0: `simulation_only`; derivative/reload/frozen-state/overfit-ratio checks: `engineering_proxy`. Internal derivative references are not retrieval GT. X `tools/check_msvr_cross_scene_smooth_ap_math.py:33–91`; R `tools/train_msvr_supported_gradient_balance.py:185–216,358–442`. |

## Complete coverage and numerical evidence

Independent remote CPU terminal result: `PASS_INDEPENDENT_COMPLETE_M0_SAVED_EVIDENCE`, **{proof['elapsed_seconds']:.6f} seconds**, CPU threads 2 / interop 1, nice +10, CUDA uninitialized. Script: `independent_recompute_02.py`; exact request, raw stdout/stderr and transport outputs are retained alongside `independent_recompute_02.json`.

- All **4,945,920** saved float32 distance elements across four spaces, all **248 × 14** component ledgers, support/EMA/weights and **8,280** norm/cosine identities checked. Maximum cross-scene AP error 2.4798321118790767e-7; weighted-loss error 5.45134147600379e-7.
- All **90** saved current/history/full ranking, auxiliary and applied references meet unchanged 0.005 relative / 1e-8 zero-reference gates; maximum relative error **{maxref:.16g}**. These are checked saved runtime witnesses, not regenerated gradient vectors.
- All eight runs record cumulative **203/203** nonzero trainable tensors and **0 overflow**; 189 encoder tensors are balanced and 14 neck/head tensors preserve the original total derivative. Six capacity checkpoint contents and frozen-state subsets plus three B0 states were independently hashed on CPU. Strict forward-output equality is a saved runtime assertion. Peak reserved memory: **12,186 MiB**.
- Both overfits pass unchanged excess-loss ratio ≤0.1: control **0.0007012137860048812**, balanced **0.0007211864075456174**, with analytic floor **0.5857136327437849**. These are engineering fit checks, not retrieval comparisons.
- Full source census: **780 batches, 49,920 anchor exposures, 177,622 historical candidates, 107,608 cross-scene-positive positions**. Four active zero-support schedule positions exist (fold 0:180, fold 1:221, fold 2:133/232), plus five warmup positions. They are metadata evidence, not actual M0/Q1 branch coverage.
- Actual M0: **15,872 current exposures; 9,296 eligible anchor exposures; 30 history steps; 3,448 historical candidates; 90 replay groups**. Overfits contain **557 supported role-step cases with zero ranking norm**; support is still true and EMA still updates. Those must not be relabeled unsupported steps.

## R1/R2, spent work and corrected wording

R1 completed **3** updates and failed step **4 before its optimizer update**, relative auxiliary error **0.006824872357540746 > 0.005**, exit 1 / `STOPPED_AT_M0`, elapsed **37.34809142164886 s**. Eight original texts and ten execution snapshots remain verified and retained. R2 did not resume them.

R2 directly differentiates the original other thirteen terms for auxiliary gradients on both endpoints. Supported balanced updates combine weighted direct auxiliary with full current+history ranking; control/warmup/unsupported retain original `current_total+history`. The finite-precision subtraction and direct-sum discrepancies remain diagnostics; the failed R1 identity is not silently made a gate. References survive until checked, use consistent AMP scale, and head gradients remain original. Evidence: R trainer `155–299`; R combination code `65–96`; `r1_r2.diff:245–320`.

Recorded R2 costs: **248 ranking + 248 auxiliary derivative calls**, **24 direct component + 6 direct-total reference calls**, **6,272 fresh role record forwards**, **5,760 history-VJP record forwards**, **384 direct-reference forwards**. These exclude initialization/preflight/strict reload and do not assign wall time or an AdamW update share to ranking. Original M0 wall time was **1016.5666994191706 s**; original CPU verifier **9.100542893633246 s**.

**C-01 resolved:** original master §41.233:6754 could imply all eight runs passed strict reload. The corrected text explicitly limits it to the six capacity checkpoints and excludes both overfits. Independent byte comparison confirms only the two documented text replacements; original and revised versions remain in `latest_claims/` and `corrected_claims/`. Corrected SHA: `{correction['after_sha256']}`. The accompanying overview edit concerns an earlier sealed experiment and does not supply M0 retrieval evidence.

Auditor tooling failures remain in `audit_failures.json` with scripts/requests/outputs. The first CPU pass used an overly strict exact CPU/GPU reduction comparison. The diagnostic found maximum norm discrepancy **5.684341886080802e-14** with exact checkpoint content hashes matching. The revised audit uses the existing scalar consistency tolerance for that added comparison only; **no experimental gate was changed**. Transport, inventory-key and quoting failures are also retained.

## Closure and claim boundary

**CLOSED_WITH_LIMITS; no remaining M0 engineering or audit blocker.** A/B/C/D/F PASS, E WARN. Supported: the registered source-only R2 M0 engineering gates and all CPU-feasible saved-evidence checks pass. Required limits: no actual active unsupported M0 update; no independent per-step gradient/AdamW regeneration; RNG/buffer/head/pixel/strict-output checks remain original runtime witnesses; overfits have no final checkpoint/reload; reviewer acceptance remains same-family/provisional without backend attestation.

No new model forward, optimizer step, image access, GPU work, tensor/array download, live Q1 metric read or reviewer repository edit occurred. M0 does **not** establish retrieval superiority, robustness, generalization, official benchmark performance or promotion. Q1's full fixed-endpoint results, registered checks and terminal audit remain outside this audit; no additional run or gate change is requested to erase the stated limits.

Reports: `EXPERIMENT_AUDIT.md`, `EXPERIMENT_AUDIT.json`, `AUDIT_DETAILS.md`. Verbatim response: `final_response.md`. Hash receipt: `audit_receipt.json`. Full UTF-8 inventory: `artifact_manifest.json` plus `artifact_manifest.sha256`.
'''
(ROOT/'EXPERIMENT_AUDIT.md').write_text(concise,encoding='utf-8')
print(json.dumps(dict(status='REPORTS_WRITTEN',verdict='WARN',closure_status='CLOSED_WITH_LIMITS',
    markdown_sha256=sha('EXPERIMENT_AUDIT.md'),json_sha256=sha('EXPERIMENT_AUDIT.json')),indent=2))
