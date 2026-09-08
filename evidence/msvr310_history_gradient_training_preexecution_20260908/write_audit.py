"""Write the review's structured artifact from independently checked local inputs."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

A=Path(__file__).resolve().parent
R=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
checks=json.loads((A/'independent_static_checks.json').read_bytes())
hashes=json.loads((A/'audited_input_hashes.json').read_bytes())
for name,entry in hashes.items():
    path=(A/name.removeprefix('AUDITOR/')) if name.startswith('AUDITOR/') else (R/name)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],name

sections={
 'A':dict(name='Ground truth provenance and complete-path isolation',status='WARN',
          evidence=['tools/train_msvr_history_gradient.py:26-40','tools/train_msvr310_source_style.py:25-51',
                    'tools/build_msvr310_train_oof_protocol.py:12-123','tools/train_msvr310_signal_oof.py:31-99,223-238'],
          details='Independent reconstruction of all 1032 dataset-label mappings, three identity partitions, local class maps, 600 query masks and full galleries passes. External Signal imports/checkpoints/images and remote execution remain untested here.'),
 'B':dict(name='Score and gradient normalization',status='PASS',
          evidence=['tools/probe_msvr_history_candidate_gradients.py:26-50','tools/msvr_instance_memory.py:40-80',
                    'tools/train_msvr_history_gradient.py:55-75,111-203','modeling/trifusion/signal_preserving_v8.py:264-339,483-653,690-742',
                    'tools/run_signal_preserving_v5.py:99-142'],
          details='Only history-candidate VJP addition differs. Current peers retain gradients and only 64 current anchors are averaged. Lambda1, FP32 upstream/accumulation, one scaled backward, one unscale and one AdamW step are consistent. All189 encoder tensors lie in the 42/54/93 role blocks; fusion has no trainable tensors and all14 other trainable tensors belong to ID-only neck/classifier paths.'),
 'C':dict(name='Artifact existence and contract/status correspondence',status='PASS',
          evidence=['configs/MSVR310/TriFusion-history-gradient-paired-v1.json','refine-logs/msvr310_history_gradient_v1/EXPERIMENT_TRACKER.md:3-14',
                    'tools/train_msvr_history_gradient.py:26-40','tools/run_msvr_history_gradient.py:14-65'],
          details='Final four-script/plan/config hashes agree. Tracker correctly records no new runtime results. Existing source diagnostic summary and CPU mirrors match exact pinned prerequisite hashes; they are not substituted for this implementation results. New scripts compile under independent Python3.10 grammar checks.'),
 'D':dict(name='Live calls and proof coverage',status='WARN',
          evidence=['tools/run_msvr_history_gradient.py:26-65','tools/train_msvr_history_gradient.py:127-203,292-337',
                    'tools/verify_msvr_history_gradient.py:17-134,156-290'],
          details='Every registered stage and both five-gate groups are live. First M0 step-label mismatch was resolved without source change. CPU recomputes fused Triplet/all13 statistics and weighted total from saved14 components; it does not regenerate all14 model component values or raw gradients. Direct real-model proof is one single-group189-encoder-tensor comparison per capacity arm.'),
 'E':dict(name='Scope and resources',status='WARN',
          evidence=['refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md:28-54','tools/train_msvr_history_gradient.py:79-90,225-238,251-339',
                    'tools/run_msvr_history_gradient.py:16-22,39-65'],
          details='All780 Q1 source batches, all24 capacity registry entries and fold0 fixed100 overfit indices independently replayed. Fixed seed42, original initialization, B64K8,20x13 steps,6 arms and complete galleries are preserved. Runtime capacity/speed/bitwise equality/gradients are pending; capacity eviction is not exercised by actual metadata schedules. Repeated internal development and one seed limit inference.'),
 'F':dict(name='Evaluation type classification',status='PASS',
          evidence=['tools/check_msvr_history_gradient.py:14-52','tools/probe_msvr_history_candidate_gradients.py:109-131',
                    'tools/train_msvr_history_gradient.py:284-338','tools/verify_msvr_history_gradient.py:283-290'],
          details='Planned dataset-label internal identity-OOF retrieval is real_gt. Synthetic tensor algebra is synthetic_proxy. M0 is source engineering evidence; per-step gradients/coordinates are runtime_model_witness; scalar/matrix/metadata reconstruction is saved-artifact consistency. No new measured training or retrieval outcome.')
}
unresolved=[
 'Remote external Signal-source/weight/image hashes and actual Python/PyTorch/CUDA/Mamba availability were not live-checked by the reviewer.',
 'New T0 synthetic functions and actual 248-step M0/CPU receipts have not run in this audit; previous fixed-state diagnostic results do not substitute.',
 'Actual grad/no-grad/checkpoint/autocast coordinate equality, RNG/buffer/current-grad preservation, finite/live gradients, direct error, memory capacity and paired pixels require runtime gates.',
 'Applied-gradient evidence covers pre-AdamW encoder inputs; it is not parameter-displacement verification or independent raw-gradient replay.',
 'Direct model proof is limited to first single-history group per capacity arm; no direct multiple-group graph is registered.',
 'Saved14 component scalars permit weighted-total reconstruction; the other13 losses are not independently regenerated from logits/embeddings.',
 'The actual source schedule has no capacity evictions and fold0 overfit excludes every historical duplicate.',
 'Complete Q1/strict reload/Signal parity/ranking/gates are pending; single seed and repeated internal development remain scientific limits.'
]
report=A/'reviewer_full_response.md'
audit=dict(
 audit_skill='experiment-audit',audit_stage='preexecution',
 verdict='WARN',overall_verdict='warn',integrity_status='warn',
 reason_code='static_ready_runtime_evidence_pending',
 summary='No blocking implementation defect remains in final inspected bytes. Enter the registered T0/M0 sequence; Q1 remains conditional on actual M0 and complete CPU receipts. This is not a runtime or scientific PASS.',
 preexecution_verdict='READY_FOR_REGISTERED_STAGED_EXECUTION_WITH_RUNTIME_GATES',
 deterministic_checks_status='pass',engineering_runtime_status='NOT_EXECUTED_BY_REVIEWER',scientific_status='NOT_MEASURED',
 agent_id='/root/audit_msvr_history_gradient_training',verdict_id='/root/audit_msvr_history_gradient_training/final',
 reviewer_backend='native_codex',reviewer_model='gpt-6-astra',reviewer_reasoning_effort='max',reviewer_reasoning='max',
 reviewer_family='openai',review_independence='same-family',acceptance_status='provisional',
 generated_at=datetime.now().astimezone().isoformat(),date='2026-09-08',
 project_dir=str(R),observed_checkout_commit=checks['project_commit'],
 trace_path=str(A),reviewer_full_response_path=str(report),reviewer_full_response_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
 audited_input_hashes=hashes,audited_input_hashes_path=str(A/'audited_input_hashes.json'),
 deterministic_evidence_path=str(A/'independent_static_checks.json'),
 primary_final_config_sha256=hashes['configs/MSVR310/TriFusion-history-gradient-paired-v1.json']['sha256'],
 primary_final_plan_sha256=hashes['refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md']['sha256'],
 initial_config_sha256='11c5fb22d7a95d2915a501a85972db79e8620276944b3dc0c58695550c3f6a00',
 initial_plan_sha256='aca0920c5bb58e691da75b77e2548386f8fd79fb535c931bc5fe2cf022c3ca2d',
 initial_snapshots=str(A/'inputs'),final_snapshots=str(A/'inputs_final'),
 checks=sections,
 resolved_findings=[dict(id='D1',severity='minor',status='RESOLVED',
                         problem='Initial plan said first M0 history at step3 without stating indexing basis; actual logged first history is step4, zero_based_step3.',
                         evidence=['refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md:39','tools/train_msvr_history_gradient.py:87,96,111,129,205-207,219'],
                         resolution='Final plan explicitly says logged step4 and zero_based_step3; bound plan SHA updated. Only the wording and matching configured digest changed; all four new scripts unchanged.',
                         independently_verified=True)],
 outstanding_blocking_findings=[],
 inherited_local_newline_cases=[c for c in checks['config_bindings'] if c['status']=='INHERITED_CRLF_VS_BOUND_LF'],
 untested_runtime_assumptions=unresolved,
 deterministic_counts=dict(input_files=len(hashes),python_sources=len(checks['syntax']),new_python_scripts=4,
                           project_binding_checks=67,exact_local_binding_checks=62,inherited_lf_bindings_with_local_crlf=5,
                           dataset_records=1032,folds=3,valid_queries=600,query_identities=60,
                           q1_source_batches=780,capacity_registry_batches=24,fixed_overfit_registry_batches=100),
 planned_runtime_counts=dict(t0_model_forwards=0,m0_optimizer_steps=248,q1_optimizer_steps=1560,q1_heldout_record_forwards=2064,official_image_reads=0),
 read_only_scope=dict(project_module_imports=0,model_instantiations=0,model_forwards=0,model_backwards=0,
                      optimizer_updates=0,image_reads=0,remote_commands=0,scientific_files_modified_by_reviewer=0),
 claims=[dict(id='static_single_gradient_scope_implementation',impact='supported_by_source_and_contract_checks'),
         dict(id='all14_saved_component_weighted_total_reconstruction',impact='implemented_static_only'),
         dict(id='independent_all14_model_loss_recomputation',impact='unsupported'),
         dict(id='new_m0_engineering_pass',impact='not_measured'),
         dict(id='new_q1_retrieval_gain',impact='not_measured'),
         dict(id='full_multiple_group_direct_model_gradient_proof',impact='not_registered_or_measured'),
         dict(id='sota_or_causal_old_failure_explanation',impact='unsupported')],
 action_items=[dict(when='before_execution',action='Bind future actual commit and final config SHA; perform live remote contract/resource checks and run the registered fixed T0/M0 stages.'),
               dict(when='before_q1',action='Require the actual same-config248-step M0 PASS and complete M0 CPU receipt, as implemented.'),
               dict(when='claims',action='Keep recorded runtime gradient/component-reassembly/single-group/single-seed boundaries and wait for complete Q1 CPU verification.')]
)
(A/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(verdict=audit['verdict'],preexecution_verdict=audit['preexecution_verdict'],
                     audit_path=str(A/'audit.json'),report_sha256=audit['reviewer_full_response_sha256'],
                     config_sha256=audit['primary_final_config_sha256'],blocking_findings=0),indent=2))
