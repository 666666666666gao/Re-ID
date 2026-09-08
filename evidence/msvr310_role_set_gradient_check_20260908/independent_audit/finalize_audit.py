"""Package report provenance and retained failed audit-harness attempts locally."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')


def sha(data):
    return hashlib.sha256(data).hexdigest()


local = json.loads((OUT/'local_verification.json').read_bytes())
remote = json.loads((OUT/'remote_verification.json').read_bytes())
synthetic = json.loads((OUT/'remote_math_verification.json').read_bytes())
assert local['status'].startswith('PASS_') and remote['status'].startswith('PASS_') and synthetic['status'].startswith('PASS_')
supplemental = []
items = [(Path('C:/Users/gb/.codex_tmp/role_set_training_storage_inventory_20260908.json'),
          OUT/'snapshots/supplemental/role_set_training_storage_inventory_20260908.json')]
for name in ['experiment-audit/SKILL.md','shared-references/local-codex-policy.md',
             'shared-references/reviewer-independence.md','shared-references/experiment-integrity.md']:
    items.append((Path('C:/Users/gb/.codex/skills')/name,OUT/'snapshots/skills'/name))
for name in ['results/MSVR310_ROLE_SET_GRADIENT_CHECK_2026-09-08.md',
             'refine-logs/msvr310_role_set_v1/TRAINING_IMPLEMENTATION_DRAFT.md']:
    items.append((ROOT/name,OUT/'final_claim_snapshots/repo'/name))
for path,target in items:
    data=path.read_bytes()
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
    supplemental.append({'input':str(path),'snapshot':str(target),'sha256':sha(data),'bytes':len(data)})
(OUT/'supplemental_inputs.json').write_text(json.dumps(supplemental,indent=2)+'\n',encoding='utf-8')

failed=OUT/'failed_attempts'
failed.mkdir(exist_ok=True)
first_transport='''"""Use the authorized private SSH session without printing its contents."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import io
import sys

OUT = Path(__file__).resolve().parent
helper = Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py')
namespace = {'__name__': 'independent_audit_transport'}
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    exec(compile(helper.read_text(encoding='utf-8'), str(helper), 'exec'), namespace)
c = namespace['c']
stdin, stdout, stderr = c.exec_command('CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /root/miniconda3/envs/tri_reid/bin/python -B -u -', timeout=300)
stdin.write((OUT / 'remote_verify.py').read_text(encoding='utf-8'))
stdin.channel.shutdown_write()
result = stdout.read()
errors = stderr.read()
exit_code = stdout.channel.recv_exit_status()
(OUT / 'remote_verification.json').write_bytes(result)
(OUT / 'remote_verification.stderr.txt').write_bytes(errors)
(OUT / 'remote_exit_code.txt').write_text(str(exit_code)+'\\n', encoding='utf-8')
c.close()
print('Read-only remote audit exit code:', exit_code)
print('Result bytes:', len(result), 'stderr bytes:', len(errors))
sys.exit(exit_code)
'''
(failed/'attempt1_remote_transport.py').write_text(first_transport,encoding='utf-8')
first_remote=(OUT/'remote_verify.py').read_text(encoding='utf-8').replace(
    "history_cfg = read(absolute(spec['previous_config']))\ncoordinate_cfg = read(absolute(history_cfg['coordinate_config']))\nmemory_cfg = read(absolute(coordinate_cfg['memory_config']))\nbase_cfg = read(absolute(memory_cfg['base_config']))",
    "base_cfgs = [v for v in configs.values() if 'BASELINE' in v and 'SOURCE_METADATA' in v]\nassert len(base_cfgs) == 1\nbase_cfg = base_cfgs[0]")
(failed/'attempt2_remote_verify.py').write_text(first_remote,encoding='utf-8')
(failed/'attempt1_error_sanitized.txt').write_bytes((OUT/'transport_attempt1_error.txt').read_bytes())
(failed/'attempt2_error.txt').write_text('Traceback (most recent call last):\n  File "<stdin>", line 119, in <module>\nAssertionError\n',encoding='utf-8')
attempts=[{'attempt':1,'operation':'SSH audit harness initialization','status':'ERROR','experiment_failure':False,
           'remote_verifier_started':False,'source':'failed_attempts/attempt1_remote_transport.py','error':'failed_attempts/attempt1_error_sanitized.txt',
           'fix':'Private stdout capture implements the reconfigure method required by the existing helper.'},
          {'attempt':2,'operation':'independent remote CPU verification','status':'ERROR','experiment_failure':False,
           'remote_verifier_started':True,'source':'failed_attempts/attempt2_remote_verify.py','error':'failed_attempts/attempt2_error.txt',
           'fix':'Traverse the actual previous/coordinate/memory/base config chain instead of ambiguous historical config discovery.'},
          {'attempt':3,'operation':'independent remote CPU verification','status':'PASS','result':'remote_verification.json','exit_code':0},
          {'attempt':4,'operation':'local intake and claim aggregates','status':'PASS','result':'local_verification.json','exit_code':0},
          {'attempt':5,'operation':'independent synthetic CPU gradients and resource read','status':'PASS','result':'remote_math_verification.json','exit_code':0}]
(OUT/'audit_attempts.json').write_text(json.dumps(attempts,indent=2)+'\n',encoding='utf-8')

md=(OUT/'EXPERIMENT_AUDIT.md').read_bytes()
inputs={x['input']:'sha256:'+x['sha256'] for x in local['audited_input_hashes']+supplemental}
inputs.update({k:'sha256:'+v['sha256'] for k,v in remote['audited_files'].items()})
checks={
    'A':{'status':'PASS','name':'ground_truth_provenance','evidence':['tools/probe_msvr_role_set_gradients.py:78-89','tools/msvr_role_set_relations.py:22-35','remote_verification.json:15-51'],
         'details':'Dataset filename labels, complete source identity partitions, source checkpoint metadata, all sampled indices and historical queue verified independently.'},
    'B':{'status':'PASS','name':'metric_normalization','evidence':['tools/msvr_role_set_relations.py:25-56','tools/probe_msvr_history_candidate_gradients.py:53-66'],
         'details':'Raw loss/gradient norms plus standard numerical cosine/relative error. No retrieval score rescaling; mean-hinge softening explicitly disclosed.'},
    'C':{'status':'PASS','name':'files_numbers_status','evidence':['intake/pipeline.json:2-51','remote_verification.json:3154-3231','local_verification.json:10-12'],
         'details':'All 14 remote artifacts, 24 batches, 834560 float32 distance elements, all aggregate numbers and terminal receipts match.'},
    'D':{'status':'WARN','name':'execution_and_gradient_witnesses','evidence':['tools/probe_msvr_role_set_gradients.py:53-68','tools/probe_msvr_role_set_gradients.py:93-96','tools/probe_msvr_role_set_gradients.py:104-159'],
         'details':'Live objective and VJP path supported; all-role equality only at the unqueued first batch, historical equality fused-only; no retained original gradients/fields/RNG for independent GPU replay; same-graph backward repeats only.'},
    'E':{'status':'PASS','name':'scope_resources','evidence':['results/MSVR310_ROLE_SET_GRADIENT_CHECK_2026-09-08.md:7-23','remote_verification.json:52-153','remote_verification.json:3226'],
         'details':'One seed, three fixed initial states, 24 source batches, 15 history batches, zero updates and zero new retrieval evaluation. Outputs within probe budget; no method qualification.'},
    'F':{'status':'PASS','name':'evaluation_type','evidence':['tools/check_msvr_role_set_relations.py:62-68','tools/verify_msvr_role_set_gradients.py:79-80','results/MSVR310_ROLE_SET_GRADIENT_CHECK_2026-09-08.md:19-29'],
         'details':'real_gt identity provenance; synthetic_proxy implementation self-comparison on real source outputs; simulation_only synthetic numerical unit tests. No task-performance claim.'}}
report={'audit_skill':'experiment-audit','verdict':'WARN','overall_verdict':'warn','integrity_status':'warn',
        'deterministic_checks_status':'pass','engineering_status':'fixed_state_probe_pass_with_limits','scientific_qualification':'not_evaluated',
        'reason_code':'bounded_runtime_witness_and_plan_coverage_gap',
        'summary':'Complete source relation arithmetic and provenance pass; GPU derivative/state evidence remains bounded runtime witness. Final reporting correction closes overstatement, not missing coverage.',
        'executor_model':'codex-gpt-6-astra','executor_family':'openai','reviewer_model':'gpt-6-astra','reviewer_family':'openai','reviewer_reasoning':'max',
        'model_identity_basis':'requested native Codex reviewer route; backend identity not independently attested',
        'review_independence':'same-family','acceptance_status':'provisional','reviewer_task':'/root/audit_msvr_role_set_gradient_check',
        'verdict_id':'sha256:'+sha(md),'generated_at':datetime.now(timezone.utc).isoformat(),'date':'2026-09-08',
        'run_commit':remote['run_commit'],'trace_path':str(OUT),'audited_input_hashes':inputs,
        'checks':checks,'evaluation_types':['real_gt','synthetic_proxy','simulation_only'],
        'claim_limits':['No unknown-identity retrieval or method efficacy claim.','No total 14-loss gradient or all-203-tensor coverage claim.',
                        'No all-historical-role bitwise or multi-group real-model direct-graph proof.',
                        'No independent original GPU parameter-gradient/state reproduction.','Backward-repeat noise only; no independently repeated stochastic forward.',
                        'Objective softening remains confounded with role-coverage intervention.'],
        'reporting_correction':{'status':'closed','final_result_evidence':'final_claim_snapshots/repo/results/MSVR310_ROLE_SET_GRADIENT_CHECK_2026-09-08.md:27-31',
                                'bound_execution_plan_changed':False,'missing_evidence_filled':False},
        'outside_scope':['Unregistered new training/verification draft implementation','Post-probe transport-bundle deletion ledger','Future M0/Q1 execution'],
        'claims':[{'id':'relation_counts_and_scalar_replay','impact':'supported_deterministically'},
                  {'id':'fixed_state_role_gradient_change','impact':'supported_runtime_witness_with_limits'},
                  {'id':'all_historical_role_bitwise_equality','impact':'not_established_report_now_qualified'},
                  {'id':'training_or_retrieval_qualification','impact':'not_evaluated'}],
        'verification_results':['remote_verification.json','local_verification.json','remote_math_verification.json'],
        'failed_attempts_preserved':'audit_attempts.json'}
(OUT/'EXPERIMENT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
artifact_manifest=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='audit_bundle_manifest.json':
        data=p.read_bytes()
        artifact_manifest.append({'path':p.relative_to(OUT).as_posix(),'bytes':len(data),'sha256':sha(data)})
(OUT/'audit_bundle_manifest.json').write_text(json.dumps(artifact_manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_status':'WARN','deterministic_checks':'PASS','checks':{k:v['status'] for k,v in checks.items()},
                  'input_hash_count':len(inputs),'bundle_files':len(artifact_manifest),'report_bytes':len(md),
                  'final_claim_snapshots':2,'failed_attempts_preserved':2},indent=2))
