from pathlib import Path
import datetime, hashlib, json, shutil, subprocess

root = Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
out = Path(__file__).parent
old = Path(r'C:\Users\gb\.codex_tmp\smooth_ap_source_coverage_independent_audit_20260909')
trace = root/'.aris/traces/experiment-audit/2026-09-20_smooth_ap_source_coverage'
trace.mkdir(parents=True, exist_ok=True)
prompt = '''Use experiment-audit skill C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and its local policy/review-tracing. Concrete bounded task: independently audit completed fixed-source Smooth-AP candidate/full-gallery diagnostic. Prior auditor is unavailable (list_agents returns root only); its incomplete artifacts are inputs, not verdict. Do not repeat an existing verification if actual complete receipts exist; inspect first. No new subagents. Paths: repo C:/Users/gb/.trifusion_github_publish_22c3bee; AGENTS.md; configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json; refine-logs/msvr310_smooth_ap_source_coverage_v1/DIAGNOSTIC_PLAN.md; tools/diagnose_msvr_smooth_ap_coverage.py; follow actual loader/model/protocol source dependencies; results/MSVR310_SMOOTH_AP_SOURCE_COVERAGE_2026-09-09.md; evidence/smooth_ap_source_coverage_{preparation,launch,verification_progress,complete,analysis,completion_support}_20260909; master docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md sections41.203-41.208. Prior auditor artifacts C:/Users/gb/.codex_tmp/smooth_ap_source_coverage_independent_audit_20260909 including snapshots,inventory,independent_remote_cpu_verify.py,remote_discovery.jsonl; private .aris/traces/experiment-audit/2026-09-09_smooth_ap_source_coverage. Remote repo /root/autodl-tmp/trifusion-v2/TriFusion-ReID; diagnostic run /root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590. Existing executor verification is evidence, not independent verdict. Check all GT/masks/self/dedup/gallery/denominators/ties/weightings/scope/source-vs-heldout claims and all registered results. CPU-only reading arrays/records allowed; no GPU/model forwards, no training, no official image reads, no weight deletion. A distinct objective-gradient diagnostic is now running GPU; do not touch it. Existing authenticated helper C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py defines paramiko c; execute privately without printing its credentials. Transport C:/Users/gb/.codex_tmp/role_set_remote_command_20260908.py has60s channel timeout; use durable CPU process with logs for longer verification, never restart on timeout. Remote mutations only to new audit artifact directory/scripts/logs if required, not source or science outputs. Save output under C:/Users/gb/.codex_tmp/smooth_ap_source_coverage_independent_audit_20260920; preserve input snapshots hashes, scripts/errors, whole verification scope and limitations, full report MD/JSON and exact final reply. Private trace .aris/traces/experiment-audit/2026-09-20_smooth_ap_source_coverage: save this entire prompt verbatim as001-review.request.json, model requestedgpt-6-astra/effortmax/forknone, actual tool-visible agent id; no invented UUID/backend identity. Same-family/provisional. Do not commit/push or edit master; return report paths and actual findings. Do not audit unfinished objective-gradient run or repeat sealed Q1/M0 audits.'''
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
meta = dict(skill='experiment-audit', run_id=trace.name, started_at=now,
    executor='codex', executor_model='gpt-6-astra', executor_family='openai',
    review_independence='same-family', acceptance_status='provisional', project_dir=str(root))
request = dict(call_number=1,purpose='review',timestamp=now,tool='collaboration.spawn_agent',
    model='gpt-6-astra',reasoning_effort='max',fork_turns='none',
    agent_id='/root/audit_smooth_coverage_resume_20260920',agent_uuid=None,
    backend_identity_independently_verified=False,
    metadata_source='Parent task requested model/effort/fork; collaboration.list_agents independently returned canonical agent name only.',
    prompt=prompt)
for name, data in [('run.meta.json',meta),('001-review.request.json',request)]:
    p=trace/name
    if p.exists():
        existing=json.loads(p.read_bytes())
        if name.endswith('request.json'):
            assert existing['prompt']==prompt
    else:
        p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

declared='''AGENTS.md
configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json
configs/MSVR310/TriFusion-smooth-ap-paired-v1.json
refine-logs/msvr310_smooth_ap_source_coverage_v1/DIAGNOSTIC_PLAN.md
tools/diagnose_msvr_smooth_ap_coverage.py
tools/diagnose_msvr_source_relations.py
tools/msvr310_exact_signal_inference.py
tools/train_msvr_smooth_ap.py
tools/train_msvr_instance_memory.py
tools/train_msvr310_trifusion_oof.py
tools/train_msvr310_signal_oof.py
tools/train_msvr310_source_style.py
tools/run_signal_preserving_v5.py
tools/msvr_instance_memory.py
tools/msvr_freshness_probe.py
protocols/msvr310_train_oof_v1.json
results/MSVR310_SMOOTH_AP_SOURCE_COVERAGE_2026-09-09.md
docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'''.splitlines()
files=[(root/p,'repo/'+p) for p in declared]
for suffix in ['preparation','launch','verification_progress','complete','analysis','completion_support']:
    d=root/'evidence'/('smooth_ap_source_coverage_'+suffix+'_20260909')
    files.extend((p,'repo/'+p.relative_to(root).as_posix()) for p in sorted(d.rglob('*')) if p.is_file())
files.extend((p,'prior_audit/'+p.relative_to(old).as_posix()) for p in sorted(old.rglob('*')) if p.is_file())
prior_trace=root/'.aris/traces/experiment-audit/2026-09-09_smooth_ap_source_coverage'
files.extend((p,'prior_trace/'+p.relative_to(prior_trace).as_posix()) for p in sorted(prior_trace.rglob('*')) if p.is_file())
skill=Path(r'C:\Users\gb\.codex\skills')
for p in ['experiment-audit/SKILL.md','shared-references/local-codex-policy.md','shared-references/review-tracing.md','shared-references/experiment-integrity.md','shared-references/reviewer-independence.md']:
    files.append((skill/p,'audit_policy/'+p))
inventory=[]
for p,rel in files:
    if not p.exists():
        inventory.append(dict(source=str(p),snapshot=rel,exists=False));continue
    data=p.read_bytes(); dst=out/'snapshots'/rel; dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.exists()
    dst.write_bytes(data)
    inventory.append(dict(source=str(p),snapshot=str(dst.relative_to(out)),exists=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),lines=len(data.splitlines())))
(out/'local_input_inventory.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(out/'initial_environment_errors.md').write_text('''# Preserved local runtime discovery errors

1. E:\\python.exe importing paramiko,numpy: ModuleNotFoundError: No module named 'paramiko'.
2. C:\\Users\\gb\\AppData\\Roaming\\uv\\python\\cpython-3.12.12-windows-x86_64-none\\python.exe importing paramiko,numpy: same missing paramiko.
3. E:\\anaconda\\python.exe importing paramiko failed with ImportError: DLL load failed in cryptography.hazmat.bindings._openssl. No packages or environments were modified.
4. Prior trace identified the established ClawX uv offline runtime route. Subsequent transport uses that existing route.
''',encoding='utf-8')
(out/'initial_git_status.txt').write_text(subprocess.check_output(['git','-C',str(root),'status','--short'],text=True),encoding='utf-8')
print(json.dumps(dict(files=len(inventory),missing=[r['source'] for r in inventory if not r['exists']],bytes=sum(r.get('bytes',0) for r in inventory),trace=str(trace)),ensure_ascii=False))
