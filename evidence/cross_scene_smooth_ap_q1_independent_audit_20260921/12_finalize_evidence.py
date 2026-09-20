from pathlib import Path
import hashlib
import json
import math

OUT=Path(__file__).resolve().parent
REPO=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
RAW=OUT.parent/'trifusion_cross_scene_q1_complete_20260921'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
first=read(OUT/'03_remote_intake.stdout');last=read(OUT/'09_remote_final_checks.stdout')
assert first['inventory']==last['inventory']
assert first['pipeline']==last['pipeline']
assert not any(last['original_process_presence'].values())
arithmetic=read(OUT/'07_remote_arithmetic.stdout');desc=read(OUT/'descriptive_claim_checks.json')
prov=read(OUT/'05_remote_provenance.stdout');links=read(OUT/'10_remote_receipt_links.stdout')
summary=read(RAW/'q1/summary.json');outputs=('baseline_only','fused','cnn','transformer','mamba')
identity_metrics_checked=0;query_changes_checked=0
for end in ('control','cross_scene'):
    saved=summary['comparison']['endpoints'][end]
    for ident in saved['per_identity']:
        for output in outputs:
            rows=[r for r in arithmetic['query_rows'] if r['endpoint']==end and r['output']==output and r['identity']==ident['identity']]
            assert len(rows)==ident['query_count']
            expected=math.fsum(r['ap'] for r in rows)/len(rows)*100
            assert abs(expected-ident['map_by_output'][output])<1e-10
            identity_metrics_checked+=1
    baseline=[r for r in arithmetic['query_rows'] if r['endpoint']==end and r['output']=='baseline_only']
    for output in outputs[1:]:
        actual=[r for r in arithmetic['query_rows'] if r['endpoint']==end and r['output']==output]
        delta=[b['ap']-a['ap'] for a,b in zip(baseline,actual)]
        changes=dict(ap_improved=sum(v>0 for v in delta),ap_declined=sum(v<0 for v in delta),ap_unchanged=sum(v==0 for v in delta),
            rank1_repaired=sum(a['first_rank']>1 and b['first_rank']==1 for a,b in zip(baseline,actual)),
            rank1_new_errors=sum(a['first_rank']==1 and b['first_rank']>1 for a,b in zip(baseline,actual)))
        assert changes==saved['query_changes'][output];query_changes_checked+=1
target=OUT/'remote_source_r2';target.mkdir();remote_manifest=[]
texts=dict(last['source_texts']);texts.update(prov['source_texts'])
remote_hashes={r['path']:r['current_sha256'] for r in last['source_matches']}
remote_hashes.update({r['path']:r['actual'] for r in prov['bindings']})
for n,(path,text) in enumerate(sorted(texts.items())):
    p=target/(str(n).zfill(3)+'_'+Path(path).name);p.write_text(text,encoding='utf-8',newline='\n')
    row=dict(path=path,snapshot=str(p.relative_to(OUT)),source_sha256=remote_hashes[path],text_snapshot_sha256=sha(p),newline_policy='UTF-8 universal newlines normalized to LF')
    prefix='/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'
    if path.startswith(prefix):
        local=REPO/path.removeprefix(prefix);data=local.read_bytes()
        row['local_sha256']=sha(local);row['local_exact']=sha(local)==remote_hashes[path]
        row['local_lf_equivalent']=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==sha(p)
        assert row['local_exact'] or row['local_lf_equivalent'],path
    remote_manifest.append(row)
(OUT/'remote_source_manifest.json').write_text(json.dumps(remote_manifest,indent=2)+'\n',encoding='utf-8')
snapshot_manifest=read(OUT/'local_input_manifest.json')
for r in snapshot_manifest['files']:
    assert sha(OUT/r['snapshot'])==r['sha256']
allhashes={r['path']:r['sha256'] for r in snapshot_manifest['files']}
allhashes.update({r['path']:r['sha256'] for r in read(OUT/'additional_input_manifest.json')})
allhashes.update({r['path']:r['sha256'] for r in last['inventory']})
allhashes.update({r['path']:r['actual'] for r in prov['bindings']})
allhashes.update({r['path']:r['current_sha256'] for r in last['source_matches']})
(OUT/'audited_input_hashes.json').write_text(json.dumps(allhashes,indent=2)+'\n',encoding='utf-8')
evidence=dict(
    status='PASS_DETERMINISTIC_CHECKS_WITH_EXPLICIT_RUNTIME_LIMITS',
    first_observation=first['observed_at'],last_observation=last['observed_at'],
    original_artifact_files_unchanged=len(first['inventory']),registered_hash_bindings=len(prov['bindings']),
    registered_execution_source_matches=len(prov['execution_source_matches']),
    followed_project_source_matches=len(last['source_matches']),
    local_source_exact=sum(r.get('local_exact',False) for r in remote_manifest),
    local_source_lf_only=[r['path'] for r in remote_manifest if not r.get('local_exact',True)],
    original_process_presence=last['original_process_presence'],output_parent_free_bytes=last['output_parent_free_bytes'],
    protocol=prov['protocol'],checkpoint_states=prov['checkpoint_results'],initialization_links=links,
    totals=arithmetic['totals'],maximum_errors=arithmetic['errors'],training=arithmetic['training'],
    zero_eligible_batches=arithmetic['zero_batches'],warmup_comparison=arithmetic['warmup_pairing'],
    endpoint_metrics=arithmetic['endpoints'],paired_gains_pp=arithmetic['paired_gains_pp'],
    paired_fold_gains_pp=arithmetic['paired_fold_gains_pp'],paired_bootstrap_lower_pp=arithmetic['paired_bootstrap_lower_pp'],
    paired_checks=arithmetic['paired_checks'],scientific_qualification=arithmetic['scientific_qualification'],
    integer_mask_counts=last['masks'],descriptive_claim_checks=desc,
    endpoint_identity_output_metrics_checked=identity_metrics_checked,endpoint_query_change_groups_checked=query_changes_checked,
    cumulative_total_role_record_computations=sum(arithmetic['totals'][k] for k in ('training_anchor_exposures','fresh_role_record_forwards','history_vjp_record_forwards')),
    training_original_elapsed_seconds=next(s['elapsed_seconds'] for s in last['pipeline']['stages'] if s['stage']=='q1'),
    original_pipeline=last['pipeline'],audited_input_count=len(allhashes),
    raw_results=['03_remote_intake.stdout','05_remote_provenance.stdout','07_remote_arithmetic.stdout','descriptive_claim_checks.json','09_remote_final_checks.stdout','10_remote_receipt_links.stdout'],
    model_forwards=0,optimizer_updates=0,official_dataset_reads=0)
(OUT/'AUDIT_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
incidents=[
    dict(attempt='11_finalize_evidence',status='FAILED_AUDIT_SNAPSHOT_HELPER',error='Normalized text snapshot SHA did not equal original source SHA',reason='Remote read_text normalizes CRLF in source_style_msvr.py, while original remote file and execution blob match byte for byte.',resolution='Retained partial remote_source directory; attempt 12 writes new remote_source_r2 and records original source SHA separately from LF text snapshot SHA. No source changed.'),
    dict(attempt='01_remote_intake',status='FAILED_AUDIT_HELPER',error="KeyError: BASELINE",reason='Audit helper omitted the coordinate_config link.',resolution='Preserved; explicit actual config chain followed in attempt 03.'),
    dict(attempt='02_remote_intake',status='FAILED_AUDIT_HELPER',error="KeyError: BASELINE",reason='Audit helper omitted the memory_config link.',resolution='Preserved; explicit actual config chain followed in attempt 03.'),
    dict(attempt='06_remote_arithmetic',status='FAILED_ADDITIONAL_CHECK',error='AssertionError at line 257: paired warmup step records exact',reason='Actual paired common-objective trajectories differ despite exact initial and input bindings.',resolution='Retained as WARN evidence. Attempt 07 quantifies all differences; original protocol did not require bitwise warmup trajectories. No experiment gate changed.'),
    dict(attempt='executor terminal report v1',status='FAILED_EXECUTOR_REPORT_HELPER',error='Wrong query-only ranking index based on gallery position',reason='Documented in supplied FAILURE.md and failed_v1.py.',resolution='Executor corrected indexing into a new v2 output directory. Every v2 CSV row independently checked.')]
(OUT/'AUDIT_ATTEMPTS.json').write_text(json.dumps(incidents,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=evidence['status'],input_hashes=len(allhashes),unchanged_artifacts=len(first['inventory']),
    local_source_exact=evidence['local_source_exact'],lf_only=evidence['local_source_lf_only'],identity_metrics=identity_metrics_checked,
    total_role_computations=evidence['cumulative_total_role_record_computations'])))
