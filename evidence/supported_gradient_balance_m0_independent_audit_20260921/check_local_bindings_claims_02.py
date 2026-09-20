from pathlib import Path
import json,hashlib,csv,difflib,subprocess,ast,statistics
root=Path(__file__).parent;repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
snap=root/'snapshots';remote=json.loads((root/'remote_intake_02.json').read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inventory=json.loads((snap/'intake/inventory.json').read_bytes())
checked=[]
for item in inventory['files']:
    p=snap/'intake'/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    candidates=[x for x in remote['m0_inventory'] if x['path'].endswith('/'+item['path'])]
    if candidates:assert candidates[0]['sha256']==item['sha256']
    else:
        name='/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/'+item['path']
        assert hashlib.sha256(remote['text_inputs'][name].encode()).hexdigest()==item['sha256']
    checked.append(item['path'])
r1=snap/'repo/evidence/supported_gradient_balance_m0_r1_failure_20260921'
inv1=json.loads((r1/'inventory.json').read_bytes());r1rows=[]
for item in inv1['files']:
    p=r1/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    r1rows.append(item['path'])
old=r1/'executed_snapshot';diffs=[];executed=[]
for p in sorted(old.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(old).as_posix()
    gitdata=subprocess.check_output(['git','show','92a75e4ac46cccb79bb6f78f913998c0d0039ce8:'+rel],cwd=repo)
    assert p.read_bytes()==gitdata
    executed.append(rel)
    new=repo/rel
    diffs.extend(difflib.unified_diff(p.read_text(encoding='utf-8').splitlines(True),new.read_text(encoding='utf-8').splitlines(True),fromfile='R1/'+rel,tofile='R2/'+rel))
(root/'r1_r2.diff').write_text(''.join(diffs),encoding='utf-8')
r1pipe=json.loads((r1/'remote/pipeline.json').read_bytes());assert r1pipe['status']=='STOPPED_AT_M0'
failure=ast.literal_eval((r1/'remote/m0.log').read_text(encoding='utf-8').split('AssertionError: ')[-1].strip())
assert failure['difference_norm']/failure['first_norm']==failure['relative_l2_error'] and failure['relative_l2_error']>.005 and failure['passed'] is False
r1count=len((r1/'remote/m0/fold_0_control/memory_steps.jsonl').read_text().splitlines());assert r1count==3
latest=[]
for dirname in ('evidence/supported_gradient_balance_m0_complete_20260921','evidence/supported_gradient_balance_m0_analysis_20260921'):
    for p in (repo/dirname).rglob('*'):
        if not p.is_file():continue
        dst=root/'latest_claims'/p.relative_to(repo);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());latest.append(str(dst))
for name in ('refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md','docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md','AGENTS.md'):
    p=repo/name;dst=root/'latest_claims'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());latest.append(str(dst))
analysis=root/'latest_claims/evidence/supported_gradient_balance_m0_analysis_20260921'
summary=json.loads((analysis/'summary.json').read_bytes())
records=[];refs=[];epochs=[];coverage=[]
for directory in sorted((snap/'intake/m0').iterdir()):
    if not directory.is_dir():continue
    tr=json.loads((directory/'training.json').read_bytes());audits=[json.loads(s) for s in (directory/'memory_steps.jsonl').read_text().splitlines()]
    case_count={r:dict(supported_zero_rank=0,clipped_rank_upper=0,clipped_rank_lower=0,negative_rank_aux_cosine=0) for r in ('cnn','transformer','mamba')}
    for a,t in zip(audits,tr['steps'],strict=True):
        for r in ('cnn','transformer','mamba'):
            b=a['gradient_balance'][r];p=b['rank_vs_auxiliary'];s=a['support']
            records.append(dict(endpoint=directory.name,step=a['step'],epoch=t['epoch'],role=r,active_fused_metric=t['active_fused_metric'],supported=b['supported'],eligible_anchors=s['eligible_anchors'],eligible_identities=s['eligible_identities'],identity_directed_scene_relations=s['identity_directed_scene_relations'],rank_norm=p['first_norm'],auxiliary_norm=p['second_norm'],rank_auxiliary_cosine=p['cosine'],rank_to_auxiliary_norm=p['first_norm']/p['second_norm'] if p['second_norm'] else None,proposed_rank_weight=b['proposed_rank_weight'],applied_rank_weight=b['applied_rank_weight'],applied_auxiliary_weight=b['applied_auxiliary_weight'],ratio=b['ratio'],history_rank_norm=b['current_rank_vs_history']['second_norm'],current_history_cosine=b['current_rank_vs_history']['cosine'],original_to_applied_difference=b['original_sum_vs_applied']['difference_norm'],direct_sum_to_original_difference=b['direct_sum_vs_original']['difference_norm'],subtraction_to_direct_auxiliary_difference=b['subtraction_auxiliary_vs_direct']['difference_norm'],actual_adamw_parameter_delta_norm=a['actual_parameter_updates'][r]['difference_norm'],classification_head_gradient_preserved=a['classification_head_gradients_unchanged']))
            for component,check in a['rank_auxiliary_reference_checks'].get(r,{}).items():refs.append(dict(endpoint=directory.name,step=a['step'],role=r,component=component,**check))
            if b['supported']:
                case_count[r]['supported_zero_rank']+=p['first_norm']==0
                case_count[r]['clipped_rank_upper']+=b['ratio']==4
                case_count[r]['clipped_rank_lower']+=b['ratio']==.25
                case_count[r]['negative_rank_aux_cosine']+=p['cosine'] is not None and p['cosine']<0
    epochs.extend(dict(endpoint=directory.name,**e) for e in tr['history'])
    coverage.append(dict(endpoint=directory.name,role_cases=case_count))
for name,data in [('role_steps.csv',records),('direct_reference_checks.csv',refs),('epochs.csv',epochs)]:
    actual=list(csv.DictReader((analysis/name).open(encoding='utf-8',newline='')))
    expected=[{k:'' if v is None else str(v) for k,v in row.items()} for row in data]
    assert actual==expected,name
assert summary['role_step_rows']==len(records)==744 and summary['direct_reference_checks']==len(refs)==90
assert summary['steps']==248 and summary['maximum_direct_reference_relative_l2_error']==max(x['relative_l2_error'] or 0 for x in refs)
for s in summary['role_summaries']:
    selected=[r for r in records if r['endpoint']==s['endpoint'] and r['role']==s['role'] and r['supported']]
    assert s['supported_role_steps']==len(selected)
    for field in ('rank_to_auxiliary_norm','applied_rank_weight','applied_auxiliary_weight','actual_adamw_parameter_delta_norm'):
        values=[r[field] for r in selected if r[field] is not None]
        assert s[field]==dict(count=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values))
    assert s['negative_rank_auxiliary_cosines']==sum(r['rank_auxiliary_cosine'] is not None and r['rank_auxiliary_cosine']<0 for r in selected)
result=dict(status='PASS_LOCAL_INTAKE_ARCHIVES_AND_DESCRIPTIVE_CLAIMS',intake_files=len(checked),intake_bytes=sum(x['bytes'] for x in inventory['files']),r1_archive_files=len(r1rows),r1_execution_snapshot_files=len(executed),r1_completed_optimizer_steps=r1count,r1_failed_step=4,r1_failure=failure,r1_elapsed_seconds=r1pipe['stages'][1]['elapsed_seconds'],r2_claim_csv_rows=dict(role_steps=len(records),direct_reference_checks=len(refs),epochs=len(epochs)),m0_additional_role_coverage=coverage,latest_claims_snapshots=latest)
(root/'local_bindings_claims.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
