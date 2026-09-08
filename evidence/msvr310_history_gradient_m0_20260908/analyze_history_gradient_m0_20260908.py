from pathlib import Path
import hashlib,json
tmp=Path('C:/Users/gb/.codex_tmp');p=tmp/'history_gradient_m0_complete_20260908'
s=json.loads((p/'m0/summary.json').read_bytes());v=json.loads((p/'m0_cpu.json').read_bytes())
assert s['status']=='PASS_ENGINEERING_ONLY' and v['status']=='PASS_COMPLETE_HISTORY_GRADIENT_M0'
assert v['summary_sha256']==hashlib.sha256((p/'m0/summary.json').read_bytes()).hexdigest()
rows=[];direct=[];vjp=0;fresh=0;steps=0
for fold in s['folds']:
 assert fold['all_paired_source_pixels_exact']
 for end,r in fold['endpoints'].items():
  tr=r['training'];assert all(r['engineering_checks'].values())
  assert r['strict_reload_all_outputs_bitwise_equal']
  assert tr['overflow_events']==0 and tr['nonzero_gradient_tensors']==tr['trainable_tensors']==203
  assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
  assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
  audits=[json.loads(x) for x in (p/'m0'/f'fold_{fold["fold"]}_{end}'/'memory_steps.jsonl').read_text().splitlines()]
  checks=[a['direct_single_group_check']['relative_l2_error'] for a in audits if a['direct_single_group_check']]
  assert len(checks)==1 and checks[0]<=.005;direct+=checks
  assert all(a['selected_reencoding_bitwise'] and a['history_rng_buffers_preserved'] and a['history_vjp_leaves_current_grad_unchanged'] and a['final_gradient_addition_bitwise'] for a in audits)
  assert all(a['history_anchor_count']==0 and a['current_anchor_count']==64 and a['history_candidate_vjp_applied']==(end=='history_gradient') for a in audits)
  vjp+=tr['extra_history_vjp_record_forwards'];fresh+=tr['extra_fresh_role_record_forwards'];steps+=tr['optimizer_steps']
  rows.append(dict(fold=fold['fold'],endpoint=end,steps=tr['optimizer_steps'],live_tensors=tr['nonzero_gradient_tensors'],overflow=tr['overflow_events'],direct_relative_l2=checks[0],peak_allocated_mib=tr['peak_allocated_mib'],vjp_record_forwards=tr['extra_history_vjp_record_forwards'],fresh_record_forwards=tr['extra_fresh_role_record_forwards']))
overfit=[]
for end,r in s['overfit'].items():
 tr=r['training'];assert all(r['checks'].values()) and r['gate']['passed'];steps+=tr['optimizer_steps']
 vjp+=tr['extra_history_vjp_record_forwards'];fresh+=tr['extra_fresh_role_record_forwards']
 overfit.append(dict(endpoint=end,steps=tr['optimizer_steps'],gate=r['gate'],live_tensors=tr['nonzero_gradient_tensors'],overflow=tr['overflow_events'],peak_allocated_mib=tr['peak_allocated_mib']))
assert steps==248 and vjp==v['checked_history_vjp_record_forwards']
result=dict(status='PASS_LOCAL_TEXT_M0_CLOSURE',scope='Complete M0 text and upstream remote CPU receipt; no independent model or binary distance replay locally',summary_sha256=v['summary_sha256'],cpu_sha256=hashlib.sha256((p/'m0_cpu.json').read_bytes()).hexdigest(),completed_at=s['completed_at'],capacity=rows,overfit=overfit,steps=steps,direct_relative_l2_max=max(direct),extra_history_vjp_record_forwards=vjp,extra_fresh_role_record_forwards=fresh,cpu_distance_elements=v['checked_memory_distance_elements'])
out=p/'local_m0_analysis.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
