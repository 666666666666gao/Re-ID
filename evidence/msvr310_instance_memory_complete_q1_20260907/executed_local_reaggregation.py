from pathlib import Path
import json,hashlib
import numpy as np
r=Path('C:/Users/gb/.codex_tmp/msvr_instance_memory_q1_complete_20260907')
s=json.loads((r/'q1/summary.json').read_bytes());cpu=json.loads((r/'q1_cpu.json').read_bytes());p=json.loads((r/'pipeline.json').read_bytes())
assert s['status'] in {'Q1_PASS','Q1_FAIL'} and p['status']=='COMPLETE_VERIFIED_'+s['status']
assert cpu['status']=='PASS_COMPLETE_INSTANCE_MEMORY_Q1'
assert cpu['summary_sha256']==hashlib.sha256((r/'q1/summary.json').read_bytes()).hexdigest()==p['terminal_summary_sha256']
assert hashlib.sha256((r/'q1_cpu.json').read_bytes()).hexdigest()==p['terminal_cpu_sha256']
ends=('control','instance_memory');outputs=('baseline_only','fused','cnn','transformer','mamba')
aps={e:{o:[] for o in outputs} for e in ends};first={e:{o:[] for o in outputs} for e in ends}
ids=[];training=[];fold_maps={e:[] for e in ends};checked=0;all_samples=0
def quantiles(values):
 a=np.asarray(values,dtype=np.float64)
 return dict(count=len(a),mean=float(a.mean()),minimum=float(a.min()),p25=float(np.quantile(a,.25)),
  median=float(np.median(a)),p75=float(np.quantile(a,.75)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))
def boot(delta,labels):
 unique=np.unique(labels);sums=np.array([delta[labels==i].sum() for i in unique]);counts=np.array([(labels==i).sum() for i in unique])
 select=np.random.default_rng(42).integers(0,len(unique),size=(10000,len(unique)))
 return float(np.quantile(sums[select].sum(1)/counts[select].sum(1),.025,method='linear'))
for fold in s['folds']:
 paired=[]
 for end in ends:
  row=fold['endpoints'][end];d=r/f"q1/fold_{fold['fold']}_{end}"
  assert json.loads((d/'receipt.json').read_bytes())==row
  tr=json.loads((d/'training.json').read_bytes());assert tr==row['training']
  assert tr['optimizer_steps']==260 and tr['nonzero_gradient_tensors']==tr['trainable_tensors']==203
  assert tr['missing_nonzero_gradients']==[] and tr['overflow_events']==0
  assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
  assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
  assert tr['final_state_sha256']==row['strict_reload_state_sha256'] and all(row['engineering_checks'].values())
  audit=[json.loads(x) for x in (d/'memory_steps.jsonl').read_text().splitlines()]
  assert len(audit)==260
  for i,(a,t) in enumerate(zip(audit,tr['steps'],strict=True)):
   assert a['step']==t['step']==i+1 and a['zero_based_step']==i
   assert a['record_indices']==t['sampled_record_indices']
   assert a['replacement_active']==(end=='instance_memory' and i>=65)
   assert a['current_features_finite'] and a['historical_features_detached']
   assert len(set(x['record_index'] for x in a['memory']))==len(a['memory'])<=512
   assert all(x['record_index'] not in a['record_indices'] and 1<=x['age']<=8 for x in a['memory'])
   assert all(x['age']==i-x['stored_step'] for x in a['memory'])
   assert (not a['memory']) if i<66 else bool(a['memory'])
   assert a['statistics']['memory_records']==len(a['memory'])
   assert all(np.isfinite(x) for x in t['components'].values()) and np.isfinite(t['loss'])
  paired.append([(a['record_indices'],a['pixel_sha256']) for a in audit])
  assert len(tr['fixed_pixel_drift'])==20
  for epoch,probe in enumerate(tr['fixed_pixel_drift'],1):
   assert probe['epoch']==epoch and probe['age_updates']==8 and len(probe['feature_l2_drift'])==64
   assert probe['same_pixels_and_rng'] and probe['model_buffers_restored']
  witness=tr['memory_parameter_gradient_witness'];assert all(x>0 for x in witness['actual_parameter_gradient_l2'].values())
  probe_post=[x for q in tr['fixed_pixel_drift'] if q['epoch']>=6 for x in q['feature_l2_drift']]
  training.append(dict(fold=fold['fold'],endpoint=end,updates=260,all203gradients=True,zero_overflow=True,
   history_candidate_exposures=sum(a['statistics']['memory_records'] for a in audit),
   history_positive_pair_exposures=sum(a['statistics']['memory_positive_pairs'] for a in audit),
   history_cross_scene_positive_pair_exposures=sum(a['statistics']['memory_cross_scene_positive_pairs'] for a in audit),
   history_negative_pair_exposures=sum(a['statistics']['memory_negative_pairs'] for a in audit),
   harder_positive_anchor_exposures=sum(a['statistics']['harder_positive_anchors'] for a in audit),
   harder_negative_anchor_exposures=sum(a['statistics']['harder_negative_anchors'] for a in audit),
   history_negative_hinge_exposures=sum(a['statistics']['memory_negative_violations_against_batch_hard_positive'] for a in audit),
   history_active_steps=sum(bool(a['memory']) for a in audit),
   postwarmup_original_triplet_mean=float(np.mean([a['original_triplet'] for a in audit[65:]])),
   postwarmup_expanded_triplet_mean=float(np.mean([a['statistics']['expanded_triplet'] for a in audit[65:]])),
   last65_original_triplet_mean=float(np.mean([a['original_triplet'] for a in audit[-65:]])),
   last65_expanded_triplet_mean=float(np.mean([a['statistics']['expanded_triplet'] for a in audit[-65:]])),
   actual_parameter_gradient_witness=witness,postwarmup_age8_drift=quantiles(probe_post),
   all_epoch_age8_drift=[dict(epoch=q['epoch'],**quantiles(q['feature_l2_drift'])) for q in tr['fixed_pixel_drift']],
   peak_allocated_mib=tr['peak_allocated_mib'],peak_reserved_mib=tr['peak_reserved_mib']))
  all_samples+=260
  rt=row['retrieval'];ranks=json.loads((d/'rankings.json').read_bytes())
  assert hashlib.sha256((d/'rankings.json').read_bytes()).hexdigest()==rt['rankings_sha256']
  assert rt['baseline_features_and_distances_bitwise_equal_to_b0']
  gallery=rt['gallery_manifest'];queries=rt['query_rows']
  if end=='control':ids.extend(q['identity'] for q in queries)
  metrics={}
  for out in outputs:
   out_ap=[];out_first=[]
   assert len(ranks[out])==len(queries)
   for q,ranking in zip(queries,ranks[out],strict=True):
    assert sorted(ranking)==list(range(len(gallery)))
    qrow=gallery[q['gallery_position']];assert qrow['index']==q['record_index'] and qrow['identity']==q['identity']
    legal=[g for g in ranking if not(gallery[g]['identity']==qrow['identity'] and gallery[g]['scene']==qrow['scene'])]
    positives=[i+1 for i,g in enumerate(legal) if gallery[g]['identity']==qrow['identity']]
    assert positives
    out_ap.append(sum((i+1)/rank for i,rank in enumerate(positives))/len(positives));out_first.append(positives[0]);checked+=len(ranking)
   score=rt['outputs'][out]
   assert np.allclose(out_ap,score['average_precision'],rtol=0,atol=1e-14) and out_first==score['first_match_rank']
   metric=dict(mAP=float(np.mean(out_ap)*100),**{f'Rank-{k}':float(np.mean(np.array(out_first)<=k)*100) for k in (1,5,10)})
   assert all(abs(v-score['metrics'][k])<1e-10 for k,v in metric.items())
   metrics[out]=metric;aps[end][out].extend(out_ap);first[end][out].extend(out_first)
  fold_maps[end].append(metrics)
 assert paired[0]==paired[1] and fold['all_paired_source_pixels_exact']
 assert fold['endpoints']['control']['initialization']==fold['endpoints']['instance_memory']['initialization']
assert all_samples==1560==cpu['checked_training_steps'] and checked==cpu['checked_retrieval_distance_and_rank_elements']
labels=np.array(ids);assert len(ids)==600 and len(set(ids))==60
comparison=s['comparison'];results={}
for end in ends:
 ms={o:dict(mAP=float(np.mean(aps[end][o])*100),**{f'Rank-{k}':float(np.mean(np.array(first[end][o])<=k)*100) for k in (1,5,10)}) for o in outputs}
 recorded=comparison['endpoints'][end]
 for o in outputs:
  assert all(abs(v-recorded['metrics'][o][k])<1e-10 for k,v in ms[o].items())
 for row in recorded['per_identity']:
  mask=labels==row['identity'];assert int(mask.sum())==row['query_count']
  assert all(abs(float(np.array(aps[end][o])[mask].mean()*100)-row['map_by_output'][o])<1e-10 for o in outputs)
 gains={o:ms[o]['mAP']-ms['baseline_only']['mAP'] for o in outputs}
 lb=boot((np.array(aps[end]['fused'])-np.array(aps[end]['baseline_only']))*100,labels)
 fgs=[f['fused']['mAP']-f['baseline_only']['mAP'] for f in fold_maps[end]]
 gates=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_gains_nonnegative=all(g>=0 for g in fgs),
  all_full_branches_not_below_signal=all(gains[o]>=0 for o in outputs[2:]),identity_bootstrap_lower_positive=lb>0,
  fused_strictly_best=all(ms['fused']['mAP']>ms[o]['mAP'] for o in outputs if o!='fused'))
 assert gates==recorded['scientific_checks'] and abs(lb-recorded['identity_bootstrap']['lower_bound_pp'])<1e-10
 results[end]=dict(metrics=ms,gains_over_signal=gains,fold_fused_gains=fgs,bootstrap_lower=lb,gates=gates)
delta={o:(np.array(aps['instance_memory'][o])-np.array(aps['control'][o]))*100 for o in outputs}
gains={o:float(delta[o].mean()) for o in outputs};lb=boot(delta['fused'],labels)
fgs=[b['fused']['mAP']-a['fused']['mAP'] for a,b in zip(fold_maps['control'],fold_maps['instance_memory'],strict=True)]
candidate=results['instance_memory']['metrics']
gates=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_nonnegative=all(g>=0 for g in fgs),
 all_role_gains_nonnegative=all(gains[o]>=0 for o in outputs[2:]),paired_identity_bootstrap_lower_positive=lb>0,
 candidate_fused_strictly_best=all(candidate['fused']['mAP']>candidate[o]['mAP'] for o in outputs if o!='fused'))
assert gates==comparison['paired_checks'] and np.allclose(fgs,comparison['fold_fused_gains_mAP'],rtol=0,atol=1e-10)
assert abs(lb-comparison['paired_bootstrap_lower_pp'])<1e-10
assert all(abs(gains[o]-comparison['matched_gains_mAP'][o])<1e-10 for o in outputs)
changes={};by_identity=[]
for o in outputs:
 a=np.array(first['control'][o]);b=np.array(first['instance_memory'][o])
 changes[o]=dict(ap_improved=int((delta[o]>0).sum()),ap_declined=int((delta[o]<0).sum()),ap_unchanged=int((delta[o]==0).sum()),
  rank1_repaired=int(((a>1)&(b==1)).sum()),rank1_new_errors=int(((a==1)&(b>1)).sum()))
for row in comparison['paired_per_identity']:
 mask=labels==row['identity'];assert int(mask.sum())==row['query_count']
 dg={o:float(delta[o][mask].mean()) for o in outputs}
 assert all(abs(dg[o]-row['gains_mAP'][o])<1e-10 for o in outputs)
 by_identity.append(dict(identity=row['identity'],query_count=int(mask.sum()),gains=dg))
passed=all(gates.values()) and all(results['instance_memory']['gates'].values())
assert comparison['next_phase_qualified']==passed and s['status']==('Q1_PASS' if passed else 'Q1_FAIL')
report=dict(status='PASS_COMPLETE_Q1_TEXT_REAGGREGATION',scientific_status=s['status'],training_updates=all_samples,
 checked_rank_positions=checked,query_count=600,unique_identities=60,all_endpoints=results,paired_gains=gains,
 paired_fold_fused_gains=fgs,paired_bootstrap_lower=lb,paired_gates=gates,paired_query_changes=changes,
 paired_per_identity=by_identity,all_training_coverage=training,summary_sha256=cpu['summary_sha256'],
 cpu_sha256=p['terminal_cpu_sha256'],registered_extra_probe_record_forwards=7680,source_memory_only=True,
 official_test_reads=s['official_image_reads'],whole_goal_achieved=False,
 limitations='Source-instance XBM intervention includes hard positive and negative coverage and stale detached history. Full CPU distance/weight checks independently verified remotely; local replay covers all saved text rankings and training audits, not new model forward. Seed42 repeatedly used OOF is not independent external validation.')
(r/'local_complete_reaggregation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('paired_per_identity','all_training_coverage')}))
