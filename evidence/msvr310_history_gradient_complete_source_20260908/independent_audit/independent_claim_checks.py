"""Fresh read-only claim, mirror and PDF-position checks, derived from complete raw rows."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, math, re, statistics
import pymupdf

O=Path(__file__).parent;R=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');T=Path('C:/Users/gb/.codex_tmp')
S=T/'history_gradient_complete_source_20260908';M=R/'evidence/msvr310_history_gradient_complete_source_20260908'
prior=json.loads((O/'independent_checks.json').read_bytes());counts=Counter();inputs={}
def read(p):
 b=p.read_bytes();inputs[str(p)]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());return b
def js(p):return json.loads(read(p))
def eq(a,b,k):assert a==b,(k,a,b);counts[k]+=1
def near(a,b,k):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(k,a,b);counts[k]+=1
def rec(a,b,k):
 if isinstance(a,dict):
  eq(set(a),set(b),k+'_keys')
  for key in a:rec(a[key],b[key],k)
 elif isinstance(a,list):
  eq(len(a),len(b),k+'_list')
  for x,y in zip(a,b):rec(x,y,k)
 elif isinstance(a,float):near(a,b,k+'_float')
 else:eq(a,b,k+'_value')

for src,dst in [(S,M),(T/'history_gradient_complete_processing_20260908',M/'processing'),(T/'history_gradient_complete_analysis_20260908',M/'analysis'),(T/'history_gradient_complete_figures_20260908',M/'figures')]:
 for p in src.rglob('*'):
  if p.is_file():eq(read(p),read(dst/p.relative_to(src)),'mirror_file_bytes')
for source,target in [('finish_history_gradient_source_20260908.py','processing/driver.py'),('receive_history_gradient_complete_source_20260908.py','processing/receive.py'),('verify_history_gradient_source_pdf_20260908.py','figures/verify_pdf.py')]:eq(read(T/source),read(M/target),'mirror_processing_source_bytes')
read(T/'build_history_gradient_complete_report_20260908.py')
summary=js(S/'source/summary.json');pipe=js(S/'pipeline.json');manifest=js(S/'intake_manifest.json')
allrows=[];negative=Counter()
fields=['history_to_current_ratio','current_history_cosine','current_both_cosine','task_total_both_cosine']
for state in prior['states']:
 rows=[json.loads(x) for x in read(S/'source'/f"fold_{state['fold']}_{state['state']}"/'steps.jsonl').splitlines()]
 for row in rows:
  counts['raw_batch_rows']+=1
  for role,g in row['roles'].items():
   uv=g['current_vs_history'];points=dict(fold=state['fold'],state=state['state'],role=role,history_to_current_ratio=uv['second_norm']/uv['first_norm'],current_history_cosine=uv['cosine'],current_both_cosine=g['current_vs_both']['cosine'],task_total_both_cosine=g['total_vs_both']['cosine'])
   allrows.append(points)
   if points['current_both_cosine']<0:negative[(points['fold'],points['state'],role)]+=1
numbers=js(M/'report_numbers.json')
global_stats={f:dict(count=len(allrows),mean=statistics.fmean(p[f] for p in allrows),minimum=min(p[f] for p in allrows),maximum=max(p[f] for p in allrows),negative=sum(p[f]<0 for p in allrows)) for f in fields}
rec(global_stats,numbers['global_statistics'],'global_report_numbers')
rec([dict(fold=k[0],state=k[1],role=k[2],count=v) for k,v in sorted(negative.items())],numbers['negative_current_both_groups'],'negative_group_counts')
for key,value in numbers['coverage_totals'].items():eq(value,prior['totals'][key],'report_coverage_totals')
for key,value in dict(extra_role_record_forwards=prior['totals']['extra_role_record_forwards'],maximum_chain_relative_error=max(p['relative_l2_error'] for p in prior['direct_proofs']),maximum_peak_allocated_mib=max(s['peak_allocated_mib'] for s in prior['states']),observed_record_memberships=sum(s['source_records'] for s in prior['states'])).items():
 if isinstance(value,float):near(value,numbers[key],'report_resource_numbers')
 else:eq(value,numbers[key],'report_resource_numbers')

report_path=R/'results/MSVR310_HISTORY_CANDIDATE_GRADIENT_SOURCE_2026-09-08.md'
text=read(report_path).decode('utf-8');lines=text.splitlines();tables=[];active=[]
for i,line in enumerate(lines,1):
 if line.startswith('|'):
  active.append((i,[x.strip() for x in line.strip('|').split('|')]))
 elif active:tables.append(active);active=[]
if active:tables.append(active)
eq(len(tables),4,'report_table_count')
state_labels={'initial':'原角色初始化','control':'陈旧记忆终点','fresh_memory':'新鲜坐标终点'};role_labels={'cnn':'CNN','transformer':'Transformer','mamba':'Mamba'}
expected_tables=[[],[],[],[]]
for s in prior['states']:
 f=next(x for x in prior['fold_scope'] if x['fold']==s['fold'])
 expected_tables[0].append([str(s['fold']),state_labels[s['state']],f"{s['source_records']} / {f['source_identities']}",'260 / 194',str(s['extra_role_record_forwards']),f"{s['peak_allocated_mib']:.3f}",f"{s['chain_rule_relative_error']:.9g}"])
 for role in role_labels:
  ps=[p for p in allrows if (p['fold'],p['state'],p['role'])==(s['fold'],s['state'],role)]
  expected_tables[1].append([str(s['fold']),state_labels[s['state']],role_labels[role]]+[f'{statistics.fmean(p[field] for p in ps):.6f}' for field in fields]+[str(sum(p['current_both_cosine']<0 for p in ps))])
 c=s['queue_and_vjp_coverage'];expected_tables[3].append([str(s['fold']),state_labels[s['state']]]+[str(c[key]) for key in ['available_history_group_exposures','selected_vjp_group_exposures','zero_upstream_group_skip_exposures','age_expired_record_exposures','excluded_current_history_exposures']]+[f"{c['maximum_selected_history_records']} / {c['maximum_queue_after_update']}"])
for field in fields:
 s=global_stats[field];expected_tables[2].append([field]+[f'{s[k]:.6f}' for k in ['mean','minimum','maximum']]+[str(s['negative'])])
for observed,expected in zip(tables,expected_tables):
 eq(len(observed)-2,len(expected),'report_table_rows')
 for (line_no,cells),expected_cells in zip(observed[2:],expected):
  eq(len(cells),len(expected_cells),'report_table_column_count')
  for cell,target in zip(cells,expected_cells):
   eq(cell,target,'report_table_cell');counts['report_table_numeric_tokens']+=len(re.findall(r'(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?',cell))

# Explicit source numbers used in prose, checked against raw/terminal evidence and exact prose substrings.
claims=[]
def claim(line,fragment,actual,expected):
 eq(actual,expected,'prose_derived_value');assert fragment in lines[line-1],(line,fragment)
 claims.append(dict(line=line,fragment=fragment,value=actual))
claim(5,'2340 个 B64 batch',counts['raw_batch_rows'],2340)
claim(5,'1746 个含历史 batch',prior['totals']['history_batches'],1746)
claim(5,'5238 条',len(allrows),5238)
source=next(s for s in pipe['stages'] if s['stage']=='source');cpu_stage=next(s for s in pipe['stages'] if s['stage']=='source_cpu')
claim(11,'进程3799',source['original_pid'],3799);claim(11,'11:53:48.389849',source['ended_at'],'2026-09-08T11:53:48.389849+08:00')
claim(11,'23330.506955秒',f"{source['elapsed_seconds']:.6f}",'23330.506955')
claim(11,'进程18174',cpu_stage['original_pid'],18174);claim(11,'11:53:52.275165',cpu_stage['ended_at'],'2026-09-08T11:53:52.275165+08:00')
claim(11,'30420项',prior['totals']['remote_reported_statistic_checks'],30420);claim(11,'43688064个',prior['totals']['distance_elements'],43688064)
claim(11,'25份',len(manifest['files']),25);claim(11,'53366938字节',sum(p['bytes'] for p in manifest['files']),53366938)
claim(67,'有23条为负',sum(negative.values()),23)
claim(67,'Transformer为21条',sum(v for k,v in negative.items() if k[1:] == ('control','transformer')),21)
claim(67,'control的Mamba为1条',sum(v for k,v in negative.items() if k[1:] == ('control','mamba')),1)
claim(67,'fresh_memory的Transformer为1条',sum(v for k,v in negative.items() if k[1:] == ('fresh_memory','transformer')),1)
claim(67,'余弦全部仍为正',all(p['task_total_both_cosine']>0 for p in allrows),True)
for fragment,key in [('13716','available_history_group_exposures'),('12498','selected_vjp_group_exposures'),('1218','zero_upstream_group_skip_exposures'),('901','batches_with_zero_upstream_group_skip'),('58041','age_expired_record_exposures'),('51948','current_record_duplicate_exposures'),('11871','excluded_current_history_exposures')]:claim(96,fragment,prior['totals'][key],int(fragment))
claim(96,'历史集合372',max(s['queue_and_vjp_coverage']['maximum_selected_history_records'] for s in prior['states']),372)
claim(96,'队列408',max(s['queue_and_vjp_coverage']['maximum_queue_after_update'] for s in prior['states']),408)
claim(96,'实际年龄8',max(s['queue_and_vjp_coverage']['maximum_selected_history_age'] for s in prior['states']),8)
claim(96,'容量淘汰始终0',prior['totals']['capacity_evicted_record_exposures'],0)
claim(98,'801024',64*prior['totals']['selected_vjp_group_exposures']+9*64*2,801024)
claim(98,'16087.170 MiB',f"{max(s['peak_allocated_mib'] for s in prior['states']):.3f}",'16087.170')
claim(102,'2.02738058622413e-10',prior['maxima']['gradient_sum_norm_closure'],2.02738058622413e-10)
claim(102,'6.105141938845793e-5',max(p['relative_l2_error'] for p in prior['direct_proofs']),6.105141938845793e-5)

# Prior Q1 claims remain explicitly inherited, not re-evaluated retrieval in this diagnostic.
q=js(R/'evidence/msvr310_fresh_coordinate_complete_q1_20260908/q1/summary.json')
prior_report=read(R/'results/MSVR310_FRESH_COORDINATE_V1_Q1_2026-09-08.md').decode('utf-8')
assert '0/5' in prior_report and '-0.07158042' in prior_report
assert '−0.07158042pp' in text and '均0/5' in text
eq(f"{q['comparison']['matched_gains_mAP']['fused']:.8f}",'-0.07158042','inherited_q1_gain_value')
eq(len(q['comparison']['paired_checks']),5,'inherited_q1_paired_gate_count')
eq(sum(q['comparison']['paired_checks'].values()),0,'inherited_q1_paired_gate_passes')
eq(len(q['comparison']['endpoints']['fresh_memory']['scientific_checks']),5,'inherited_q1_signal_gate_count')
eq(sum(q['comparison']['endpoints']['fresh_memory']['scientific_checks'].values()),0,'inherited_q1_signal_gate_passes')

# Independent PDF extraction compares grid positions, not just the value multiset.
pdf=M/'figures/msvr310_history_candidate_gradients_source.pdf';pdf_bytes=read(pdf);doc=pymupdf.open(stream=pdf_bytes,filetype='pdf');eq(len(doc),1,'pdf_page_count')
page=doc[0];spans=[span for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans']]
outside=[s for s in spans if s['bbox'][0]<0 or s['bbox'][1]<0 or s['bbox'][2]>page.rect.width or s['bbox'][3]>page.rect.height];eq(len(outside),0,'pdf_out_of_page_spans')
means=[];ns=[]
for s in spans:
 value=s['text'].strip().replace('\u2212','-');x0,y0,x1,y1=s['bbox'];p=dict(value=value,x=(x0+x1)/2,y=(y0+y1)/2)
 if re.fullmatch(r'-?\d+\.\d{3}',value):means.append(p)
 if re.fullmatch(r'\(n=\d+\)',value):ns.append(p)
eq(len(means),108,'pdf_mean_count');eq(len(ns),108,'pdf_n_count')
for j,field in enumerate(fields):
 side=j%2;lower=j//2
 own=sorted([p for p in means if (p['x']>=page.rect.width/2)==bool(side) and (p['y']>=page.rect.height/2)==bool(lower)],key=lambda p:(round(p['y'],1),p['x']))
 nown=sorted([p for p in ns if (p['x']>=page.rect.width/2)==bool(side) and (p['y']>=page.rect.height/2)==bool(lower)],key=lambda p:(round(p['y'],1),p['x']))
 eq(len(own),27,'pdf_panel_means');eq(len(nown),27,'pdf_panel_n')
 expected=[(s,role) for s in prior['states'] for role in role_labels]
 for mp,np,(state,role) in zip(own,nown,expected):
  values=[p[field] for p in allrows if (p['fold'],p['state'],p['role'])==(state['fold'],state['state'],role)]
  eq(mp['value'],f'{statistics.fmean(values):.3f}','pdf_mean_at_grid_position');eq(np['value'],f'(n={len(values)})','pdf_n_at_grid_position')
  assert abs(mp['x']-np['x'])<1 and 0<np['y']-mp['y']<12
eq(len(page.get_images()),8,'pdf_embedded_images')
validation=js(M/'figures/pdf_validation.json')
eq(validation['pdf_sha256'],hashlib.sha256(pdf_bytes).hexdigest(),'pdf_validation_sha');eq(validation['render_sha256'],hashlib.sha256(read(M/'figures/pdf_inspection.png')).hexdigest(),'pdf_render_sha')
for key,value in dict(pages=1,checked_mean_cells=108,checked_n_cells=108,embedded_images=8,out_of_page_text_spans=0,page_size_points=[page.rect.width,page.rect.height]).items():rec(value,validation[key],'pdf_validation_fields')
render=O/'independent_pdf_render.png';page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False).save(render)
eq(hashlib.sha256(render.read_bytes()).hexdigest(),validation['render_sha256'],'independent_pdf_render_sha')
claim(106,'108格',len(means),108);claim(106,'20952个',len(allrows)*4,20952);claim(106,'108个n',len(ns),108);claim(106,'8个嵌入图像',len(page.get_images()),8)
tracker_path=R/'refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_TRACKER.md';tracker=read(tracker_path).decode('utf-8')
assert '| COMPLETE exit0 11:53:48; 2340batches, optimizer0 |' in tracker
assert '| PASS exit0 11:53:52; additional13-statistics30420checks PASS |' in tracker
assert 'postprocessing11:55:58 exit0' in tracker
completion=js(M/'processing/completion.json');eq(completion['finished_at'],'2026-09-08T11:55:58.161377+08:00','processing_chronology')
assert 'Full audit and next-hypothesis registration remain manual evidence-dependent steps.' in tracker
read(Path(__file__))
result=dict(status='PASS_COMPLETE_REPORT_MIRROR_AND_POSITIONAL_PDF_CHECKS',generated_at=datetime.now(timezone.utc).isoformat(),counts=dict(counts),report_sha256=hashlib.sha256(read(report_path)).hexdigest(),tracker_sha256=hashlib.sha256(read(tracker_path)).hexdigest(),global_statistics=global_stats,negative_current_both_groups=[dict(fold=k[0],state=k[1],role=k[2],count=v) for k,v in sorted(negative.items())],prose_claim_checks=claims,pdf=dict(pages=1,mean_cells=108,n_cells=108,grid_position_validated=True,embedded_images=8,out_of_page_text_spans=0,independent_render=str(render),pymupdf_version=pymupdf.VersionBind),scope='All four report tables and reported aggregate/prose values checked against raw rows; complete mirrored bytes and positional PDF content independently checked. Inherited prior-Q1 negative gain is referenced, not remeasured.',wording_note='Report line112 uses learning direction; prefer recorded role-block gradient direction to avoid ambiguity with optimizer directions, although line33 already disclaims AdamW update interpretation.',input_hashes=inputs)
assert '实际改变所记录的角色参数块梯度方向' in lines[111]
assert 'WARN' in lines[2] and 'provisional' in lines[2]
assert 'CLOSED_WARN (same-family/provisional)' in tracker
result['wording_note']='RESOLVED: the final report line112 explicitly says recorded role-block gradient direction. Closure-only publication edits retain WARN/same-family/provisional and do not alter source numbers.'
result['publication_snapshot_deltas']=dict(report_lines=[3,9,112,118],tracker_lines=[15,35],scientific_number_changes=0,source_artifact_changes=0)
(O/'independent_claim_checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','prose_claim_checks']},indent=2,ensure_ascii=False))
