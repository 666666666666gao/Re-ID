from pathlib import Path
import json,csv,hashlib,math
OUT=Path(__file__).parent;D=Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_complete_20260909');REPO=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee');C=Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_ranking_analysis_20260909')
def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(v,w):
 if isinstance(w,bool):assert v==str(w),(v,w)
 elif isinstance(w,(int,float)):assert math.isclose(float(v),w,rel_tol=0,abs_tol=1e-10),(v,w)
 else:assert v==w,(v,w)
records={}
for fi in range(3):
 for e in ['control','smooth_ap']:
  p=D/'q1'/f'fold_{fi}_{e}';ret=load(p/'receipt.json')['retrieval'];ranks=load(p/'rankings.json');gallery=ret['gallery_manifest']
  for output,allr in ranks.items():
   for q,rank in zip(ret['query_rows'],allr):
    legal=[gallery[i] for i in rank if not(gallery[i]['identity']==q['identity'] and gallery[i]['scene']==q['scene'])];pos=[i+1 for i,r in enumerate(legal) if r['identity']==q['identity']];neg=next(r for r in legal if r['identity']!=q['identity'])
    records[fi,e,output,q['record_index']]=dict(fold=fi,record_index=q['record_index'],identity=q['identity'],scene=q['scene'],positive_count=len(pos),ap=sum((i+1)/p for i,p in enumerate(pos))/len(pos),first_match_rank=pos[0],last_positive_rank=pos[-1],nearest_negative_record=neg['index'],nearest_negative_identity=neg['identity'],nearest_negative_scene=neg['scene'])
count=0
with (C/'all3000_query_output_changes.csv').open(newline='',encoding='utf8') as f:
 for row in csv.DictReader(f):
  fi=int(row['fold']);q=int(row['record_index']);o=row['output'];a=records[fi,'control',o,q];b=records[fi,'smooth_ap',o,q];expected={k:a[k] for k in ['fold','record_index','identity','scene','positive_count']};expected.update(output=o,delta_ap_pp=(b['ap']-a['ap'])*100,rank1_repaired=a['first_match_rank']>1 and b['first_match_rank']==1,rank1_new_error=a['first_match_rank']==1 and b['first_match_rank']>1)
  for tag,r in [('control',a),('candidate',b)]:expected.update({tag+'_'+k:v for k,v in r.items() if k not in ['fold','record_index','identity','scene','positive_count']})
  assert set(row)==set(expected)
  for k,v in expected.items():same(row[k],v)
  count+=1
assert count==3000
identities=0
with (C/'all300_identity_output_changes.csv').open(newline='',encoding='utf8') as f:
 for row in csv.DictReader(f):
  identity=int(row['identity']);o=row['output'];aa=[r['ap'] for (fi,e,out,q),r in records.items() if e=='control' and out==o and r['identity']==identity];bb=[r['ap'] for (fi,e,out,q),r in records.items() if e=='smooth_ap' and out==o and r['identity']==identity];a=sum(aa)/len(aa)*100;b=sum(bb)/len(bb)*100
  for k,v in dict(identity=identity,output=o,query_count=len(aa),delta_mAP_pp=b-a,control_mAP=a,candidate_mAP=b).items():same(row[k],v)
  identities+=1
assert identities==300
snapshots=0
for name,expected in load(OUT/'input_hashes_final.json').items():
 p=Path(name);relative=Path('repo')/p.relative_to(REPO) if p.is_relative_to(REPO) else Path('local_text')/p.relative_to(Path(r'C:\Users\gb\.codex_tmp'));copy=OUT/'snapshots'/relative;assert copy.is_file() and copy.stat().st_size==expected['bytes'] and sha(copy)==expected['sha256'];snapshots+=1
report=load(OUT/'EXPERIMENT_AUDIT.json');assert report['verdict']=='WARN' and report['deterministic_checks_status']=='pass' and report['scientific_qualification']=='Q1_FAIL'
assert report['review_independence']=='same-family' and report['acceptance_status']=='provisional'
result=dict(status='PASS',full_query_csv_rows_and_all_fields=count,full_identity_csv_rows_and_all_fields=identities,input_snapshots_rehashed=snapshots,report_schema_and_verdict_consistent=True)
with (OUT/'package_validation.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
