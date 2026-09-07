from pathlib import Path
import csv,json,hashlib,math,statistics
p=Path('C:/Users/gb/.codex_tmp/v29_joint_similarity_r2_terminal_20260907')
proof=json.loads((p/'analysis.json').read_bytes())
rows={}
for name,e in proof['exports'].items():
 f=p/name
 assert len(f.read_bytes())==e['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256']
 rr=list(csv.DictReader(f.open(encoding='utf-8',newline='')))
 assert len(rr)==e['rows']
 rows[name]=rr
for r in rows['pair_affine.csv']:
 if r['rmse']:
  n=float(r['n']);sx=float(r['sum_x']);sy=float(r['sum_y']);xx=float(r['sum_x2']);xy=float(r['sum_xy']);yy=float(r['sum_y2'])
  slope=(xy-sx*sy/n)/(xx-sx*sx/n)
  assert abs(slope-float(r['slope']))<1e-12
  assert abs((sy-slope*sx)/n-float(r['intercept']))<1e-12
global_pair=[{k:r[k] for k in ('fold','view','pair_kind','count','slope','intercept','r2','rmse','fixed_08_02_rmse')} for r in rows['pair_affine.csv'] if r['stratum']=='all']
global_rel=[]
for r in rows['relation_residual.csv']:
 if r['stratum']=='all':
  n=int(r['count']); mx=float(r['before_margin_sum'])/n;my=float(r['after_margin_sum'])/n
  global_rel.append(dict(fold=r['fold'],view=r['view'],protocol=r['protocol'],n=n,mx=mx,my=my,mean_change=my-mx,affine_residual_mean=float(r['affine_residual_sum'])/n,affine_residual_rms=math.sqrt(float(r['affine_residual_square_sum'])/n),fraction_change_squared_explained=1-float(r['affine_residual_square_sum'])/float(r['margin_change_square_sum']),nonpositive_before=int(r['before_nonpositive_count']),nonpositive_after=int(r['after_nonpositive_count'])))
identities=[]
for view in ('original','registered_style'):
 for stratum in ('all','active','inactive'):
  for protocol in ('identity','cross_camera'):
   selected=[r for r in rows['identity_relation_residual.csv'] if (r['view'],r['stratum'],r['protocol'])==(view,stratum,protocol)]
   assert len(selected)==282
   active=[r for r in selected if int(r['count'])]
   for fold in ('0','1','2'):
    g=next(r for r in rows['relation_residual.csv'] if (r['view'],r['stratum'],r['protocol'],r['fold'])==(view,stratum,protocol,fold))
    members=[r for r in selected if r['fold']==fold]
    for field in ('count','before_margin_sum','after_margin_sum','affine_residual_sum','affine_residual_square_sum'):
     assert math.isclose(sum(float(r[field]) for r in members),float(g[field]),rel_tol=1e-10,abs_tol=1e-6)
   identities.append(dict(view=view,stratum=stratum,protocol=protocol,eligible=len(active),zero=len(selected)-len(active),positive_mean_residual=sum(float(r['affine_residual_sum'])>0 for r in active),negative_mean_residual=sum(float(r['affine_residual_sum'])<0 for r in active),mean_margin_decrease=sum(float(r['after_margin_sum'])<float(r['before_margin_sum']) for r in active)))
batch_summary=[]
for fold in ('0','1','2'):
 for view in ('original','registered_style'):
  selected=[r for r in rows['batch_affine.csv'] if r['fold']==fold and r['view']==view]
  batch_summary.append(dict(fold=fold,view=view,count=len(selected),**{k:{'min':min(float(r[k]) for r in selected),'median':statistics.median(float(r[k]) for r in selected),'max':max(float(r[k]) for r in selected)} for k in ('slope','intercept','r2','rmse')}))
out=dict(status='PASS_ALL_CSV_HASHES_ROWS_AFFINE_ARITHMETIC_AND_IDENTITY_SUMS',all_pair_classes=global_pair,all_protocol_relations=global_rel,identity_summary=identities,batch_summary=batch_summary)
(p/'local_digest.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))

