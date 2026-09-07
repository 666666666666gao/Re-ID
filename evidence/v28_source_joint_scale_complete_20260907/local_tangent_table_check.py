from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math
p=Path('C:/Users/gb/.codex_tmp/v28_source_joint_scale_terminal_20260907')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
s=json.loads((p/'supplement_tangent_geometry.json').read_bytes())
a=json.loads((p/'complete_source_scale_verification.json').read_bytes())
intake=json.loads((p/'supplement_tangent_intake.json').read_bytes())
assert intake['all_bytes_and_sha256_match']
for name,row in intake['files'].items():
 assert sha(p/name)==row['sha256'] and (p/name).stat().st_size==row['bytes']
assert s['complete_verification_sha256']==sha(p/'complete_source_scale_verification.json')
assert s['checked_slots']==1935360 and s['zero_c']==0
for filename,key,rows in [('supplement_tangent_all_strata.csv','cells_csv_sha256',s['all_cells']),
                          ('supplement_tangent_all_source_identities.csv','identity_csv_sha256',s['all_identity_rows'])]:
 assert sha(p/filename)==s[key]
 with (p/filename).open(encoding='utf-8',newline='') as f: other=list(csv.DictReader(f))
 assert len(other)==len(rows)
 for raw,row in zip(other,rows,strict=True):
  assert set(raw)==set(row)
  for k,v in row.items(): assert raw[k]==v if isinstance(v,str) else float(raw[k])==v
cells={(r['fold'],r['view'],r['role'],r['modality'],r['stratum']):r for r in s['all_cells']}
people=defaultdict(list)
for r in s['all_identity_rows']: people[(r['fold'],r['view'],r['role'],r['modality'])].append(r)
old={(r['fold'],r['view'],r['role'],r['modality'],r['stratum']):r for r in a['all_cells']}
error=0.
def combine(target,rows):
 global error
 n=sum(r['observations'] for r in rows);assert n==target['observations']
 for metric in s['metrics']:
  for suffix,expected in [('mean',sum(r[metric+'_mean']*r['observations'] for r in rows)/n),
                          ('min',min(r[metric+'_min'] for r in rows)),('max',max(r[metric+'_max'] for r in rows))]:
   actual=target[metric+'_'+suffix];error=max(error,abs(actual-expected)/(1+abs(expected)))
   assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10)
for key,row in cells.items():
 assert row['observations']==old[key]['observations']
 assert -1e-10<=row['tangent_energy_fraction_min']<=row['tangent_energy_fraction_max']<=1+1e-10
 assert 0<=row['angle_y_h_degrees_min']<=row['angle_y_h_degrees_max']<=180
 if key[-1]=='all':
  combine(row,[cells[(*key[:-1],v)] for v in ['plan_active','plan_inactive']])
  assert len(people[key[:-1]])==94;combine(row,people[key[:-1]])
 if key[-1]=='plan_inactive':
  other=cells[(key[0],'registered_style' if key[1]=='original' else 'original',*key[2:])]
  assert {k:v for k,v in row.items() if k!='view'}=={k:v for k,v in other.items() if k!='view'}
for view,row in s['overall_by_view'].items(): combine(row,[v for k,v in cells.items() if k[1]==view and k[-1]=='all'])
result={'status':'PASS_LOCAL_COMPLETE_TANGENT_TABLES_AND_AGGREGATION','checked_cells':162,'checked_identity_rows':5076,
        'checked_slots':1935360,'maximum_scaled_aggregation_error':error,'all_csv_values_equal_json':True,
        'all_identity_stratum_and_overall_aggregations_match':True,'all_inactive_views_equal':True,
        'supplement_sha256':sha(p/'supplement_tangent_geometry.json'),'local_model_image_npy_calls':0,'new_optimizer_updates':0,
        'boundary':'Scalar table verification only; all raw scalar derivations ran on server. No external independent audit.'}
out=p/'local_complete_tangent_aggregation.json';assert not out.exists()
out.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
