from pathlib import Path
from datetime import datetime
import ast,glob,hashlib,json,os.path as osp,re,subprocess,time
root=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
output=root/'evidence/trifusion_rgbnt100_main_v1_label_protocol_verification_20260906.json'
assert not output.exists()
started=time.perf_counter()
cfg=json.loads((root/'configs/RGBNT100/Signal-source-oof-v1-r2.json').read_text())
source=Path(cfg['signal_source'])/'data/datasets/RGBNT100.py'
archive=root/'evidence/rgbnt100_signal_source_text_20260906/data/datasets/RGBNT100.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert source.read_bytes()==archive.read_bytes()
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RGBNT100')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_process_dir')
namespace={'glob':glob,'osp':osp,'re':re}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
protocol_path=root/'protocols/rgbnt100_official_main_v1.json'
p=json.loads(protocol_path.read_text())
assert p['counts']=={'train':8675,'query':1715,'gallery':8575}
matched={}
for split,directory in {'train':'bounding_box_train','query':'query','gallery':'bounding_box_test'}.items():
 rows=namespace['_process_dir'](None,str(Path(cfg['dataset_root'])/'rgbir'/directory),False)
 a={(Path(path).relative_to(cfg['dataset_root']).as_posix(),identity,camera,view) for path,identity,camera,view in rows}
 b={(r['path'],r['identity'],r['camera'],r['view']) for r in p['records'][split]}
 assert len(a)==len(rows)==p['counts'][split] and a==b
 matched[split]=len(a)
gallery=p['records']['gallery']
counts=[]
for query,mask in zip(p['records']['query'],p['query_rows'],strict=True):
 excluded=[];positive=[];negative=0
 for position,row in enumerate(gallery):
  if row['identity']!=query['identity']:
   negative+=1
  elif row['camera']==query['camera']:
   excluded.append(position)
  else:
   positive.append(position)
 assert mask['query_index']==query['index'] and mask['identity']==query['identity']
 assert mask['positive_gallery_positions']==positive and mask['excluded_gallery_positions']==excluded
 assert mask['valid_positive_count']==len(positive)>0
 assert mask['negative_count']==negative
 assert mask['retained_gallery_count']==len(gallery)-len(excluded)==len(positive)+negative
 counts.append({'query_index':query['index'],'identity':query['identity'],'positive':len(positive),'excluded':len(excluded),'negative':negative})
assert len(counts)==1715
result={'observed_at':datetime.now().astimezone().isoformat(),
'status':'PASS_ALL18965_AUTHOR_LABEL_ROWS_AND1715_COMPLETE_MASKS',
'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
'author_parser_source':str(source),'author_parser_sha256':sha(source),'archived_source_sha256':sha(archive),
'author_parser_execution':'Original _process_dir AST method, relabel=False, explicit installed split directories; no author class initialization or model imports',
'protocol_sha256':sha(protocol_path),'matched_label_records':matched,
'query_masks_verified':1715,'gallery_positions_per_query':8575,'total_query_gallery_label_comparisons':1715*8575,
'positive_count_range':[min(x['positive'] for x in counts),max(x['positive'] for x in counts)],
'all_query_counts':counts,'model_forwards':0,'training_updates':0,'image_decodes':0,
'elapsed_seconds':time.perf_counter()-started}
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='all_query_counts'},ensure_ascii=False))
