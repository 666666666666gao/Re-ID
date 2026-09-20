import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
os.nice(10)
from pathlib import Path
import hashlib,json,subprocess,datetime
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
commit='1381639f778f77f124a2726ee55c07610092a438'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
checks=[];copies={};visited=set();configs={}
def bind(p,wanted,origin):
    p=Path(p); actual=sha(p)
    checks.append(dict(path=str(p),registered_sha256=wanted,actual_sha256=actual,match=actual==wanted,origin=origin))
    assert actual==wanted,(p,wanted,actual)
def visit(p):
    p=Path(p)
    if p in visited:return
    visited.add(p);d=json.loads(p.read_bytes());configs[str(p)]=d;copies[str(p)]=p.read_text()
    for key in ('project_file_sha256','project_source_file_sha256','fixed_file_sha256'):
        for name,value in d.get(key,{}).items():
            q=repo/name if not name.startswith('/') else Path(name)
            bind(q,value,str(p)+':'+key)
            if q.suffix in ('.py','.json','.md','.yaml','.yml'):copies[str(q)]=q.read_text()
    for key in ('previous_config','coordinate_config','memory_config','base_config'):
        if key in d:visit(repo/d[key])
    if 'BASELINE' in d:
        b=d['BASELINE'];bind(repo/b['CONFIG'],b['CONFIG_SHA256'],str(p));visit(repo/b['CONFIG'])
        bind(b['SUMMARY'],b['SUMMARY_SHA256'],str(p));copies[b['SUMMARY']]=Path(b['SUMMARY']).read_text()
    if 'SOURCE_METADATA' in d:
        m=d['SOURCE_METADATA'];bind(repo/m['PATH'],m['SHA256'],str(p));copies[str(repo/m['PATH'])]=(repo/m['PATH']).read_text()
    if 'protocol' in d:
        bind(repo/d['protocol'],d['protocol_sha256'],str(p));copies[str(repo/d['protocol'])]=(repo/d['protocol']).read_text()
    if 'signal_source_file_sha256' in d:
        for name,value in d['signal_source_file_sha256'].items():
            bind(Path(d['signal_source'])/name,value,str(p));copies[str(Path(d['signal_source'])/name)]=(Path(d['signal_source'])/name).read_text()
        actual=subprocess.check_output(['git','-C',d['signal_source'],'rev-parse','HEAD'],text=True).strip()
        assert actual==d['signal_commit']
        diff=hashlib.sha256(subprocess.check_output(['git','-C',d['signal_source'],'diff','--binary'])).hexdigest()
        assert diff==d['signal_diff_sha256']
        bind(d['clip_weight'],d['clip_weight_sha256'],str(p))
cfg=repo/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json'
visit(cfg)
support=repo/json.loads(cfg.read_bytes())['source_support'];copies[str(support)]=support.read_text()
for name,wanted in json.loads(support.read_bytes())['input_bindings'].items():bind(repo/name,wanted,str(support))
executed=[]
for p in sorted(set(x['path'] for x in checks)):
    q=Path(p)
    if q.is_relative_to(repo):
        data=subprocess.check_output(['git','show',commit+':'+q.relative_to(repo).as_posix()],cwd=repo)
        actual=sha(q);expected=hashlib.sha256(data).hexdigest()
        executed.append(dict(path=p,current_sha256=actual,executed_blob_sha256=expected,match=actual==expected))
for name in ('t0.json','t0.log','pipeline.json','m0.log','m0_cpu.json','m0_cpu.log','m0/summary.json'):
    copies[str(run/name)]=(run/name).read_text()
for p in sorted((run/'m0').rglob('*')):
    if p.is_file() and p.suffix in ('.json','.jsonl','.log'):copies[str(p)]=p.read_text()
inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((run/'m0').rglob('*')) if p.is_file()]
print(json.dumps(dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),cwd=str(repo),
    execution_commit=commit,observed_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    registered_bindings=checks,executed_bindings=executed,configs=configs,text_inputs=copies,m0_inventory=inventory,
    model_forwards=0,optimizer_updates=0,images_opened=0,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cpu_threads=2)))
