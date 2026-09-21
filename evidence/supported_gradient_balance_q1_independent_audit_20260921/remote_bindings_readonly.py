from pathlib import Path
import hashlib,json,subprocess,time,platform
REPO=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def js(p):return json.loads(p.read_bytes())
summary=js(RUN/'q1/summary.json');pipeline=js(RUN/'pipeline.json');cpu=js(RUN/'q1_cpu_arithmetic_recheck/verification.json')
todo=[REPO/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json'];visited=set();bindings=[];repo_paths=set();configs={}
while todo:
    path=todo.pop()
    if path in visited:continue
    visited.add(path);c=js(path);configs[str(path.relative_to(REPO))]=c;repo_paths.add(path)
    for k in ('project_file_sha256','project_source_file_sha256','fixed_file_sha256'):
        for name,expected in c.get(k,{}).items():
            p=Path(name) if k=='fixed_file_sha256' else REPO/name
            actual=sha(p);assert actual==expected,(str(p),actual,expected)
            bindings.append(dict(config=str(path.relative_to(REPO)),kind=k,path=str(p),sha256=actual,bytes=p.stat().st_size))
            if k!='fixed_file_sha256':
                repo_paths.add(p)
                if name.startswith('configs/') and name.endswith('.json'):todo.append(p)
    for k in ('previous_config','base_config','coordinate_config','memory_config'):
        if k in c:todo.append(REPO/c[k])
    if 'BASELINE' in c:
        todo.append(REPO/c['BASELINE']['CONFIG'])
        for p,expected in ((REPO/c['BASELINE']['CONFIG'],c['BASELINE']['CONFIG_SHA256']),(Path(c['BASELINE']['SUMMARY']),c['BASELINE']['SUMMARY_SHA256'])):
            assert sha(p)==expected;bindings.append(dict(config=str(path.relative_to(REPO)),kind='baseline',path=str(p),sha256=expected,bytes=p.stat().st_size))
    if 'protocol' in c:assert sha(REPO/c['protocol'])==c['protocol_sha256']
    if 'SOURCE_METADATA' in c:assert sha(REPO/c['SOURCE_METADATA']['PATH'])==c['SOURCE_METADATA']['SHA256']
base=js(REPO/'configs/MSVR310/Signal-source-oof-v1.json');src=Path(base['signal_source'])
for name,expected in base['signal_source_file_sha256'].items():
    p=src/name;assert sha(p)==expected;bindings.append(dict(kind='upstream_signal',path=str(p),sha256=expected,bytes=p.stat().st_size))
assert subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()==base['signal_commit']
assert hashlib.sha256(subprocess.check_output(['git','-C',str(src),'diff','--binary'])).hexdigest()==base['signal_diff_sha256']
assert sha(Path(base['clip_weight']))==base['clip_weight_sha256']
git_rows=[]
for p in sorted(repo_paths):
    rel=p.relative_to(REPO).as_posix();h=sha(p)
    row=dict(path=rel,current_sha256=h)
    for tag,commit in (('pipeline',pipeline['code_commit']),('q1',summary['project_commit'])):
        data=subprocess.check_output(['git','-C',str(REPO),'show',commit+':'+rel])
        row[tag+'_blob_sha256']=hashlib.sha256(data).hexdigest();assert row[tag+'_blob_sha256']==h,(rel,tag)
    git_rows.append(row)
verified=[]
for name,proof in cpu['files'].items():
    p=Path(name);assert p.is_file();h=sha(p);assert h==proof['sha256'] and p.stat().st_size==proof['bytes']
    verified.append(dict(path=name,bytes=p.stat().st_size,sha256=h))
assert cpu['summary_sha256']==sha(RUN/'q1/summary.json')
assert summary['m0_receipt_sha256']==sha(RUN/'m0/summary.json') and summary['m0_verification_sha256']==sha(RUN/'m0_cpu.json')
m0=js(RUN/'m0/summary.json');m0cpu=js(RUN/'m0_cpu.json')
assert m0['status']=='PASS_ENGINEERING_ONLY' and m0cpu['summary_sha256']==summary['m0_receipt_sha256']
assert m0cpu['status']=='PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_M0'
for f in summary['folds']:
    for e in ('control','balanced'):assert f['endpoints'][e]['initialization']==m0['folds'][f['fold']]['endpoints'][e]['initialization']
inventory=[]
for p in RUN.rglob('*'):
    if p.is_file():inventory.append(dict(path=str(p.relative_to(RUN)),bytes=p.stat().st_size))
texts={str(src/name):(src/name).read_text() for name in ('utils/metrics.py','data/datasets/msvr310.py','configs/MSVR310/Signal.yml')}
texts[str(REPO/'tools/recheck_msvr_supported_balance_sqrt.py')]=(REPO/'tools/recheck_msvr_supported_balance_sqrt.py').read_text()
protocol=js(REPO/base['protocol']);root=Path(base['dataset_root'])
for r in protocol['records']:
    for p in r['paths']:assert p.startswith('bounding_box_train/') and (root/p).is_file()
print(json.dumps(dict(status='PASS_REMOTE_CURRENT_EXECUTION_BINDINGS',python=platform.python_version(),pipeline_commit=pipeline['code_commit'],q1_observed_commit=summary['project_commit'],current_head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),bindings=bindings,git_rows=git_rows,verified_files=verified,original_pipeline=pipeline,posthoc_receipt=js(RUN/'q1_cpu_arithmetic_recheck/receipt.json'),posthoc_log=(RUN/'q1_cpu_arithmetic_recheck/verification.log').read_text(),original_cpu_log=(RUN/'q1_cpu.log').read_text(),m0_binding_only=dict(summary_sha256=sha(RUN/'m0/summary.json'),cpu_sha256=sha(RUN/'m0_cpu.json'),status=m0['status']),run_file_inventory=inventory,storage_total_bytes=sum(x['bytes'] for x in inventory),texts=texts,official_image_reads=0,source_path_existence_checks=3096,model_forwards=0,optimizer_updates=0,downloaded_binary_files=0)))
