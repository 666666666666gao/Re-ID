from pathlib import Path
import ast,hashlib,json

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
out=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_resume_source868_r2')
assert not out.exists();out.mkdir()
source=(repo/'tools/resume_signal_selection_reference.py').read_text()
tree=ast.parse(source)
remote=Path('/data/gaob/Re-ID/Trifusion')
namespace=dict(ROOT=remote,Path=Path,ORIGINAL=remote/'logs/signal_selection_reference_v1_20261006_866',PREVIOUS=remote/'logs/signal_selection_reference_v1_20261006_867',NORMALIZATION=remote/'logs/selection_counter_record_normalization_20261006_868')
subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('origin','metadata_path')],type_ignores=[])
exec(compile(subset,'real-resume-origin-functions','exec'),namespace)
class Panel:
    @staticmethod
    def output_dir(c,d,s,p):return remote/'trained-model'/f'{c.name}_{p}_{s}_{d}'
namespace['panel']=Panel
campaign=remote/'logs/signal_selection_reference_v1_20261006_868'
rows=[]
for dataset in ('RGBNT201','MSVR310','RGBNT100'):
    for selection in ('global_only','masked','all_patch'):
        full=namespace['origin'](campaign,dataset,selection,'full')
        m0=namespace['origin'](campaign,dataset,selection,'m0')
        rows.append(dict(dataset=dataset,selection=selection,full=str(full),m0=str(m0),metadata_path=str(namespace['metadata_path'](campaign,dataset,selection))))
assert sum(r['full']!=str(campaign) for r in rows)==4
assert sum(r['m0']!=str(campaign) for r in rows)==4
assert all(r['full']==str(campaign) and r['m0']==str(campaign) for r in rows if r['dataset']=='RGBNT100')
assert next(r for r in rows if (r['dataset'],r['selection'])==('RGBNT201','masked'))['m0']==str(namespace['ORIGINAL'])
assert next(r for r in rows if (r['dataset'],r['selection'])==('MSVR310','global_only'))['metadata_path']==str(namespace['NORMALIZATION']/'MSVR310_optimizer_batch_metadata.jsonl')
assert sum('/selection_counter_record_normalization_' in r['metadata_path'].replace('\\','/') for r in rows)==1
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
text=ast.unparse(functions['main'])
assert text.index('administrative_reuse_acceptance.json')<text.index("retire_probe(campaign, 'MSVR310'")<text.index('shutil.disk_usage(ROOT).free >= STORAGE_BYTES')
assert any(isinstance(n,ast.If) and ast.unparse(n.test)=='(dataset, selection) in retained' and isinstance(n.body[0],ast.Continue) for n in ast.walk(functions['main']))
assert text.count("'--batch-metadata-paths'")==1
assert 'STORAGE_BYTES=6*360*1024**2+2*1024**3' in source
for name in ('run_signal_selection_reference.py','queue_signal_selection_reference.py','report_signal_selection_reference.py'):
    ast.parse((repo/'tools'/name).read_bytes())
facts=dict(status='PASS_ROOT_SOURCE_CHECK',rows=rows,reused_full=4,new_full=5,reused_m0=4,new_m0=5,fresh_initializers=9,
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),storage_budget_bytes=4412407808,
    boundary='Realorigin/metadatapath functions executed onCPU; AST orchestration check only, not fresh fullmodel/M0/runtime acceptance or independent review. Existing producer/consumer realmetadata CPU red/green regressions separately preserved.')
(out/'FACTS.json').write_text(json.dumps(facts,indent=2)+'\n');(out/'SOURCE.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(status=facts['status'],new_full=5,reused_full=4,new_m0=5,reused_m0=4),indent=2))
