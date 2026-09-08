import pathlib,json,hashlib,shutil
A=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
x=json.loads((A/'remote_collect_primary_text.stdout.json').read_text())
hashes=json.loads((A/'audited_input_hashes.json').read_text())
text_index=[]
for path,txt in x['texts'].items():
    tag='repo' if '/TriFusion-ReID/' in path else 'signal'
    rel=path.split('/TriFusion-ReID/' if tag=='repo' else '/Signal-cd1b0a6/',1)[1]
    p=A/'remote_texts'/tag/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(txt,encoding='utf-8',newline='')
    digest=hashlib.sha256(p.read_bytes()).hexdigest();hashes[path]=digest
    text_index.append({'remote':path,'snapshot':str(p),'sha256':digest,'bytes':p.stat().st_size,'lines':len(txt.splitlines())})
for pin in x['pins']:
    hashes[pin['path']]=pin['sha256']
    if '/TriFusion-ReID/' in pin['path']:
        rel=pin['path'].split('/TriFusion-ReID/',1)[1];p=R/rel
        if p.is_file():
            target=A/'snapshots'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
report={k:v for k,v in x.items() if k not in ('texts','baseline_summary_top','run_files')}
report['text_index']=text_index
report['run_file_count']=len(x['run_files'])
report['run_file_total_bytes']=sum(r['bytes'] for r in x['run_files'])
report['git_source_version_failures']=[{'commit':v['commit'],'mismatches':[f for f in v['files'] if not f['equal']]} for v in x['git_source_versions']]
(A/'remote_primary_bindings.json').write_text(json.dumps(report,indent=2)+'\n')
(A/'audited_input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
p=json.loads((A/'remote_texts/repo/protocols/msvr310_train_oof_v1.json').read_text())
print(json.dumps({'pins':len(x['pins']),'failed_pins':[r for r in x['pins'] if not r['match']],'git_version_mismatches':report['git_source_version_failures'],'protocol_top_keys':list(p),'first_record':p['records'][0],'fold_metadata':[{k:v for k,v in f.items() if k not in ('source_ids','heldout_ids','source_record_indices','gallery_record_indices','query_rows','source_label_map')} for f in p['folds']],'baseline_scope':x['baseline_scope'],'text_index':text_index},indent=2))
