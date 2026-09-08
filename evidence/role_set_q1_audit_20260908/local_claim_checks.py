"""Independent, complete text claim reconciliation; never imports author tools."""
import pathlib,json,csv,hashlib,collections,math,datetime,shutil
import numpy as np
A=pathlib.Path(__file__).resolve().parent
I=A.parent/'role_set_q1_complete_20260908'
EXEC=A.parent/'role_set_q1_executor_report_20260908'
RANK=A.parent/'role_set_q1_rankings_20260908'
def j(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float,np.number)) and not isinstance(a,bool):assert abs(float(a)-float(b))<1e-11,(path,a,b)
    else:assert a==b,(path,a,b)
def phase(rows,tr):
    entries=[]
    for row in rows:
        props=row['relation_objective']['proposals'];rec=row['record_indices']+[m['record_index'] for m in row['memory']];ids=row['identities']+[m['identity'] for m in row['memory']]
        for n,p in enumerate(props):entries.append((p,set(p),{rec[k] for k in p},{ids[k] for k in p},rec,ids,row['relation_objective']['extra_active_counts'][n]))
    c={'steps':len(rows),'anchor_exposures':len(entries),'memory_record_exposures':sum(len(r['memory']) for r in rows),
       'extra_active_hinge_exposures':sum(e[6] for e in entries),'anchors_with_extra_active_hinge':sum(e[6]>0 for e in entries),
       'selected_position_exposures':sum(len(e[1]) for e in entries),'extra_position_exposures':sum(len(e[1])-1 for e in entries),
       'extra_record_exposures':sum(len(e[2])-1 for e in entries),'extra_negative_identity_exposures':sum(len(e[3])-1 for e in entries),
       'anchors_with_extra_position':sum(len(e[1])>1 for e in entries),'anchors_with_extra_record':sum(len(e[2])>1 for e in entries),
       'anchors_with_extra_negative_identity':sum(len(e[3])>1 for e in entries),'position_duplicates_of_selected_records':sum(len(e[1])-len(e[2]) for e in entries),
       'selected_record_redundancy_within_identities':sum(len(e[2])-len(e[3]) for e in entries),
       'extra_positions_same_negative_identity_as_fused':sum(e[5][k]==e[5][e[0][0]] for e in entries for k in e[1]-{e[0][0]}),
       'extra_positions_other_negative_identity_than_fused':sum(e[5][k]!=e[5][e[0][0]] for e in entries for k in e[1]-{e[0][0]}),
       'extra_historical_positions':sum(k>=64 for e in entries for k in e[1]-{e[0][0]})}
    roles={}
    for ri,name in enumerate(('cnn','transformer','mamba'),1):
        roles[name]={'position_differs_from_fused':sum(e[0][ri]!=e[0][0] for e in entries),
                     'record_differs_from_fused':sum(e[4][e[0][ri]]!=e[4][e[0][0]] for e in entries),
                     'identity_differs_from_fused':sum(e[5][e[0][ri]]!=e[5][e[0][0]] for e in entries),
                     'position_exclusive_among_four_proposals':sum(e[0][ri] not in e[0][:ri]+e[0][ri+1:] for e in entries),
                     'identity_exclusive_among_four_proposals':sum(e[5][e[0][ri]] not in [e[5][p] for p in e[0][:ri]+e[0][ri+1:]] for e in entries)}
    losses={'actual_total':float(np.mean([s['loss'] for s in tr])), 'batch_hard':float(np.mean([r['statistics']['current_triplet'] for r in rows])),
            'expanded_hard':float(np.mean([r['relation_objective']['hard_loss'] for r in rows])),'role_set':float(np.mean([r['relation_objective']['role_set_loss'] for r in rows]))}
    result={'counts':c,'unique_positions_histogram':dict(collections.Counter(str(len(e[1])) for e in entries)),
            'unique_records_histogram':dict(collections.Counter(str(len(e[2])) for e in entries)),
            'unique_negative_identities_histogram':dict(collections.Counter(str(len(e[3])) for e in entries)),
            'role_proposal_exposures':roles,'mean_losses':losses,
            'mean_components':{k:float(np.mean([s['components'][k] for s in tr])) for k in tr[0]['components']}}
    return result,float(np.mean([1/len(e[1]) for e in entries]))
source=j(A.parent/'role_set_q1_source_analysis_20260908.json');aggregate=j(EXEC/'source_aggregate.json');summary=j(I/'q1/summary.json')
recomputed=[];primary={};phase_slices={'all_steps_1_260':(0,260),'warmup_steps_1_65':(0,65),'post_warmup_steps_66_260':(65,260),'last_five_epochs_steps_196_260':(195,260)}
for fi in range(3):
    for endpoint in ('control','role_set'):
        d=I/'q1'/f'fold_{fi}_{endpoint}';tr=j(d/'training.json');rows=[json.loads(line) for line in (d/'memory_steps.jsonl').read_text().splitlines()]
        primary[(fi,endpoint)]=(rows,tr);author=next(e for e in source['endpoints'] if e['fold']==fi and e['endpoint']==endpoint)
        phases={};weights={}
        for name,(lo,hi) in phase_slices.items():
            phases[name],weights[name]=phase(rows[lo:hi],tr['steps'][lo:hi]);compare(phases[name],author['phases'][name],f'{fi}/{endpoint}/{name}')
        cost=dict(fresh_role_record_forwards=tr['extra_fresh_role_record_forwards'],historical_vjp_record_forwards=tr['extra_history_vjp_record_forwards'],fit_epoch_seconds=sum(r['elapsed_seconds'] for r in tr['history']),peak_allocated_mib=tr['peak_allocated_mib'])
        compare(cost,author['cost'])
        recomputed.append(dict(fold=fi,endpoint=endpoint,phases=phases,hard_relation_weight=weights,cost=cost))
newaggregate={}
for endpoint in ('control','role_set'):
    es=[r for r in recomputed if r['endpoint']==endpoint];newaggregate[endpoint]={}
    for name in ('post_warmup_steps_66_260','last_five_epochs_steps_196_260'):
        p=[e['phases'][name] for e in es]
        counts={k:sum(x['counts'][k] for x in p) for k in p[0]['counts']}
        ids=collections.Counter()
        for x in p:ids.update(x['unique_negative_identities_histogram'])
        roles={role:{k:sum(x['role_proposal_exposures'][role][k] for x in p) for k in p[0]['role_proposal_exposures'][role]} for role in ('cnn','transformer','mamba')}
        losses={k:float(np.mean([x['mean_losses'][k] for x in p])) for k in p[0]['mean_losses']}
        val=dict(counts=counts,negative_identity_histogram=dict(ids),roles=roles,mean_losses=losses);compare(val,aggregate['aggregation'][endpoint][name]);newaggregate[endpoint][name]=val
    newaggregate[endpoint]['cost']=dict(fresh_role_record_forwards=sum(e['cost']['fresh_role_record_forwards'] for e in es),historical_vjp_record_forwards=sum(e['cost']['historical_vjp_record_forwards'] for e in es),fit_epoch_seconds=sum(e['cost']['fit_epoch_seconds'] for e in es),max_peak_allocated_mib=max(e['cost']['peak_allocated_mib'] for e in es))
    compare(newaggregate[endpoint]['cost'],aggregate['aggregation'][endpoint]['cost'])
ranking=j(RANK/'ranking_replay.json');output_names=('baseline_only','fused','cnn','transformer','mamba');detail={e:{o:[] for o in output_names} for e in ('control','role_set')};rank_elements=0
for fold in summary['folds']:
    for endpoint in ('control','role_set'):
        retrieval=fold['endpoints'][endpoint]['retrieval'];gal=retrieval['gallery_manifest'];ranks=j(I/'q1'/f"fold_{fold['fold']}_{endpoint}"/'rankings.json')
        for output in output_names:
            for query,order in zip(retrieval['query_rows'],ranks[output]):
                assert sorted(order)==list(range(len(gal)));rank_elements+=len(order);q=gal[query['gallery_position']]
                legal=[gal[pos] for pos in order if (gal[pos]['identity'],gal[pos]['scene'])!=(q['identity'],q['scene'])]
                positives=[n for n,g in enumerate(legal,1) if g['identity']==q['identity']];neg=next(g for g in legal if g['identity']!=q['identity'])
                detail[endpoint][output].append(dict(fold=fold['fold'],record_index=q['index'],identity=q['identity'],scene=q['scene'],ap=sum(n/r for n,r in enumerate(positives,1))/len(positives),first_match_rank=positives[0],last_positive_rank=positives[-1],positive_count=len(positives),nearest_negative_record=neg['index'],nearest_negative_identity=neg['identity'],nearest_negative_scene=neg['scene']))
assert rank_elements==2069520
queries=[];identities=[];paired_changes={}
for output in output_names:
    qrows=[]
    for old,new in zip(detail['control'][output],detail['role_set'][output]):
        key=('fold','record_index','identity','scene','positive_count');row={k:old[k] for k in key};assert all(old[k]==new[k] for k in key)
        row.update(output=output,delta_ap_pp=(new['ap']-old['ap'])*100,rank1_repaired=old['first_match_rank']>1 and new['first_match_rank']==1,rank1_new_error=old['first_match_rank']==1 and new['first_match_rank']>1)
        for tag,x in [('control',old),('candidate',new)]:row.update({tag+'_'+k:v for k,v in x.items() if k not in key})
        queries.append(row);qrows.append(row)
    changes=dict(ap_improved=sum(r['delta_ap_pp']>0 for r in qrows),ap_declined=sum(r['delta_ap_pp']<0 for r in qrows),ap_unchanged=sum(r['delta_ap_pp']==0 for r in qrows),rank1_repaired=sum(r['rank1_repaired'] for r in qrows),rank1_new_errors=sum(r['rank1_new_error'] for r in qrows))
    compare(changes,ranking['paired_changes'][output]);paired_changes[output]=changes
    for identity in sorted({r['identity'] for r in qrows}):
        subset=[r for r in qrows if r['identity']==identity]
        identities.append(dict(identity=identity,output=output,query_count=len(subset),delta_mAP_pp=float(np.mean([r['delta_ap_pp'] for r in subset])),control_mAP=float(np.mean([r['control_ap'] for r in subset])*100),candidate_mAP=float(np.mean([r['candidate_ap'] for r in subset])*100)))
for name,records in [('all3000_query_output_changes.csv',queries),('all300_identity_output_changes.csv',identities)]:
    author=list(csv.DictReader((RANK/name).open(newline='')));assert len(author)==len(records)
    for actual,saved in zip(records,author):
        assert actual.keys()==saved.keys()
        for k,v in actual.items():
            if isinstance(v,bool):assert saved[k]==str(v)
            elif isinstance(v,(int,float)):assert abs(v-float(saved[k]))<1e-11,(name,k,v,saved[k])
            else:assert v==saved[k]
    with (A/('independent_'+name)).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    assert h(RANK/name)==ranking['csv_sha256'][name]
fused=[r for r in identities if r['output']=='fused'];idchanges=dict(improved=sum(r['delta_mAP_pp']>0 for r in fused),declined=sum(r['delta_mAP_pp']<0 for r in fused),unchanged=sum(r['delta_mAP_pp']==0 for r in fused),queries=sum(r['query_count'] for r in fused))
compare(idchanges,aggregate['fused_identity_summary'])
assert h(A.parent/'role_set_q1_source_analysis_20260908.json')==aggregate['source_analysis_sha256']
assert h(RANK/'ranking_replay.json')==aggregate['ranking_replay_sha256']
paths=[EXEC/'source_aggregate.json',EXEC/'MSVR310_ROLE_SET_V1_Q1_2026-09-08.md',A.parent/'role_set_q1_source_analysis_20260908.json',RANK/'ranking_replay.json',RANK/'all3000_query_output_changes.csv',RANK/'all300_identity_output_changes.csv',A.parent/'report_role_set_q1_terminal_20260908.py']
hashes=j(A/'audited_input_hashes.json')
for p in paths:
    hashes[str(p)]=h(p);target=A/'author_artifact_snapshots'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
save(A/'audited_input_hashes.json',hashes)
save(A/'independent_local_claim_checks.json',dict(status='PASS_ALL_SOURCE_PHASE_STATISTICS_AND_ALL_QUERY_IDENTITY_CSV_FIELDS',generated_at=datetime.datetime.now().astimezone().isoformat(),author_conclusions_not_used_as_oracle=True,endpoint_phase_blocks_checked=24,loss_components_per_step=14,training_rows=1560,query_output_rows=3000,identity_output_rows=300,all_csv_fields_checked=True,rank_elements=rank_elements,source_aggregate=newaggregate,source_phase_details=recomputed,paired_query_changes=paired_changes,fused_identity_changes=idchanges,model_forwards=0,optimizer_updates=0))
print(json.dumps(dict(status='PASS_ALL_CLAIM_ARITHMETIC',phase_blocks=24,training_rows=1560,query_output_rows=3000,identity_output_rows=300,fused_query_changes=paired_changes['fused'],fused_identity_changes=idchanges)))
