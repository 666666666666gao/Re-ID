from pathlib import Path
import ast,hashlib,json

p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
def put(name,text):
    target=p/'tools'/name;assert not target.exists();ast.parse(text);target.write_text(text,encoding='utf-8')
    return dict(path=str(target.relative_to(p)),sha256=hashlib.sha256(target.read_bytes()).hexdigest())

trainer=(p/'tools/train_msvr_role_set.py').read_text(encoding='utf-8')
trainer=trainer.replace('role_set','smooth_ap').replace('role-set','Smooth-AP').replace('ROLE_SET','SMOOTH_AP')
start=trainer.index('def context(path):');stop=trainer.index('\n\ndef encode_graph_all',start)
trainer=trainer[:start]+'''def context(path):
    from tools.train_msvr_role_set import context as previous_context
    spec=json.loads(path.read_bytes())
    assert spec['schema']=='msvr310-smooth-ap-paired-v1' and spec['seed']==42
    assert spec['temperature']==.01 and spec['fused_metric_weight']==1
    for name,digest in spec['project_file_sha256'].items():assert sha256(ROOT/name)==digest,name
    for name in ('core_audit','positive_audit'):
        audit=json.loads((ROOT/spec[name]).read_bytes())
        assert audit['verdict'] in ('PASS','WARN')
    previous,base=previous_context(ROOT/spec['previous_config'])
    assert spec['memory']==previous['memory'] and spec['training_epochs']==20
    return spec,base
'''+trainer[stop:]
trainer=trainer.replace('from tools.msvr_smooth_ap_relations import relation_objectives, fused_distances','from tools.msvr_smooth_ap import paired_objectives\n    from tools.msvr_role_set_relations import fused_distances')
trainer=trainer.replace('from tools.probe_msvr_smooth_ap_gradients import','from tools.probe_msvr_role_set_gradients import')
trainer=trainer.replace('relation_objectives(dc,dm,ids,mids,role_distances)','paired_objectives(dc,dm,ids,mids)').replace('relation_objectives(uc,uh,ids,mids,role_distances)','paired_objectives(uc,uh,ids,mids)').replace('relation_objectives(ec,eh,ids,mids,role_distances)','paired_objectives(ec,eh,ids,mids)')
trainer=trainer.replace('hard,smooth_ap,selection=','hard,smooth_ap,per_anchor_ap=')
start=trainer.index('                relation_saved=dict(');stop=trainer.index('\n                if step==0:',start)
trainer=trainer[:start]+'''                relation_saved=dict(hard_loss=float(hard.detach()),smooth_ap_loss=float(smooth_ap.detach()),
                    per_anchor_smoothed_ap=per_anchor_ap.detach().cpu().tolist(),temperature=.01,
                    positive_counts=[sum(y==identity for y in ids+mids)-1 for identity in ids])'''+trainer[stop:]
trainer=trainer.replace('selected_objective,selection,role_distances','selected_objective,per_anchor_ap,role_distances')
trainer=trainer.replace("relation_objective=relation_saved,proposal_order=['fused',*EXPERTS],coordinate_rule='fresh'","relation_objective=relation_saved,saved_space_order=['fused',*EXPERTS],coordinate_rule='fresh'")
trainer=trainer.replace("smooth_ap='fresh_history_both_sides_smooth_ap_mean'","smooth_ap='fresh_history_both_sides_smooth_ap_tau_0.01'")
trainer=trainer.replace('role_proposal_indices_detached=True,other_thirteen_loss_terms_unchanged=True','all_true_positive_positions=True,temperature=.01,other_thirteen_loss_terms_unchanged=True')
trainer=trainer.replace('Only the fused negative-relation objective changes','Only the fused ranking objective changes')
assert 'relation_objectives' not in trainer and 'selection' not in trainer and 'probe_msvr_smooth_ap' not in trainer
rows=[put('train_msvr_smooth_ap.py',trainer)]

verifier=(p/'tools/verify_msvr_role_set.py').read_text(encoding='utf-8').replace('role_set','smooth_ap').replace('role-set','Smooth-AP').replace('ROLE_SET','SMOOTH_AP')
insert=verifier.index('\ndef training(')
reference='''
def independent_smoothed_ap(distance, identities, history_identities):
    # Float64 independent NumPy rank sums; no import of training objective.
    ids=np.asarray(identities);allids=np.asarray(identities+history_identities)
    score=1-distance.astype(np.float64)**2/2
    aps=[]
    for i in range(len(ids)):
        positive=(allids==ids[i]);positive[i]=False
        pos=np.flatnonzero(positive);assert pos.size
        columns=np.arange(len(allids))
        delta=(score[i][None,:]-score[i,pos][:,None])/.01
        comparison=1/(1+np.exp(-delta))
        comparison[(columns[None,:]==i)|(columns[None,:]==pos[:,None])]=0
        rp=1+comparison[:,positive].sum(1)
        ra=1+comparison.sum(1)
        aps.append(float(np.mean(rp/ra)))
    return np.array(aps)

'''
verifier=verifier[:insert]+reference+verifier[insert:]
start=verifier.index('            negative_mask=np.concatenate');stop=verifier.index('            expected=dict(',start)
verifier=verifier[:start]+'''            smoothed_ap=independent_smoothed_ap(distance[0],audit['identities'],[r['identity'] for r in wanted])
            smooth_ap=float(1-smoothed_ap.mean())
            relation=audit['relation_objective']
            assert audit['saved_space_order']==['fused',*EXPERTS] and relation['temperature']==.01
            assert np.allclose(relation['per_anchor_smoothed_ap'],smoothed_ap,rtol=0,atol=2e-6)
            allids=audit['identities']+[r['identity'] for r in wanted]
            assert relation['positive_counts']==[sum(y==x for y in allids)-1 for x in audit['identities']]
            target=pooled if endpoint=='control' else smooth_ap
'''+verifier[stop:]
assert 'proposals' not in verifier and 'selection' not in verifier
rows.append(put('verify_msvr_smooth_ap.py',verifier))

check=(p/'tools/check_msvr_role_set.py').read_text(encoding='utf-8').replace('role_set','smooth_ap').replace('ROLE_SET','SMOOTH_AP')
check=check.replace('from tools.check_msvr_smooth_ap_relations import checks','from tools.check_msvr_smooth_ap_math import checks')
rows.append(put('check_msvr_smooth_ap.py',check))
runner=(p/'tools/run_msvr_role_set.py').read_text(encoding='utf-8').replace('role_set','smooth_ap').replace('ROLE_SET','SMOOTH_AP')
rows.append(put('run_msvr_smooth_ap.py',runner))
print(json.dumps(rows))
