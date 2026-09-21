"""Fresh CPU-only arithmetic, checkpoint-content and saved feature replay."""
from pathlib import Path
from collections import Counter
import hashlib,json,time,math,platform,struct
import numpy as np
import torch
REPO=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
def js(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def statehash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()
def close(a,b,tol):
    d=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert d<=tol,(d,tol);return d
torch.set_num_threads(56);start=time.monotonic()
summary=js(RUN/'q1/summary.json');config=js(REPO/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
baseline=js(Path(config['BASELINE']['SUMMARY']));protocol=js(REPO/'protocols/msvr310_train_oof_v1.json')
widths={'baseline_only':3072,'fused':7680,'cnn':1536,'transformer':1536,'mamba':1536}
result=dict(status='PENDING',python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,endpoints=[],totals=Counter(),sqrt_power_mismatches=[],weighted_norm_original_failures=[],model_forwards=0,optimizer_updates=0,official_image_reads=0,binary_downloads=0)
for fold,fr,b0 in zip(protocol['folds'],summary['folds'],baseline['folds']):
    bp=Path(b0['checkpoint']);assert sha(bp)==b0['checkpoint_sha256']
    base=torch.load(bp,map_location='cpu',weights_only=True)
    assert base['source_ids']==fold['source_ids'] and base['heldout_ids']==fold['heldout_ids']
    assert all(set(s['sampled_record_indices'])<=set(fold['source_record_indices']) for s in b0['training']['steps'])
    assert len(b0['training']['steps'])==650 and b0['training']['epochs']==50
    bstate=base['model_state_dict'];assert statehash(bstate)==b0['training']['final_state_sha256']
    barraypath=bp.parent/'retrieval_arrays.pt';assert sha(barraypath)==b0['retrieval']['retrieval_arrays_sha256']
    barrays=torch.load(barraypath,map_location='cpu',weights_only=True)
    for end in ('control','balanced'):
        r=fr['endpoints'][end];tr=r['training'];directory=RUN/'q1'/f"fold_{fold['fold']}_{end}"
        record=dict(fold=fold['fold'],endpoint=end,max_cross_ap_error=0,max_standard_ap_error=0,max_objective_error=0,training_distance_elements=0,rank_distance_elements=0,zeros=[],roles={},retrieval=[],actual_coefficient_checks=0)
        p=Path(r['checkpoint']);assert sha(p)==r['checkpoint_sha256'];payload=torch.load(p,map_location='cpu',weights_only=True)
        assert payload['binding']==r['initialization'] and payload['config_sha256']==summary['config_sha256']
        assert payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids'] and payload['fold']==fold['fold']
        assert all(k.startswith('baseline.') for k in payload['baseline_aliases'])
        state={k:bstate[v] for k,v in payload['baseline_aliases'].items()};state.update(payload['role_state_dict'])
        assert statehash(state)==tr['final_state_sha256']==r['strict_reload_state_sha256']
        trainable=r['initialization']['trainable_names'];assert len(trainable)==len(set(trainable))==203
        encoder=[n for n in trainable if n.startswith('encoder.')];assert len(encoder)==189
        groups={e:[n for n in encoder if n.startswith('encoder.'+e+'_')] for e in ('cnn','transformer','mamba')}
        assert sorted(sum(groups.values(),[]))==sorted(encoder) and all(groups.values())
        assert sum(state[n].numel() for n in trainable)==r['initialization']['trainable_parameters']
        frozen={n:v for n,v in state.items() if n.startswith('baseline.') or (n not in trainable and not n.endswith(('running_mean','running_var','num_batches_tracked')))}
        assert statehash(frozen)==tr['frozen_state_after_sha256']==tr['frozen_state_before_sha256']
        assert r['initialization']['signal_checkpoint_sha256']==b0['checkpoint_sha256'] and r['initialization']['signal_state_sha256']==statehash(bstate)
        record.update(checkpoint_sha256=r['checkpoint_sha256'],final_state_sha256=tr['final_state_sha256'],reconstructed_frozen_sha256=statehash(frozen),role_tensor_counts={k:len(v) for k,v in groups.items()},classifier_tensors=14,baseline_checkpoint_sha256=b0['checkpoint_sha256'])
        arrpath=directory/'retrieval_arrays.pt';assert sha(arrpath)==r['retrieval']['retrieval_arrays_sha256']
        arrays=torch.load(arrpath,map_location='cpu',weights_only=True);positions=[q['gallery_position'] for q in fold['query_rows']]
        assert arrays['query_gallery_positions']==positions and arrays['gallery_record_indices']==fold['gallery_record_indices']
        assert torch.equal(arrays['features']['baseline_only'],barrays['features']) and torch.equal(arrays['distances']['baseline_only'],barrays['distances'])
        assert torch.equal(arrays['features']['fused'][:,:3072],arrays['features']['baseline_only'])
        ranks=js(directory/'rankings.json')
        for output,width in widths.items():
            features=arrays['features'][output];saved=arrays['distances'][output]
            assert features.shape==(len(fold['gallery_record_indices']),width) and torch.isfinite(features).all()
            unit=features.float()/torch.linalg.vector_norm(features.float(),dim=1,keepdim=True).clamp_min(1e-12)
            query=unit[positions]
            computed=query.square().sum(1,keepdim=True)+unit.square().sum(1)[None]
            computed.addmm_(query,unit.T,beta=1,alpha=-2)
            assert torch.equal(computed,saved),(fold['fold'],end,output,'distance_bitwise')
            assert np.argsort(saved.numpy(),axis=1).tolist()==ranks[output]
            dunit=features.double()/torch.linalg.vector_norm(features.double(),dim=1,keepdim=True).clamp_min(1e-12)
            dq=dunit[positions];reference=dq.square().sum(1,keepdim=True)+dunit.square().sum(1)[None]-2*dq@dunit.T
            error=close(saved.double().numpy(),reference.numpy(),2e-6)
            record['retrieval'].append(dict(output=output,shape=list(saved.shape),distance_and_rank_exact=True,float64_distance_max_error=error))
            record['rank_distance_elements']+=saved.numel()
        logs=[json.loads(s) for s in (directory/'memory_steps.jsonl').read_text().splitlines()]
        dp=directory/'memory_distances.f32';assert sha(dp)==tr['audit_files']['memory_distances.f32']['sha256']
        with dp.open('rb') as f:
            for row,t in zip(logs,tr['steps'],strict=True):
                assert f.tell()==row['distance_offset_bytes'];size=row['distance_float_count'];raw=np.fromfile(f,dtype='<f4',count=size)
                assert len(raw)==size and np.isfinite(raw).all() and np.all(raw>=0)
                n=len(row['memory']);dist=raw.reshape(4,64,64+n);record['training_distance_elements']+=size
                ids=np.array(row['identities']);scenes=np.array(row['scenes']);ai=np.array(list(ids)+[m['identity'] for m in row['memory']]);asc=np.array(list(scenes)+[m['scene'] for m in row['memory']])
                eq=ids[:,None]==ai[None,:];different_scene=scenes[:,None]!=asc[None,:]
                allowed=(~eq)|different_scene;positive=eq&different_scene
                assert not positive[:,:64].diagonal().any() and np.all(allowed[~eq])
                score=-dist[0].astype(np.float64)**2/2
                standard=[];cross=[]
                for i in range(64):
                    for is_cross,dest in ((False,standard),(True,cross)):
                        pos=positive[i].copy() if is_cross else eq[i].copy();pos[i]=False
                        legal=allowed[i].copy() if is_cross else np.ones(64+n,dtype=bool);legal[i]=False
                        pp=np.flatnonzero(pos)
                        if not len(pp):dest.append(0.);continue
                        differences=(score[i,None,:]-score[i,pp,None])/.01
                        comparison=1/(1+np.exp(-differences));comparison[:,~legal]=0;comparison[np.arange(len(pp)),pp]=0
                        value=((1+comparison[:,pos].sum(1))/(1+comparison.sum(1))).mean();dest.append(float(value))
                rel=row['relation_objective'];poscounts=positive.sum(1);eligible=poscounts>0
                assert rel['cross_scene_positive_counts']==poscounts.tolist()
                ce=close(cross,rel['cross_scene_per_anchor_ap'],2e-6);se=close(standard,rel['per_anchor_smoothed_ap'],2e-6)
                record['max_cross_ap_error']=max(record['max_cross_ap_error'],ce);record['max_standard_ap_error']=max(record['max_standard_ap_error'],se)
                crossloss=1-np.array(cross)[eligible].mean() if eligible.any() else 0.;standardloss=1-np.mean(standard)
                pc=eq[:,:64].copy();np.fill_diagonal(pc,False);nc=~eq[:,:64]
                hp=np.where(pc,dist[0,:,:64],-np.inf).max(1);hn=np.where(nc,dist[0,:,:64],np.inf).min(1)
                allpos=eq.copy();allpos[np.arange(64),np.arange(64)]=False
                allhp=np.where(allpos,dist[0],-np.inf).max(1);allhn=np.where(~eq,dist[0],np.inf).min(1)
                hard=float(np.maximum(0,hp-hn+np.float32(.3)).mean());pooled=float(np.maximum(0,allhp-allhn+np.float32(.3)).mean())
                for actual,expected in ((row['original_triplet'],hard),(rel['hard_loss'],pooled),(rel['smooth_ap_loss'],standardloss),(rel['cross_scene_loss'],crossloss),(t['components']['triplet_fused'],crossloss if row['replacement_active'] else hard)):
                    record['max_objective_error']=max(record['max_objective_error'],close(actual,expected,2e-6))
                if row['replacement_active'] and not eligible.any():record['zeros'].append(dict(step=row['step'],crossloss=crossloss,auxiliary_total=t['loss'],current_records=len(row['record_indices']),history_records=n))
                for role,b in row['gradient_balance'].items():
                    if b['supported']:
                        s=b['after'];q=(s['auxiliary']+1e-12)/(s['rank']+1e-12);root=min(4.,max(.25,math.sqrt(q)));power=min(4.,max(.25,q**.5))
                        assert root==b['ratio']
                        if root!=power:result['sqrt_power_mismatches'].append(dict(fold=fold['fold'],endpoint=end,step=row['step'],role=role,sqrt=root,power=power))
                    for k in ('applied_rank_weight','applied_auxiliary_weight'):
                        w=b[k];actual=float((torch.ones(1,dtype=torch.float32)*w).item());expected=struct.unpack('<f',struct.pack('<f',w))[0]
                        assert actual==expected;record['actual_coefficient_checks']+=1
                    if end=='balanced' and b['supported']:
                        p=b['rank_vs_auxiliary'];rnorm,anorm=p['first_norm'],p['second_norm'];dot=0 if p['cosine'] is None else rnorm*anorm*p['cosine'];wr,wa=b['applied_rank_weight'],b['applied_auxiliary_weight'];actual=b['rank_vs_applied']['second_norm']**2
                        old=wr**2*rnorm**2+wa**2*anorm**2+2*wr*wa*dot;threshold=1e-7*max(1,abs(old),abs(actual))
                        if abs(old-actual)>threshold:result['weighted_norm_original_failures'].append(dict(fold=fold['fold'],endpoint=end,step=row['step'],role=role,original_error=abs(old-actual),threshold=threshold))
            assert f.read()==b''
        result['endpoints'].append(record)
        for k in ('training_distance_elements','rank_distance_elements','actual_coefficient_checks'):result['totals'][k]+=record[k]
        del state,payload,arrays,frozen
    del base,bstate,barrays
assert result['totals']['training_distance_elements']==116501504 and result['totals']['rank_distance_elements']==2069520
assert not torch.cuda.is_initialized()
result.update(status='PASS_COMPLETE_INDEPENDENT_REMOTE_ARRAYS',elapsed_seconds=time.monotonic()-start,cuda_initialized=False)
print(json.dumps(result))
