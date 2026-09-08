"""Distance-coordinate derivative audit only: no feature/model/parameter gradients."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,sys,json,hashlib,collections,time
import numpy as np,torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
R=pathlib.Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID');Q=pathlib.Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/q1')
sys.path.insert(0,str(R))
from tools.msvr_role_set_relations import relation_objectives
assert hashlib.sha256((R/'tools/msvr_role_set_relations.py').read_bytes()).hexdigest()=='e311920a08e6bfce426e8a67acf6a9c1b413116d4e906e9ad8ca31dca03800c1'
start=time.monotonic();results=[]
for fi in range(3):
    for endpoint in ('control','role_set'):
        folder=Q/f'fold_{fi}_{endpoint}';rows=[json.loads(s) for s in (folder/'memory_steps.jsonl').read_text().splitlines()]
        counters=collections.Counter();max_error=0.;exclusive={e:collections.Counter() for e in ('cnn','transformer','mamba')}
        with (folder/'memory_distances.f32').open('rb') as stream:
            for row in rows:
                B=64;H=len(row['memory']);D=np.fromfile(stream,dtype='<f4',count=4*B*(B+H)).reshape(4,B,B+H)
                ids=np.array(row['identities']);mi=np.array([r['identity'] for r in row['memory']],dtype=int);labels=np.r_[ids,mi]
                pos=ids[:,None]==labels;pos[np.arange(B),np.arange(B)]=False;neg=ids[:,None]!=labels
                expected_hard=np.zeros_like(D[0],dtype=np.float64);expected_set=np.zeros_like(D[0],dtype=np.float64)
                proposals=np.argmin(np.where(neg[None],D,np.inf),axis=2).T
                for anchor in range(B):
                    pcur=int(np.argmax(np.where(pos[anchor,:B],D[0,anchor,:B],-np.inf)));ncur=int(np.argmin(np.where(neg[anchor,:B],D[0,anchor,:B],np.inf)))
                    p_hist= B+int(np.argmax(np.where(pos[anchor,B:],D[0,anchor,B:],-np.inf))) if H else None
                    n_hist= B+int(np.argmin(np.where(neg[anchor,B:],D[0,anchor,B:],np.inf))) if H else None
                    p_val=D[0,anchor,pcur];n_val=D[0,anchor,ncur];pweights={pcur:1.};nweights={ncur:1.}
                    if H:
                        ph=D[0,anchor,p_hist] if pos[anchor,p_hist] else -np.inf;nh=D[0,anchor,n_hist] if neg[anchor,n_hist] else np.inf
                        if ph>p_val:pweights={p_hist:1.};p_val=ph
                        elif ph==p_val:pweights={pcur:.5,p_hist:.5};counters['current_history_positive_extreme_ties']+=1
                        if nh<n_val:nweights={n_hist:1.};n_val=nh
                        elif nh==n_val:nweights={ncur:.5,n_hist:.5};counters['current_history_negative_extreme_ties']+=1
                    chosen=set(map(int,proposals[anchor]));additional=chosen-{int(proposals[anchor,0])};count=len(chosen)
                    active=float(np.float32(p_val-n_val+np.float32(.3))>0)
                    if active:
                        for position,w in pweights.items():expected_hard[anchor,position]+=w/B
                        for position,w in nweights.items():expected_hard[anchor,position]-=w/B
                    expected_set[anchor]=expected_hard[anchor]/count
                    for position in additional:
                        if np.float32(p_val-D[0,anchor,position]+np.float32(.3))>0:
                            expected_set[anchor,position]-=1/(B*count)
                            for p,w in pweights.items():expected_set[anchor,p]+=w/(B*count)
                    if row['replacement_active']:
                        for ei,expert in enumerate(('cnn','transformer','mamba'),1):
                            position=int(proposals[anchor,ei]);other=[int(proposals[anchor,k]) for k in range(4) if k!=ei]
                            if position not in other:
                                exclusive[expert]['exclusive_position_proposals']+=1
                                if np.float32(p_val-D[0,anchor,position]+np.float32(.3))>0:
                                    exclusive[expert]['exclusive_active_hinge_positions']+=1
                                    exclusive[expert]['exclusive_active_history_positions' if position>=64 else 'exclusive_active_current_positions']+=1
                                    exclusive[expert]['exclusive_active_identity_other_than_fused']+=int(labels[position]!=labels[int(proposals[anchor,0])])
                    counters['anchors']+=1;counters['extra_selected_positions']+=len(additional)
                dc=torch.from_numpy(D[0,:,:B].copy()).requires_grad_(True);dh=torch.from_numpy(D[0,:,B:].copy()).requires_grad_(True)
                roles=[torch.from_numpy(v) for v in D[1:]]
                old,new,_=relation_objectives(dc,dh,ids.tolist(),mi.tolist(),roles)
                for loss,reference in [(old,expected_hard),(new,expected_set)]:
                    if H:
                        pair=torch.autograd.grad(loss,(dc,dh),retain_graph=True)
                        actual=torch.cat(pair,dim=1).numpy()
                    else:
                        # No history columns exist during the observed warmup.
                        actual=torch.autograd.grad(loss,dc,retain_graph=True)[0].numpy()
                    err=float(np.abs(actual-reference).max());max_error=max(max_error,err);assert err<2e-8,(fi,endpoint,row['step'],err)
                counters['steps']+=1;counters['distance_derivative_elements']+=2*B*(B+H)
        results.append(dict(fold=fi,endpoint=endpoint,max_distance_gradient_abs_error=max_error,counts=dict(counters),post_warmup_exclusive_proposal_hinge_activity={k:dict(v) for k,v in exclusive.items()}))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_ALL_Q1_HARD_AND_ROLE_SET_DISTANCE_DERIVATIVES',endpoint_checks=results,training_rows=1560,objectives_per_row=2,model_forwards=0,parameter_gradient_replay=False,feature_gradient_replay=False,scope='Analytic first-index row max/min, 0.5 current/history extrema tie split, detached proposals, position deduplication, per-anchor mean and ReLU convention versus actual loss code on all saved matrices. This is derivative w.r.t. distance entries only.',elapsed_seconds=time.monotonic()-start)))
