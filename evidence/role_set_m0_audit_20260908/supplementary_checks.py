"""Read-only label-origin and CPU synthetic mathematics checks; no model object."""
import ast
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F
torch.set_num_threads(2);torch.set_num_interop_threads(2)
ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
protocol=read(ROOT/'protocols/msvr310_train_oof_v1.json')
labels_path=ROOT/'evidence/vehicle_query_protocol_labels_20260905.json'
assert sha(labels_path)==protocol['label_evidence_sha256']
assert sha(ROOT/'tools/build_msvr310_train_oof_protocol.py')==protocol['builder_sha256']
dataset=next(x for x in read(labels_path)['datasets'] if x['dataset']=='MSVR310')
train=dataset['record_manifest']['bounding_box_train']
assert len(train)==1032
for row,source in zip(protocol['records'],train,strict=True):
    assert all(row[k]==source[k] for k in ('identity','camera','scene'))
    assert row['paths'][0]=='bounding_box_train/'+source['path']
train_ids={x['identity'] for x in train}
official_ids={x['identity'] for split in ('query3','bounding_box_test') for x in dataset['record_manifest'][split]}
assert train_ids.isdisjoint(official_ids)

ns={'torch':torch,'F':F,'OrderedDict':OrderedDict}
selections={
    'tools/msvr_role_set_relations.py':['relation_objectives','fused_distances'],
    'tools/msvr_instance_memory.py':['InstanceMemory','expanded_triplet','check_math'],
    'tools/probe_msvr_history_candidate_gradients.py':['differentiable_history_loss','math_check'],
}
function_receipts=[]
for path,names in selections.items():
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),'exec'),ns)
    function_receipts.append(dict(path=path,sha256=sha(ROOT/path),functions=names))
memory_result=ns['check_math']();chain_result=ns['math_check']()

# Independently prescribed cross-current/history tie derivatives, exactly 1/8.
dc=torch.full((4,4),.6,requires_grad=True);dh=torch.full((4,2),.6,requires_grad=True)
combined=torch.cat((dc,dh),1).detach()
hard,role,selected=ns['relation_objectives'](dc,dh,[0,0,1,1],[0,1],[combined]*3)
ga=torch.autograd.grad(hard,(dc,dh),retain_graph=True)
gb=torch.autograd.grad(role,(dc,dh))
expected_dc=torch.zeros_like(dc);expected_dh=torch.zeros_like(dh)
for i in range(4):
    identity=i//2
    expected_dc[i,i^1]=.125
    expected_dc[i,2 if identity==0 else 0]=-.125
    expected_dh[i,identity]=.125;expected_dh[i,1-identity]=-.125
assert torch.equal(ga[0],expected_dc) and torch.equal(ga[1],expected_dh)
assert all(torch.equal(x,y) for x,y in zip(ga,gb,strict=True))
assert selected['counts'].eq(1).all()

# Random synthetic distances: explicit legal-index set and mean-hinge reference.
torch.manual_seed(72491)
ids=[0,0,1,1,2,2,3,3];mids=[0,1,4,5]
random_checks=[]
for trial in range(12):
    current=torch.rand((8,8),dtype=torch.float64,requires_grad=True)
    historical=torch.rand((8,4),dtype=torch.float64,requires_grad=True)
    role_distances=[torch.rand((8,12),dtype=torch.float64) for _ in range(3)]
    hard,role,selection=ns['relation_objectives'](current,historical,ids,mids,role_distances)
    all_distances=torch.cat((current,historical),1)
    terms=[];reference_proposals=[]
    for i in range(8):
        positives=[j for j,x in enumerate(ids+mids) if x==ids[i] and j!=i]
        negatives=[j for j,x in enumerate(ids+mids) if x!=ids[i]]
        positive=torch.stack([all_distances[i,j] for j in positives]).max()
        proposed=[min(negatives,key=lambda j:float(d[i,j].detach())) for d in [all_distances,*role_distances]]
        reference_proposals.append(proposed)
        terms.append(torch.stack([F.relu(positive-all_distances[i,j]+.3) for j in sorted(set(proposed))]).mean())
    reference=torch.stack(terms).mean()
    gradients=torch.autograd.grad(role,(current,historical),retain_graph=True)
    wanted=torch.autograd.grad(reference,(current,historical))
    error=max((a-b).abs().max().item() for a,b in zip(gradients,wanted,strict=True))
    assert abs(float(role-reference))<1e-12 and error<1e-12
    assert selection['proposals'].tolist()==reference_proposals
    assert 1<=int(selection['counts'].min())<=int(selection['counts'].max())<=4
    random_checks.append(dict(trial=trial,scalar_error=abs(float(role-reference)),distance_gradient_max_error=error))

# Candidate VJP chain-rule identity in a synthetic linear tensor expression.
torch.manual_seed(42)
cur=torch.randn(8,16,requires_grad=True);raw=torch.randn(4,16);matrix=torch.randn(16,16,requires_grad=True)
hist=F.normalize(raw@matrix,dim=1);dc,dh=ns['fused_distances'](cur,hist)
roles=[torch.arange(12,dtype=torch.float32).repeat(8,1).roll(k,1) for k in (0,4,8)]
_,loss,_=ns['relation_objectives'](dc,dh,ids,mids,roles)
direct=torch.autograd.grad(loss,matrix,retain_graph=True)[0]
leaf=hist.detach().requires_grad_(True);lc,lh=ns['fused_distances'](cur.detach(),leaf)
_,partial,_=ns['relation_objectives'](lc,lh,ids,mids,roles)
upstream=torch.autograd.grad(partial,leaf)[0]
vjp=torch.autograd.grad(hist,matrix,grad_outputs=upstream)[0]
assert torch.equal(direct,vjp) and direct.abs().sum()>0
assert not torch.cuda.is_initialized() and torch.get_num_threads()==2

paths=['evidence/vehicle_query_protocol_labels_20260905.json','evidence/msvr310_dataset_install_20260905.json','tools/build_msvr310_train_oof_protocol.py','tools/audit_vehicle_query_protocol_labels.py','tools/check_msvr_role_set_relations.py']
texts={str(ROOT/p):(ROOT/p).read_text() for p in paths}
result=dict(status='PASS_LABEL_ORIGIN_AND_SYNTHETIC_CPU_OBJECTIVE',label_evidence_sha256=sha(labels_path),protocol_builder_sha256=protocol['builder_sha256'],all_1032_records_match_original_label_evidence=True,train_official_identity_sets_disjoint=True,official_image_reads=0,model_forwards=0,model_backwards=0,synthetic_tensor_autograd_only=True,isolated_function_receipts=function_receipts,existing_memory_math=memory_result,existing_history_chain_math=chain_result,independent_exact_tie_derivative={'current':ga[0].tolist(),'historical':ga[1].tolist(),'expected_derivative_magnitude':.125,'pass':True},independent_random_distance_checks=random_checks,synthetic_candidate_vjp_bitwise_equal=True,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),cuda_initialized=torch.cuda.is_initialized(),package_versions={name:importlib.metadata.version(name) for name in ('torch','numpy','torchvision','mamba-ssm','causal-conv1d')},texts=texts)
print(json.dumps(result,indent=2))
