import sys,json,torch
sys.path.insert(0,'/data/gaob/Re-ID/Trifusion/modeling')
from trifusion.selective_role_transport import partial_log_assignment
torch.set_num_threads(1);torch.manual_seed(42)
strong=torch.full((2,16,16),-8.0);strong.diagonal(dim1=-2,dim2=-1).fill_(8.0)
weak=torch.full_like(strong,-8.0);mixed=torch.randn_like(strong)
target=torch.tensor([1.0]*16+[16.0]);rows=[]
for label,score in [('diagonal',strong),('unmatched',weak),('mixed',mixed)]:
 p=partial_log_assignment(score).exp()
 rows.append(dict(condition=label,row_residual=(p.sum(-1)-target).abs().amax(0).tolist(),
  col_residual=(p.sum(-2)-target).abs().amax(0).tolist(),row_real_mass=p[:,:16,:16].sum(-1).tolist()))
print(json.dumps(dict(status='DIAGNOSIS_ONLY_ORIGINAL_COMPONENT_FAIL_RETAINED',rows=rows)))