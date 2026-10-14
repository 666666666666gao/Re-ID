"""Refresh the unchanged role histories using the F1 model feature widths."""
import torch
from tools.probe_msvr_role_set_gradients import encode_all, EXPERTS

def refresh_all(model,fields,metadata,step):
    for key in [k for k in fields.fields if step-k>8]:del fields.fields[key]
    before={n:v.clone() for n,v in model.named_buffers()}
    cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
    groups=sorted({r['stored_step'] for r in metadata})
    encoded={k:encode_all(model,fields.fields[k]) for k in groups}
    values={name:torch.stack([encoded[r['stored_step']][name][fields.fields[r['stored_step']]['positions'][r['record_index']]] for r in metadata])
            if metadata else torch.empty((0,model.fused_embedding_width if name=='fused' else model.branch_embedding_width),device='cuda')
            for name in ('fused',*EXPERTS)}
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
    assert all(torch.equal(v,before[n]) for n,v in model.named_buffers())
    return values,64*len(groups)
