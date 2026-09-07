#!/usr/bin/env python3
"""CUDA geometry, analytic zero derivative and full joint-block connectivity."""
import json
import numpy as np
import torch
from torch.nn import functional as F

from trifusion.joint_geometry_v29 import bounded_slots, bounded_joint_bank, TANGENT_BOUND, COSINE_FLOOR
from trifusion.joint_tokens_v28 import normalized_joint_bank
from trifusion.joint_tokens_v28_fp32 import FP32JointResidualTokens
from trifusion.state import EXPERT_ORDER


def run():
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        h = F.normalize(torch.randn(64, 3, 512, device='cuda'), dim=-1)
        c = torch.zeros_like(h, requires_grad=True)
        y, update = bounded_slots(h, c)
        assert torch.equal(y, F.normalize(h + torch.zeros_like(h), dim=-1))
        target = torch.randn_like(h)
        (y * target).sum().backward()
        h2 = h.square().sum(-1, keepdim=True)
        expected = (target - (target * h).sum(-1, keepdim=True) / h2 * h) / h2.sqrt()
        zero_gradient_error = float((c.grad - expected).abs().max())
        assert zero_gradient_error < 3e-6 and torch.count_nonzero(c.grad) > 0
        cases = []
        for scale in (0.0, 1.0, 100000.0):
            correction = torch.randn_like(h) * scale
            actual, update = bounded_slots(h, correction)
            a = h.cpu().numpy().astype(np.float64)
            b = correction.cpu().numpy().astype(np.float64)
            h2_np = (a*a).sum(-1, keepdims=True)
            v = b - (a*b).sum(-1, keepdims=True) / h2_np * a
            r = v / np.sqrt(1 + (v*v).sum(-1, keepdims=True) / (.25*h2_np))
            wanted = a+r;wanted /= np.linalg.norm(wanted, axis=-1, keepdims=True)
            error = float(np.max(np.abs(wanted-actual.cpu().numpy())))
            assert error < 2e-6
            ratio = float((update.norm(dim=-1) / h.norm(dim=-1)).max())
            cosine = float(F.cosine_similarity(h, actual, dim=-1).min())
            assert ratio <= TANGENT_BOUND+2e-6 and cosine >= COSINE_FLOOR-2e-6
            cases.append({'input_scale':scale,'numpy_output_max_error':error,
                          'maximum_relative_update':ratio,'minimum_cosine':cosine})
        # Exact-axis parallel/opposed corrections are removed, including large magnitudes.
        axis = torch.zeros(3,512,device='cuda');axis[:,0]=1
        axial = axis * torch.tensor([0.,100000.,-100000.],device='cuda')[:,None]
        axial_y, axial_r = bounded_slots(axis,axial)
        assert torch.equal(axial_y,axis) and torch.count_nonzero(axial_r)==0
        model = FP32JointResidualTokens().cuda()
        x = torch.randn(2,3,3,128,768,device='cuda',requires_grad=True)
        modal = {name:F.normalize(torch.randn(2,3,512,device='cuda'),dim=-1) for name in EXPERT_ORDER}
        calls=[]
        def capture(_module, inputs, _output): calls.append(list(inputs[0].shape))
        with model.mixer.register_forward_hook(capture): correction=model(x)
        assert calls==[[2,1152,128],[2,1152,128]]
        bank,stats=bounded_joint_bank(modal,correction)
        assert torch.equal(bank,normalized_joint_bank(modal,correction))
        bank_target=torch.randn_like(bank)
        (bank*bank_target).sum().backward()
        zero_stage={}
        for name,p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            zero_stage[name]=float(p.grad.abs().sum())
            assert (zero_stage[name]>0)==name.startswith('up.')
        model.zero_grad(set_to_none=True);x.grad=None
        for up in model.up.values(): torch.nn.init.normal_(up.weight,std=.001)
        correction=model(x)
        correction['transformer'].square().sum().backward()
        cross_role=float(x.grad[:,0].abs().sum());assert cross_role>0
        model.zero_grad(set_to_none=True);x.grad=None
        bank,stats=bounded_joint_bank(modal,model(x))
        (bank*bank_target).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())
        assert all(x.grad[:,i].abs().sum()>0 for i in range(3))
        return {'status':'PASS_V29_SYNTHETIC_CUDA_GEOMETRY_AND_MAMBA','parameters':sum(p.numel() for p in model.parameters()),
                'parameter_tensors':len(list(model.parameters())),'mamba_call_shapes':calls,
                'zero_correction_bank_exact':True,'zero_gradient_max_error':zero_gradient_error,
                'zero_init_gradient_abs_sums':zero_stage,'geometry_cases':cases,'axial_correction_identity_exact':True,
                'transformer_correction_gradient_into_cnn_tokens':cross_role,
                'all_joint_parameters_connected_after_nonzero_output':True,'all_three_input_roles_connected':True,
                'final_geometry':stats,'numpy_bank_max_error':max(r['numpy_output_max_error'] for r in cases),
                'optimizer_updates':0,'model_or_dataset_files_loaded':0}


if __name__=='__main__': print(json.dumps(run(),indent=2))
