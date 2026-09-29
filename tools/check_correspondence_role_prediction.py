#!/usr/bin/env python3
"""CPU contract checks; synthetic CLIP/mixer do not constitute a production M0."""

from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modeling'))
from trifusion.correspondence_roles import CorrespondenceTriFusion
from trifusion.correspondence_role_prediction import RolePredictionTriFusion
from trifusion.state import MODALITY_ORDER


class SyntheticVisual(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.patch_projection = nn.Linear(3, 768)
        self.global_projection = nn.Linear(768, 512)
        self.base = nn.Module()
        self.base.transformer = nn.Module()
        self.base.transformer.resblocks = nn.ModuleList(nn.Identity() for _ in range(12))

    def forward(self, images, cam_label, view_label):
        patches = F.adaptive_avg_pool2d(images, self.grid).flatten(2).transpose(1, 2)
        patches = self.patch_projection(patches)
        sequence = torch.cat((patches.mean(dim=1, keepdim=True), patches), dim=1).transpose(0, 1)
        for block in self.base.transformer.resblocks:
            sequence = block(sequence)
        return sequence.transpose(0, 1), self.global_projection(sequence[0])


def make_model(grid, address=None, prediction=None):
    torch.manual_seed(42)
    signal = nn.Module()
    signal.clip_vision_encoder = SyntheticVisual(grid)
    kwargs = dict(num_classes=3, grid=grid, width=128,
                  mamba_factory=lambda width: nn.Sequential(nn.Linear(width, width), nn.Tanh()))
    if address is None:
        return CorrespondenceTriFusion(signal, **kwargs)
    return RolePredictionTriFusion(signal, address_mode=address, prediction_mode=prediction, **kwargs)


def check_grid(grid):
    original = make_model(grid)
    original_rng = torch.get_rng_state().clone()
    original_state = original.state_dict()
    models = {}
    for address in ('own', 'matched'):
        for prediction in ('direct', 'predictor'):
            model = make_model(grid, address, prediction)
            assert torch.equal(torch.get_rng_state(), original_rng)
            assert all(torch.equal(model.state_dict()[key], value) for key, value in original_state.items())
            models[address, prediction] = model
    common = models['own', 'direct'].state_dict()
    assert all(all(torch.equal(model.state_dict()[key], value) for key, value in common.items())
               for model in models.values())
    torch.manual_seed(17)
    batch = {'images': {name: torch.randn(4, 3, 32, 32) for name in MODALITY_ORDER},
             'camera_ids': torch.zeros(4, dtype=torch.long)}
    original.train()
    torch.manual_seed(91)
    old_output = original(batch, return_aux=True)
    control = models['own', 'direct'].train()
    torch.manual_seed(91)
    control_output = control(batch, return_aux=True)
    for key in ('fused', 'shared_global', 'logits'):
        assert torch.equal(old_output[key], control_output[key]), key
    assert all(torch.equal(old_output['prediction'][key], control_output['prediction'][key])
               for key in ('local', 'relation', 'cross_modal'))
    original.eval()
    with torch.no_grad():
        deployment = original(batch)
        assert all(torch.equal(model.eval()(batch), deployment) for model in models.values())
    results = []
    for (address, prediction), model in models.items():
        model.train()
        # Force a real address discrepancy, rather than testing only zero-initialized offsets.
        with torch.no_grad():
            model.roles.shared_offset.bias.fill_(0.7)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(91)
        loss = sum(model._prediction_losses(batch['images'], batch['camera_ids']).values())
        assert torch.isfinite(loss)
        loss.backward()
        offset_grad = model.roles.shared_offset.weight.grad
        assert (offset_grad is not None and offset_grad.abs().sum() > 0) if address == 'own' else offset_grad is None
        assert all(p.grad is None for p in model.teacher.parameters())
        assert all(p.grad is None for p in model.teacher_adapters.parameters())
        assert all(p.grad is None for p in model.backbone.signal.parameters())
        for head in model.predictors:
            if prediction == 'predictor':
                assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                           for p in head.parameters())
            else:
                assert all(p.grad is None and not p.requires_grad for p in head.parameters())
        model.zero_grad(set_to_none=True)
        output = model(batch, return_aux=False)
        output[:, 0].sum().backward()
        assert model.roles.shared_offset.weight.grad.abs().sum() > 0
        with torch.no_grad():
            model.roles.role_adapters[0][-1].bias.add_(0.1)
            model.backbone.adapters[0][0][-1].bias.add_(0.2)
        old_teacher = model.teacher.role_adapters[0][-1].bias.clone()
        old_adapter = model.teacher_adapters[0][0][-1].bias.clone()
        model.update_teacher()
        assert torch.allclose(model.teacher.role_adapters[0][-1].bias,
                              old_teacher * 0.996 + model.roles.role_adapters[0][-1].bias * 0.004)
        assert torch.allclose(model.teacher_adapters[0][0][-1].bias,
                              old_adapter * 0.996 + model.backbone.adapters[0][0][-1].bias * 0.004)
        fresh = make_model(grid, address, prediction)
        fresh.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.equal(model.eval()(batch), fresh.eval()(batch))
        results.append({'address_mode': address, 'prediction_mode': prediction,
                        'prediction_loss': float(loss.detach()),
                        'prediction_offset_gradient': 'student_nonzero' if address == 'own' else 'absent',
                        'clean_retrieval_offset_gradient_nonzero': True,
                        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'predictor_parameters': sum(p.numel() for p in model.predictors.parameters())})
    # At identical addresses the explicitly addressed role path must equal the old path.
    model = models['matched', 'direct']
    with torch.no_grad():
        stages, _ = model.backbone(batch['images'], batch['camera_ids'])
        own = model.roles(stages)
        matched = model.roles(stages, positions=own.positions)
        assert all(torch.equal(getattr(own, key), getattr(matched, key))
                   for key in ('cnn', 'transformer', 'mamba', 'positions'))
    return {'grid': grid, 'conditions': results, 'original_control_forward_loss_rng_exact': True,
            'shared_initial_weights_exact': True, 'same_address_role_forward_exact': True,
            'teacher_stop_gradient_and_ema_pass': True, 'strict_module_reload_exact': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(2)
    report = {'status': 'CPU_M3_ADDRESS_AND_PREDICTOR_CONTRACT_PASS',
              'at': datetime.now().astimezone().isoformat(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_source_sha256': hashlib.sha256((ROOT / 'modeling/trifusion/correspondence_role_prediction.py').read_bytes()).hexdigest(),
              'grids': [check_grid(grid) for grid in ((16, 8), (8, 16))],
              'boundary': 'Synthetic visual/mixer CPU checks only; not production CLIP/Mamba M0, training or retrieval performance.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
