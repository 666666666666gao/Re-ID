"""Compare slot-specific and pair-uniform cross-spectrum message mass."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .correspondence_roles import RoleEvidence
from .role_global_tokens import GlobalTokenRoles
from .role_input_detach import DetachedSemanticTriFusion


def partial_log_assignment(scores):
    """SuperGlue-style dustbin marginals; 100 complete log Sinkhorn updates."""
    count = scores.shape[-1]
    assert scores.shape[-2:] == (16, 16)
    null = scores.new_full((*scores.shape[:-2], 1, 1), 1.0)
    matrix = torch.cat((torch.cat((scores, null.expand(*scores.shape[:-2], count, 1)), -1),
                        null.expand(*scores.shape[:-2], 1, count + 1)), -2)
    total_log = math.log(2 * count)
    marginal = scores.new_full((count + 1,), -total_log)
    marginal[-1] += math.log(count)
    u = torch.zeros_like(matrix[..., :, 0])
    v = torch.zeros_like(matrix[..., 0, :])
    for _ in range(100):
        u = marginal - torch.logsumexp(matrix + v.unsqueeze(-2), dim=-1)
        v = marginal - torch.logsumexp(matrix + u.unsqueeze(-1), dim=-2)
    return matrix + u.unsqueeze(-1) + v.unsqueeze(-2) + total_log


def message_weights(log_real, mass_mode):
    assert mass_mode in ('slot_mass', 'uniform_mass')
    mass = torch.logsumexp(log_real, dim=-1).exp()
    conditional = log_real.softmax(dim=-1)
    weights = (log_real.exp() if mass_mode == 'slot_mass'
               else mass.mean(dim=-1, keepdim=True).unsqueeze(-1) * conditional)
    return weights, mass, conditional


class SlotMessageTransport(nn.Module):
    def __init__(self, mass_mode):
        super().__init__()
        assert mass_mode in ('slot_mass', 'uniform_mass')
        self.mass_mode = mass_mode
        self.matching_projection = nn.Linear(128, 128, bias=False)
        self.message_output = nn.Linear(128, 128, bias=False)
        nn.init.zeros_(self.message_output.weight)
        self.last_diagnostics = {}

    def forward(self, semantic, private):
        assert semantic.shape == private.shape and semantic.shape[1:] == (3, 16, 128)
        with torch.autocast(semantic.device.type, enabled=False):
            semantic, private = semantic.float(), private.float()
            projected = self.matching_projection(F.layer_norm(semantic, (128,)))
            messages = [torch.zeros_like(private[:, m]) for m in range(3)]
            masses, entropies, concentrations = [], [], []
            for m in range(3):
                for n in range(m + 1, 3):
                    scores = projected[:, m] @ projected[:, n].transpose(-1, -2) * 128 ** -0.5
                    real = partial_log_assignment(scores)[..., :16, :16]
                    for receiver, sender, log_real in ((m, n, real), (n, m, real.transpose(-1, -2))):
                        weights, mass, conditional = message_weights(log_real, self.mass_mode)
                        messages[receiver] = messages[receiver] + 0.5 * (weights @ private[:, sender])
                        masses.append(mass.detach())
                        entropies.append(-(conditional * log_real.log_softmax(-1)).sum(-1).detach())
                        concentrations.append((conditional.sum(-2) / 16).amax(-1).detach())
            peer = torch.stack(messages, dim=1)
            correction = self.message_output(peer)
            all_mass = torch.stack(masses, dim=1)
            self.last_diagnostics = dict(
                transport_real_mass_mean=all_mass.mean(),
                transport_real_mass_min=all_mass.amin(),
                transport_real_mass_max=all_mass.amax(),
                transport_conditional_entropy=torch.stack(entropies, dim=1).mean(),
                transport_conditional_max_column_share=torch.stack(concentrations, dim=1).mean(),
                transport_peer_norm=peer.detach().norm(dim=-1).mean(),
                transport_self_norm=private.detach().norm(dim=-1).mean(),
                transport_added_norm=correction.detach().norm(dim=-1).mean())
            return private + correction


class SelectiveTransportRoles(GlobalTokenRoles):
    def __init__(self, *, mass_mode, **kwargs):
        super().__init__(**kwargs)
        self.transport = SlotMessageTransport(mass_mode)

    def forward(self, stages, context, shared_global):
        context_queries = self.context_queries(context).reshape(context.shape[0], 3, self.width)
        cnn_input = self._depth_role(stages, 0)
        transformer_input = self._depth_role(stages, 1)
        mamba_input = self._depth_role(stages, 2)
        positions = self._positions(transformer_input[:, :, 0])
        height, grid_width = self.grid
        cnn_grid = cnn_input[:, :, 1:].reshape(-1, height, grid_width, self.width).permute(0, 3, 1, 2)
        cnn_grid = cnn_grid + self.cnn(cnn_grid)
        cnn_patches = cnn_grid.permute(0, 2, 3, 1).reshape(cnn_input.shape[0], 3, -1, self.width)
        cnn = self.sample_context(cnn_patches, positions, context_queries, 0)
        transformer = self.sample_context(transformer_input[:, :, 1:], positions, context_queries, 1)
        mamba = self.sample_context(mamba_input[:, :, 1:], positions, context_queries, 2)
        queries = self.anchor_queries[None, None]
        cnn = self.output_norms[0](cnn + queries)
        transformer = transformer + queries + self.cnn_to_transformer(cnn)
        fixed = self.static_token_context[None, None].expand(context.shape[0], 3, -1)
        global_token = self.global_projection(fixed)
        sequence = torch.cat((global_token.unsqueeze(2), transformer), dim=2)
        transformer = self.transformer(sequence.flatten(0, 1))[:, 1:].reshape_as(transformer)
        transformer = self.output_norms[1](transformer)
        mamba_input = mamba + queries + self.transformer_to_mamba(transformer)
        sequence = self.mamba_norm(mamba_input.flatten(0, 1))
        propagated = 0.5 * (self.mamba(sequence) + self.mamba(sequence.flip(1)).flip(1))
        private = propagated.reshape_as(mamba_input)
        mamba = self.output_norms[2](self.transport(transformer, private))
        return RoleEvidence(cnn=cnn, transformer=transformer, mamba=mamba, positions=positions)


class SelectiveTransportTriFusion(DetachedSemanticTriFusion):
    def __init__(self, *args, mass_mode, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = SelectiveTransportRoles(mass_mode=mass_mode,
                token_mode='static', grid=self.roles.grid, width=self.roles.width,
                anchor_side=self.roles.anchor_side, m1=True, m2=True,
                mamba_factory=kwargs['mamba_factory'])
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles

    def forward_features(self, batch):
        output = super().forward_features(batch)
        output['transport_diagnostics'] = self.roles.transport.last_diagnostics
        return output


class SlotMassTriFusion(SelectiveTransportTriFusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mass_mode='slot_mass', **kwargs)


class UniformMassTriFusion(SelectiveTransportTriFusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mass_mode='uniform_mass', **kwargs)
