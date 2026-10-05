"""Row-stochastic unmatched option, without OT column constraints."""
import torch
import torch.nn.functional as F

from .selective_role_transport import SelectiveTransportTriFusion, SlotMessageTransport, message_weights


def row_log_assignment(scores):
    assert scores.shape[-2:] == (16, 16)
    null = scores.new_full((*scores.shape[:-1], 1), 1.0)
    return torch.cat((scores, null), dim=-1).log_softmax(dim=-1)


class RowMassTransport(SlotMessageTransport):
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
                    # Each receiver has its own null option; no transpose-OT claim.
                    for receiver, sender, oriented in ((m,n,scores),(n,m,scores.transpose(-1,-2))):
                        real = row_log_assignment(oriented)[..., :16]
                        weights, mass, conditional = message_weights(real, self.mass_mode)
                        messages[receiver] = messages[receiver] + 0.5 * (weights @ private[:, sender])
                        masses.append(mass.detach())
                        entropies.append(-(conditional * real.log_softmax(-1)).sum(-1).detach())
                        concentrations.append((conditional.sum(-2)/16).amax(-1).detach())
            peer = torch.stack(messages,dim=1)
            correction = self.message_output(peer)
            all_mass = torch.stack(masses,dim=1)
            self.last_diagnostics = dict(
                transport_real_mass_mean=all_mass.mean(),transport_real_mass_min=all_mass.amin(),
                transport_real_mass_max=all_mass.amax(),
                transport_conditional_entropy=torch.stack(entropies,dim=1).mean(),
                transport_conditional_max_column_share=torch.stack(concentrations,dim=1).mean(),
                transport_peer_norm=peer.detach().norm(dim=-1).mean(),
                transport_self_norm=private.detach().norm(dim=-1).mean(),
                transport_added_norm=correction.detach().norm(dim=-1).mean())
            return private + correction


class RowMassTriFusion(SelectiveTransportTriFusion):
    def __init__(self,*args,mass_mode,**kwargs):
        super().__init__(*args,mass_mode=mass_mode,**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.roles.transport = RowMassTransport(mass_mode)


class RowSlotMassTriFusion(RowMassTriFusion):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,mass_mode='slot_mass',**kwargs)


class RowUniformMassTriFusion(RowMassTriFusion):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,mass_mode='uniform_mass',**kwargs)
