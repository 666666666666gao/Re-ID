"""Give shared features their own author task; route fused learning to roles."""
import torch.nn.functional as F
from torch.func import functional_call

from .evidence_author_heads import AuthorHeadEvidence


def detached_head(module, feature):
    """Use current head values without its parameter gradients or buffer updates."""
    state = {name: parameter.detach() for name, parameter in module.named_parameters()}
    state.update({name: buffer.detach().clone() for name, buffer in module.named_buffers()})
    return functional_call(module, state, (feature,), strict=True)


class GlobalTaskRoleHeads(AuthorHeadEvidence):
    def forward(self, batch, *, return_aux=False):
        output = self.evidence_model.forward_features(batch)
        if not return_aux:
            return output['fused']
        global_feature = output['shared_global'].float()
        raw = global_feature.detach() + self.evidence_model.readout_gain * output['correction'].float()
        global_features = (global_feature,) if self.signal.direct else global_feature.split(512, dim=1)
        fused_features = (raw,) if self.signal.direct else raw.split(512, dim=1)
        global_heads, fused_heads = [], []
        for (neck_name, classifier_name), global_part, fused_part in zip(
                self.head_names, global_features, fused_features):
            neck = getattr(self.signal, neck_name)
            classifier = getattr(self.signal, classifier_name)
            global_heads.append((classifier(neck(global_part)), global_part))
            fused_heads.append((detached_head(classifier, detached_head(neck, fused_part)), fused_part))
        return {**output, 'raw_fused': raw, 'fused': F.normalize(raw, dim=1),
                'heads': fused_heads, 'global_heads': global_heads}
