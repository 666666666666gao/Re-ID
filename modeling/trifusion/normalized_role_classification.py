"""RGBNT201 diagnostic: normalized role classification, unchanged raw metric."""
import torch.nn.functional as F

from .global_task_role_heads import GlobalTaskRoleHeads, detached_head


class NormalizedRoleClassification(GlobalTaskRoleHeads):
    def forward(self, batch, *, return_aux=False):
        output = self.evidence_model.forward_features(batch)
        if not return_aux:
            return output['fused']
        assert self.signal.direct
        global_feature = output['shared_global'].float()
        raw = global_feature.detach() + self.evidence_model.readout_gain * output['correction'].float()
        normalized = F.normalize(raw, dim=1)
        neck_name, classifier_name = self.head_names[0]
        neck = getattr(self.signal, neck_name)
        classifier = getattr(self.signal, classifier_name)
        global_score = classifier(neck(global_feature))
        role_score = detached_head(classifier, detached_head(neck, normalized))
        return {**output, 'raw_fused': raw, 'fused': normalized,
                'heads': [(role_score, raw)],
                'global_heads': [(global_score, global_feature)]}
