"""Apply the pinned author's heads to matched raw semantic/native features."""
from torch import nn
import torch.nn.functional as F


class SharedGlobalRawFeatures(nn.Module):
    """Independent global-only control without registered role parameters."""
    def __init__(self, initialized):
        super().__init__()
        self.backbone = initialized.backbone
        self.neck = initialized.neck
        self.classifier = initialized.classifier

    def forward_features(self, batch):
        _stages, raw = self.backbone(batch['images'], batch['camera_ids'])
        raw = raw.float()
        return {'raw_fused': raw, 'fused': F.normalize(raw, dim=1),
                'shared_global': raw}


class AuthorHeadEvidence(nn.Module):
    """Apply existing author heads to corrected raw features, once per input."""
    def __init__(self, evidence_model):
        super().__init__()
        self.evidence_model = evidence_model
        assert self.signal.feat_dim == 512
        assert not hasattr(self.signal, 'SIM') and not hasattr(self.signal, 'AlignM')
        # These historical normalized heads are not used by this training path.
        self.evidence_model.neck.requires_grad_(False)
        self.evidence_model.classifier.requires_grad_(False)
        self.head_names = (('bottleneck', 'classifier'),) if self.signal.direct else tuple(
            (f'bottleneck_{suffix}', f'classifier_{suffix}') for suffix in ('r', 'n', 't'))
        for neck_name, classifier_name in self.head_names:
            getattr(self.signal, neck_name).weight.requires_grad_(True)
            getattr(self.signal, classifier_name).weight.requires_grad_(True)
        self.train(self.training)

    @property
    def signal(self):
        return self.evidence_model.backbone.signal

    def train(self, mode=True):
        super().train(mode)
        # The existing backbone forces Signal.eval(); restore the author mode
        # explicitly, including the native BN heads and visual encoder.
        self.signal.train(mode)
        return self

    def forward(self, batch, *, return_aux=False):
        output = self.evidence_model.forward_features(batch)
        if not return_aux:
            return output['fused']
        raw = output['raw_fused']
        assert raw.shape[1] == 1536
        features = (raw,) if self.signal.direct else raw.split(512, dim=1)
        heads = [
            (getattr(self.signal, classifier_name)(getattr(self.signal, neck_name)(feature)), feature)
            for (neck_name, classifier_name), feature in zip(self.head_names, features)
        ]
        return {**output, 'heads': heads}


def build_author_optimizer_and_loss(cfg, model):
    """Keep author groups, but pass the whole evidence model to its optimizer."""
    from layers.make_loss import make_loss
    from solver.make_optimizer import make_optimizer

    loss_fn, center = make_loss(cfg, num_classes=model.signal.num_classes)
    optimizer, _unused_center_optimizer = make_optimizer(cfg, model, center)
    intended = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    assert len(actual) == len(set(actual)) and set(actual) == intended
    return optimizer, loss_fn
