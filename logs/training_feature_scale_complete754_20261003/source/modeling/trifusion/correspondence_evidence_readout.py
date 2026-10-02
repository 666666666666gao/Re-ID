"""Separate content selection from modality/region-preserving role readout."""

from copy import deepcopy

import torch
import torch.nn.functional as F

from .correspondence_roles import CorrespondenceRoles, CorrespondenceTriFusion


class ContentSelectionRoles(CorrespondenceRoles):
    def __init__(self, *, selection: str, **kwargs):
        super().__init__(**kwargs)
        assert selection in ("single", "uniform", "query")
        self.selection = selection
        height, grid_width = self.grid
        delta_y, delta_x = torch.meshgrid(
            torch.linspace(-1, 1, 3) / (height - 1),
            torch.linspace(-1, 1, 3) / (grid_width - 1),
            indexing="ij",
        )
        # Nine candidates within half a patch-center interval of each anchor.
        self.register_buffer("candidate_offsets", torch.stack((delta_x, delta_y), dim=-1)
                             .reshape(9, 2), persistent=False)

    def _sample(self, patches: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.selection == "single":
            return super()._sample(patches, positions)
        batch_size, modalities, tokens, width = patches.shape
        height, grid_width = self.grid
        assert modalities == 3 and tokens == height * grid_width
        grid = patches.reshape(batch_size * 3, height, grid_width, width).permute(0, 3, 1, 2)
        points = positions.unsqueeze(-2) + self.candidate_offsets
        samples = F.grid_sample(
            grid, points.reshape(batch_size * 3, self.anchor_count * 9, 1, 2),
            align_corners=True,
        ).squeeze(-1).transpose(1, 2).reshape(batch_size, 3, self.anchor_count, 9, width)
        if self.selection == "uniform":
            return samples.mean(dim=-2)
        queries = F.layer_norm(self.anchor_queries, (width,))[None, None, :, None]
        keys = F.layer_norm(samples, (width,))
        scores = (queries * keys).sum(dim=-1) * width ** -0.5
        return (scores.softmax(dim=-1).unsqueeze(-1) * samples).sum(dim=-2)


class EvidenceReadoutTriFusion(CorrespondenceTriFusion):
    def __init__(self, *args, selection: str, structured_readout: bool, **kwargs):
        super().__init__(*args, **kwargs)
        # Replacement construction must not change subsequent mask/augmentation RNG.
        with torch.random.fork_rng(devices=[]):
            roles = ContentSelectionRoles(
                selection=selection, grid=self.roles.grid, width=self.roles.width,
                anchor_side=self.roles.anchor_side, m1=self.roles.m1, m2=self.roles.m2,
                mamba_factory=kwargs["mamba_factory"],
            )
        roles.load_state_dict(self.roles.state_dict(), strict=True)
        self.roles = roles
        if self.m3:
            self.teacher = deepcopy(self.roles)
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        self.structured_readout = structured_readout
        assert self.roles.width == 128 and self.roles.anchor_side == 4

    def read_evidence(self, evidence):
        features = (evidence.cnn, evidence.transformer, evidence.mamba)
        if not self.structured_readout:
            pooled = torch.cat([feature.mean(dim=(1, 2)) for feature in features], dim=1)
            return self.readout(pooled)
        batch_size = features[0].shape[0]
        regions = [feature.reshape(batch_size, 3, 2, 2, 2, 2, 128)
                   .mean(dim=(3, 5)).reshape(batch_size, 3, 4, 128)
                   for feature in features]
        regional_inputs = torch.cat(regions, dim=-1)
        # The existing 1536x384 matrix supplies 3 modalities x 4 regions x 128 rows.
        weights = self.readout.weight.reshape(3, 4, 128, 384)
        return torch.einsum("bmrf,mrof->bmro", regional_inputs, weights).reshape(batch_size, 1536)

    def forward(self, batch: dict, *, return_aux: bool = False):
        stages, shared_global = self.backbone(batch["images"], batch["camera_ids"])
        evidence = self.roles(stages)
        correction = self.read_evidence(evidence)
        fused = F.normalize(shared_global.float() + self.readout_gain * correction.float(), dim=1)
        if not return_aux:
            return fused
        result = {"fused": fused, "shared_global": shared_global,
                  "logits": self.classifier(self.neck(fused))}
        if self.training and self.m3:
            result["prediction"] = self._prediction_losses(batch["images"], batch["camera_ids"])
        return result
