"""Read image-native evidence with its own keys/values.

The caller keeps the existing 128-token semantic reader. This component adds
one detail exit; it does not replace, interpolate, or overwrite semantic values.
The caller owns the matched training recipe and deployment.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ImageNativeEvidenceReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
        )
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128)
        self.norm = nn.LayerNorm(128)
        self.output = nn.Linear(128, 128, bias=False)
        # Only this exit starts at zero. Other layers are normally initialized;
        # actual cumulative gradient support must be checked by a future M0.
        nn.init.zeros_(self.output.weight)

    def forward(self, images, queries):
        # images: B x 3 modalities x 3 channels x H x W
        # queries: B x 3 modalities x 16 regions x 128, from semantic context
        batch = images.shape[0]
        assert images.ndim == 5 and images.shape[1:3] == (3, 3)
        assert queries.shape == (batch, 3, 16, 128)
        detail = self.stem(images.flatten(0, 1))
        detail = detail.flatten(2).transpose(1, 2).reshape(batch, 3, -1, 128)
        assert detail.shape[2] == 512
        # Existing role attention uses FP32 after actual M0 Q/K gradient loss;
        # retain that numerical boundary for the new independent attention.
        with torch.autocast(images.device.type, enabled=False):
            detail = detail.float()
            q = self.query(F.layer_norm(queries.float(), (128,)))
            k = self.key(F.layer_norm(detail, (128,)))
            v = self.value(detail)
            weights = (q @ k.transpose(-1, -2) * 128 ** -0.5).softmax(dim=-1)
            return self.output(self.norm(weights @ v))
