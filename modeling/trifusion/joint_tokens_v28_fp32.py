"""FP32 execution for the V28 joint block's measured small-derivative loss."""
import torch

from .joint_tokens_v28 import JointResidualTokens


class FP32JointResidualTokens(JointResidualTokens):
    def forward(self, deltas):
        with torch.autocast("cuda", enabled=False):
            return super().forward(deltas.float())
