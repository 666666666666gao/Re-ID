"""Public TriFusion-ReID collaboration seams and result types."""

from .encoder import TriBranchEncoder
from .experts.semantic_residual import (
    SemanticCNNExpert,
    SemanticMambaExpert,
    SemanticTransformerExpert,
)
from .fusion import CollaborativeFusion, FusionResult
from .model import TriFusionOutput, TriFusionReID
from .peer_teaching import PeerTeachingResult, RoleDirectedPeerTeaching
from .reliability import ReliabilityPosterior
from .relay import HeterogeneousRelay, RelayResult
from .state import ExpertState, ExpertStateMap, ReliabilityResult

__all__ = [
    "ExpertState",
    "ExpertStateMap",
    "ReliabilityResult",
    "RelayResult",
    "FusionResult",
    "TriFusionOutput",
    "PeerTeachingResult",
    "TriBranchEncoder",
    "HeterogeneousRelay",
    "ReliabilityPosterior",
    "CollaborativeFusion",
    "SemanticCNNExpert",
    "SemanticMambaExpert",
    "SemanticTransformerExpert",
    "RoleDirectedPeerTeaching",
    "TriFusionReID",
]
