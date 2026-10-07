"""Change only the gradient path from role inputs into the shared encoder."""
from .independent_native_roles import RawFeatureSemanticTriFusion, IndependentNativeTriFusion


class DetachedSemanticTriFusion(RawFeatureSemanticTriFusion):
    def role_evidence(self, batch, stages, context, shared_global):
        return super().role_evidence(batch, stages.detach(), context.detach(), shared_global.detach())


class DetachedNativeTriFusion(IndependentNativeTriFusion):
    def role_evidence(self, batch, stages, context, shared_global):
        return super().role_evidence(batch, stages.detach(), context.detach(), shared_global.detach())
