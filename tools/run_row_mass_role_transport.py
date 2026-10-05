"""Direct row-null assignment; original RAW global/role training duties."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_selective_role_transport as previous
from trifusion.row_mass_role_transport import RowSlotMassTriFusion,RowUniformMassTriFusion

SCHEMA='trifusion-row-mass-role-transport-v2'
inner=previous.inner
original_build_core=previous.build_core
original_condition=previous.condition


def build_core(args,protocol):
    model,cfg,binding=original_build_core(args,protocol)
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
                   transport_assignment_policy='directional_row_softmax_with_one_fixed_null',
                   scope='Private3x16 Mamba with row-null mass; same-state slot versus pair-uniform allocation. No OT column capacity, Sinkhorn, native image input, reconstruction, extra loss or external resources.')
    return model,cfg,binding


def condition(args):
    return {**original_condition(args),'transport_iterations':0,
            'transport_assignment_policy':'directional_row_softmax_with_one_fixed_null'}


def configure():
    previous.configure()
    inner.RawFeatureSemanticTriFusion=RowSlotMassTriFusion
    inner.IndependentNativeTriFusion=RowUniformMassTriFusion
    base=previous.previous.previous.base
    base.SCHEMA,base.build_core=SCHEMA,build_core
    base.configure()
    inner.condition=condition
    inner.foundation.condition=condition


if __name__=='__main__':
    configure()
    inner.main()
