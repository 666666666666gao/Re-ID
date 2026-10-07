"""The fixed native plan with original visual saved tensors stored on CPU."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_independent_native_evidence as entry
from trifusion.cpu_saved_evidence_clip import save_visual_tensors_on_cpu

SCHEMA = 'trifusion-independent-native-cpu-saved-v4'
original_build_core = entry.build_core
original_condition = entry.condition
runner, clean = entry.runner, entry.clean
original_train_loader = entry.original_train_loader


def build_core(args, protocol):
    model, cfg, binding = original_build_core(args, protocol)
    before = entry.runner._module_state_sha256(model)
    save_visual_tensors_on_cpu(model.evidence_model.backbone)
    assert entry.runner._module_state_sha256(model) == before
    binding.update(architecture=SCHEMA, entry_sha256=entry.runner.sha256(Path(__file__)),
                   saved_tensor_storage='original_visual_graph_saved_on_cpu_pin_false')
    return model, cfg, binding


def condition(args):
    return dict(original_condition(args),
                saved_tensor_storage='original_visual_graph_saved_on_cpu_pin_false')


def configure():
    entry.SCHEMA = SCHEMA
    entry.build_core = build_core
    entry.condition = condition
    entry.configure()


if __name__ == '__main__':
    configure()
    entry.main()
