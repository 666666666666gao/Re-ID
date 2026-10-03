"""Matched author-package evidence with one full batch across two GPU segments."""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_independent_native_evidence as entry
from trifusion.partitioned_evidence_clip import partition_visual_graph, placement_summary

SCHEMA = 'trifusion-independent-native-model-partition-v5'
original_build_core = entry.build_core
original_condition = entry.condition
original_optimization = entry.optimization
original_foundation_train = entry.foundation.train
runner, clean = entry.runner, entry.clean
original_train_loader = entry.original_train_loader
PLACEMENT = 'single_process_original_full_batch_first6_cuda1_last6_and_heads_cuda0'


def build_core(args, protocol):
    model, cfg, binding = original_build_core(args, protocol)
    before = runner._module_state_sha256(model)
    partition_visual_graph(model.evidence_model.backbone)
    assert runner._module_state_sha256(model) == before
    binding.update(architecture=SCHEMA, entry_sha256=runner.sha256(Path(__file__)),
                   visual_placement=PLACEMENT, parameter_placement=placement_summary(model))
    return model, cfg, binding


def condition(args):
    return dict(original_condition(args), visual_placement=PLACEMENT)


def optimization(args, model, cfg):
    for device in range(2):
        torch.cuda.reset_peak_memory_stats(device)
    return original_optimization(args, model, cfg)


def train(args, protocol):
    original_foundation_train(args, protocol)
    path = args.output_dir/'training.json'
    value = json.loads(path.read_text())
    value['partition_peak_memory'] = {
        f'cuda:{device}': {'allocated_bytes':torch.cuda.max_memory_allocated(device),
                          'reserved_bytes':torch.cuda.max_memory_reserved(device)}
        for device in range(2)}
    path.write_text(json.dumps(value, indent=2)+'\n')


def configure():
    entry.SCHEMA = SCHEMA
    entry.build_core = build_core
    entry.condition = condition
    entry.optimization = optimization
    entry.configure()
    entry.foundation.train = train


if __name__ == '__main__':
    configure()
    entry.main()
