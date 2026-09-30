#!/usr/bin/env python3
"""Matched attention allocation under the existing full50 ReID protocol."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modeling"))
from tools import run_correspondence_context_identity as entry
from trifusion.slot_competition_fp32_roles import FP32SlotCompetitionTriFusion

BASE_BUILD, BASE_EVALUATE = entry.build, entry.evaluate
NORMALIZATIONS = ("independent", "competitive")
NORMALIZATION = None
ARCHITECTURE = "slot_competition_fp32_roles_v2"


def build(args, protocol):
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    assert model.attention_normalization == NORMALIZATION and model.memory_mode == "full"
    binding.update(
        architecture=ARCHITECTURE, memory_mode="full", attention_normalization=NORMALIZATION,
        attention_subgraph_dtype="float32",
        slot_count=16, patch_count=128, memory_candidates=128, fixed_slot_positions=True,
        patch_memory_source_sha256=entry.sha256(ROOT / "modeling/trifusion/patch_memory_roles.py"),
        slot_competition_source_sha256=entry.sha256(ROOT / "modeling/trifusion/slot_competition_roles.py"),
        slot_competition_fp32_source_sha256=entry.sha256(ROOT / "modeling/trifusion/slot_competition_fp32_roles.py"),
        entry_sha256=entry.sha256(Path(__file__)),
        reused_context_entry_sha256=entry.sha256(ROOT / "tools/run_correspondence_context_identity.py"),
    )
    return model, cfg, config, binding


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({"schema": "trifusion-slot-competition-fp32-roles-v2",
                "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                "protocol_sha256": entry.sha256(args.protocol),
                "baseline_sha256": args.baseline_sha256,
                "variants": {"m1": args.m1, "m2": args.m2, "m3": args.m3},
                "condition": entry.CONDITION.copy(), "memory_mode": "full",
                "attention_normalization": NORMALIZATION,
                "attention_subgraph_dtype": "float32",
                "metrics": metrics, "state": entry.runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-slot-competition-fp32-roles-v2"
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == entry.sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    assert payload["variants"] == {"m1": args.m1, "m2": args.m2, "m3": args.m3}
    assert payload["condition"] == entry.CONDITION
    assert payload["memory_mode"] == model.memory_mode == "full"
    assert payload["attention_normalization"] == NORMALIZATION == model.attention_normalization
    assert payload["attention_subgraph_dtype"] == "float32"
    assert set(payload["state"]) == set(entry.runner.checkpoint_state(model))
    state = model.state_dict()
    state.update(payload["state"])
    model.load_state_dict(state, strict=True)
    return payload


def evaluate(args, protocol):
    BASE_EVALUATE(args, protocol)
    path = args.output_dir / "official_metrics.json"
    result = json.loads(path.read_text())
    result.update(architecture=ARCHITECTURE, memory_mode="full", attention_normalization=NORMALIZATION,
                  attention_subgraph_dtype="float32")
    path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    global NORMALIZATION
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-normalization", choices=NORMALIZATIONS, required=True)
    options, remaining = parser.parse_known_args()
    NORMALIZATION = options.attention_normalization
    entry.ContextIdentityTriFusion = partial(FP32SlotCompetitionTriFusion, attention_normalization=NORMALIZATION)
    entry.build, entry.evaluate = build, evaluate
    entry.save_checkpoint, entry.load_checkpoint = save_checkpoint, load_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    entry.main()


if __name__ == "__main__":
    main()
