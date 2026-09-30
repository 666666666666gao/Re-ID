#!/usr/bin/env python3
"""Local-versus-complete Patch memory under the existing full50 ReID protocol."""

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
from trifusion.patch_memory_roles import PatchMemoryTriFusion

BASE_BUILD, BASE_EVALUATE = entry.build, entry.evaluate
MEMORY_MODES = ("local", "full")
MEMORY_MODE = None
ARCHITECTURE = "patch_memory_roles_v1"


def build(args, protocol):
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    assert model.memory_mode == MEMORY_MODE
    binding.update(
        architecture=ARCHITECTURE, memory_mode=MEMORY_MODE,
        slot_count=16, patch_count=128, memory_candidates=9 if MEMORY_MODE == "local" else 128,
        fixed_slot_positions=True,
        patch_memory_source_sha256=entry.sha256(ROOT / "modeling/trifusion/patch_memory_roles.py"),
        entry_sha256=entry.sha256(Path(__file__)),
        reused_context_entry_sha256=entry.sha256(ROOT / "tools/run_correspondence_context_identity.py"),
    )
    return model, cfg, config, binding


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({"schema": "trifusion-patch-memory-roles-v1",
                "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                "protocol_sha256": entry.sha256(args.protocol),
                "baseline_sha256": args.baseline_sha256,
                "variants": {"m1": args.m1, "m2": args.m2, "m3": args.m3},
                "condition": entry.CONDITION.copy(), "memory_mode": MEMORY_MODE,
                "metrics": metrics, "state": entry.runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-patch-memory-roles-v1"
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == entry.sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    assert payload["variants"] == {"m1": args.m1, "m2": args.m2, "m3": args.m3}
    assert payload["condition"] == entry.CONDITION
    assert payload["memory_mode"] == MEMORY_MODE == model.memory_mode
    assert set(payload["state"]) == set(entry.runner.checkpoint_state(model))
    state = model.state_dict()
    state.update(payload["state"])
    model.load_state_dict(state, strict=True)
    return payload


def evaluate(args, protocol):
    BASE_EVALUATE(args, protocol)
    path = args.output_dir / "official_metrics.json"
    result = json.loads(path.read_text())
    result.update(architecture=ARCHITECTURE, memory_mode=MEMORY_MODE)
    path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    global MEMORY_MODE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--memory-mode", choices=MEMORY_MODES, required=True)
    options, remaining = parser.parse_known_args()
    MEMORY_MODE = options.memory_mode
    entry.ContextIdentityTriFusion = partial(PatchMemoryTriFusion, memory_mode=MEMORY_MODE)
    entry.build, entry.evaluate = build, evaluate
    entry.save_checkpoint, entry.load_checkpoint = save_checkpoint, load_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    entry.main()


if __name__ == "__main__":
    main()
