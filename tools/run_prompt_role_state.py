#!/usr/bin/env python3
"""Reset-versus-carry role prompts with the existing full50 ReID protocol."""

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
from trifusion.prompt_role_state import PromptRoleTriFusion

BASE_BUILD, BASE_EVALUATE = entry.build, entry.evaluate
PROMPT_MODES = ("reset", "carry")
PROMPT_MODE = None
ARCHITECTURE = "prompt_role_state_v1"


def build(args, protocol):
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    assert model.prompt_mode == PROMPT_MODE
    binding.update(
        architecture=ARCHITECTURE, prompt_mode=PROMPT_MODE,
        prompt_tokens_per_role=4, prompt_roles=3, prompt_blocks=12,
        prompt_source_sha256=entry.sha256(ROOT / "modeling/trifusion/prompt_role_state.py"),
        entry_sha256=entry.sha256(Path(__file__)),
        reused_context_entry_sha256=entry.sha256(ROOT / "tools/run_correspondence_context_identity.py"),
    )
    return model, cfg, config, binding


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({"schema": "trifusion-prompt-role-state-v1",
                "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                "protocol_sha256": entry.sha256(args.protocol),
                "baseline_sha256": args.baseline_sha256,
                "variants": {"m1": args.m1, "m2": args.m2, "m3": args.m3},
                "condition": entry.CONDITION.copy(), "prompt_mode": PROMPT_MODE,
                "metrics": metrics, "state": entry.runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-prompt-role-state-v1"
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == entry.sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    assert payload["variants"] == {"m1": args.m1, "m2": args.m2, "m3": args.m3}
    assert payload["condition"] == entry.CONDITION
    assert payload["prompt_mode"] == PROMPT_MODE == model.prompt_mode
    assert set(payload["state"]) == set(entry.runner.checkpoint_state(model))
    state = model.state_dict()
    state.update(payload["state"])
    model.load_state_dict(state, strict=True)
    return payload


def evaluate(args, protocol):
    BASE_EVALUATE(args, protocol)
    path = args.output_dir / "official_metrics.json"
    result = json.loads(path.read_text())
    result.update(architecture=ARCHITECTURE, prompt_mode=PROMPT_MODE)
    path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    global PROMPT_MODE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, required=True)
    options, remaining = parser.parse_known_args()
    PROMPT_MODE = options.prompt_mode
    entry.ContextIdentityTriFusion = partial(PromptRoleTriFusion, prompt_mode=PROMPT_MODE)
    entry.build, entry.evaluate = build, evaluate
    entry.save_checkpoint, entry.load_checkpoint = save_checkpoint, load_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    entry.main()


if __name__ == "__main__":
    main()
