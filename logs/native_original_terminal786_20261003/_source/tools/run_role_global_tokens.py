#!/usr/bin/env python3
"""Full50 role/global interaction study with fixed official retrieval scoring."""

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
from trifusion.role_global_tokens import GlobalTokenTriFusion, TOKEN_MODES

BASE_BUILD, BASE_EVALUATE = entry.build, entry.evaluate
TOKEN_MODE = None
ARCHITECTURE = "role_global_tokens_v1"
SCHEMA = "trifusion-role-global-tokens-v1"


def build(args, protocol):
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    assert model.token_mode == TOKEN_MODE and model.memory_mode == "full"
    assert model.attention_normalization == "independent"
    binding.update(architecture=ARCHITECTURE, token_mode=TOKEN_MODE, memory_mode="full",
                   attention_normalization="independent", attention_subgraph_dtype="float32",
                   slot_count=16, transformer_sequence_length=17, patch_count=128,
                   memory_candidates=128, fixed_slot_positions=True,
                   joint_local_semantics="normalized total role correction; global-conditioned in token/direct modes",
                   patch_memory_source_sha256=entry.sha256(ROOT / "modeling/trifusion/patch_memory_roles.py"),
                   slot_competition_source_sha256=entry.sha256(ROOT / "modeling/trifusion/slot_competition_roles.py"),
                   slot_competition_fp32_source_sha256=entry.sha256(ROOT / "modeling/trifusion/slot_competition_fp32_roles.py"),
                   role_global_source_sha256=entry.sha256(ROOT / "modeling/trifusion/role_global_tokens.py"),
                   entry_sha256=entry.sha256(Path(__file__)),
                   reused_context_entry_sha256=entry.sha256(ROOT / "tools/run_correspondence_context_identity.py"))
    return model, cfg, config, binding


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({"schema": SCHEMA, "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                "protocol_sha256": entry.sha256(args.protocol), "baseline_sha256": args.baseline_sha256,
                "variants": {"m1": args.m1, "m2": args.m2, "m3": args.m3},
                "condition": entry.CONDITION.copy(), "token_mode": TOKEN_MODE,
                "memory_mode": "full", "attention_normalization": "independent",
                "attention_subgraph_dtype": "float32",
                "metrics": metrics, "state": entry.runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == SCHEMA
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == entry.sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    assert payload["variants"] == {"m1": args.m1, "m2": args.m2, "m3": args.m3}
    assert payload["condition"] == entry.CONDITION
    assert payload["token_mode"] == TOKEN_MODE == model.token_mode
    assert payload["memory_mode"] == model.memory_mode == "full"
    assert payload["attention_normalization"] == model.attention_normalization == "independent"
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
    result.update(architecture=ARCHITECTURE, token_mode=TOKEN_MODE,
                  memory_mode="full", attention_normalization="independent",
                  attention_subgraph_dtype="float32",
                  joint_local_semantics="normalized total correction, not independent local training")
    path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    global TOKEN_MODE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--token-mode", choices=TOKEN_MODES, required=True)
    options, remaining = parser.parse_known_args()
    TOKEN_MODE = options.token_mode
    entry.ContextIdentityTriFusion = partial(GlobalTokenTriFusion, token_mode=TOKEN_MODE)
    entry.build, entry.evaluate = build, evaluate
    entry.save_checkpoint, entry.load_checkpoint = save_checkpoint, load_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    entry.main()


if __name__ == "__main__":
    main()
