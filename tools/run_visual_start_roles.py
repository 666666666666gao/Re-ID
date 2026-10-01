"""Run a fixed STATIC role model with the registered frozen visual initialization."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_role_global_tokens as base

BASE_BUILD = base.build
VISUAL_START = None
INPUTS_PATH = None


def build(args, protocol):
    assert base.TOKEN_MODE == 'static'
    inputs = json.loads(INPUTS_PATH.read_text())
    assert inputs['schema'] == 'trifusion-visual-start-inputs-v1'
    row = inputs['datasets'][args.dataset][VISUAL_START]
    assert Path(row['path']).resolve() == args.baseline_checkpoint.resolve()
    assert row['sha256'] == args.baseline_sha256
    assert base.entry.sha256(args.clip_weight) == inputs['public_clip_sha256']
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    binding.update(visual_start=VISUAL_START, visual_start_inputs_sha256=base.entry.sha256(INPUTS_PATH),
                   visual_start_entry_sha256=base.entry.sha256(Path(__file__)),
                   baseline_kind=('independently_trained_module_free_cls' if VISUAL_START == 'reid_visual'
                                  else 'public_visual_reset_retaining_trained_camera_and_nonvisual_state'))
    return model, cfg, config, binding


def main():
    global VISUAL_START, INPUTS_PATH
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--visual-start', choices=('reid_visual', 'public_visual'), required=True)
    parser.add_argument('--visual-start-inputs', type=Path, required=True)
    options, remaining = parser.parse_known_args()
    VISUAL_START, INPUTS_PATH = options.visual_start, options.visual_start_inputs.resolve()
    base.build = build
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == '__main__':
    main()
