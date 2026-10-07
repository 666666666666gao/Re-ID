"""Per-endpoint author-package research training on physical GPU0/1."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_native_partitioned as entry

SCHEMA = 'trifusion-independent-native-research-v6'
runner, clean = entry.runner, entry.clean
original_train_loader = entry.original_train_loader
original_build_core = entry.build_core


def build_core(args, protocol):
    model, cfg, binding = original_build_core(args, protocol)
    binding['entry_sha256'] = entry.runner.sha256(Path(__file__))
    return model, cfg, binding


def configure():
    entry.SCHEMA = SCHEMA
    entry.build_core = build_core
    entry.configure()


if __name__ == '__main__':
    configure()
    entry.entry.main()
