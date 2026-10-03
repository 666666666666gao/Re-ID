"""Reuse the full-batch initialization check for the new research campaign."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import check_partitioned_native_pair as check
from tools import run_native_research, queue_native_research

check.entry = run_native_research
check.panel = queue_native_research

if __name__ == '__main__':
    check.main()
