"""Use the checked source scope for the unchanged six-result report."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_incremental_checked_m0 as checked
from tools import report_incremental_role_objective as report


if __name__ == '__main__':
    checked.configure()
    report.panel.source_map = checked.source_map
    report.main()
