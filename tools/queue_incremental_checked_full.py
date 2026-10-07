"""Use the unchanged full50 protocol after the new six M0s are accepted."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_incremental_checked_m0 as checked
from tools import queue_incremental_role_objective_full as full


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'm0-campaign', 'report-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.campaign, args.m0_campaign, args.report_dir = args.campaign.resolve(), args.m0_campaign.resolve(), args.report_dir.resolve()
    checked.configure()
    full.source_map = checked.source_map
    full.REPORT_ENTRY = ROOT / 'tools/report_incremental_checked.py'
    raise SystemExit(full.coordinate(args))
