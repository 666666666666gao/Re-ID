"""Wait at240-second cadence, then execute the full clean-public CPU report once."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_clean_clip_joint as panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args = parser.parse_args()
    assert not args.receipt.exists() and not args.output_dir.exists()
    entry = ROOT/'tools/report_clean_clip_joint_complete.py'
    digest = panel.sha(entry)
    analyzer = ROOT/'tools/analyze_correspondence_distances.py'
    analyzer_digest = panel.sha(analyzer)
    manifest = panel.require_sources(args.campaign)
    receipt = {'status':'WAITING_FULL6','started_at':datetime.now().astimezone().isoformat(),
               'campaign':str(args.campaign),'output_dir':str(args.output_dir),
               'report_source_sha256':digest,'analyzer_source_sha256':analyzer_digest,
               'poll_seconds':240,'invocations':0}
    panel.queue.write(args.receipt,receipt)
    while True:
        state = json.loads((args.campaign/'campaign.json').read_text())
        if state['status'] == 'FAILED':
            receipt.update(status='CAMPAIGN_FAILED_NO_REPORT',completed_at=panel.queue.stamp())
            panel.queue.write(args.receipt,receipt)
            return 1
        if state['status'] == 'COMPLETE':
            break
        time.sleep(240)
    assert panel.require_sources(args.campaign) == manifest and panel.sha(entry) == digest
    assert panel.sha(analyzer) == analyzer_digest
    command = [sys.executable,'-B',str(entry),'--campaign',str(args.campaign),
               '--output-dir',str(args.output_dir)]
    receipt.update(status='REPORT_RUNNING',invocations=1,command=command,report_started_at=panel.queue.stamp())
    panel.queue.write(args.receipt,receipt)
    with args.receipt.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        receipt['report_pid'] = child.pid
        panel.queue.write(args.receipt,receipt)
        code = child.wait()
    receipt.update(status='CPU_REPORT_COMPLETE' if code == 0 else 'CPU_REPORT_FAILED',
                   completed_at=panel.queue.stamp(),exit_code=code)
    panel.queue.write(args.receipt,receipt)
    print(json.dumps(receipt),flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
