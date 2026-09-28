#!/usr/bin/env python3
"""Move one unstarted variant while adopting the donor's live training child."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.queue_correspondence_roles import run, stamp


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def wait_gpu(gpu):
    import subprocess
    while int(subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip()) >= 500:
        time.sleep(240)
    assert shutil.disk_usage(ROOT).free >= 10 * 1024**3


def adopt(args):
    assert not args.transfer_campaign.exists()
    args.transfer_campaign.mkdir()
    donor_path = args.donor_campaign / "campaign.json"
    donor = json.loads(donor_path.read_text())
    assert donor["gpu"] == args.donor_gpu
    assert donor["variants"] == [args.finishing_variant, args.variant]
    assert donor["jobs"][-1]["status"] == "RUNNING"
    row = donor["jobs"][-1]
    assert (row["dataset"], row["variant"], row["mode"]) == ("RGBNT100", args.finishing_variant, "train")
    assert not any(job["variant"] == args.variant for job in donor["jobs"])
    parent_command = Path(f"/proc/{args.donor_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "tools/queue_correspondence_roles.py" in parent_command
    assert str(args.donor_campaign) in parent_command
    trainer_command = Path(f"/proc/{args.trainer_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "tools/run_correspondence_roles.py" in trainer_command and "--mode train" in trainer_command
    assert str(row["output_dir"]) in trainer_command
    trainer_status = Path(f"/proc/{args.trainer_pid}/status").read_text()
    assert f"PPid:\t{args.donor_pid}\n" in trainer_status
    handoff_path = args.transfer_campaign / "handoff.json"
    handoff = {"status": "READY_FOR_HANDOFF", "adopter_pid": os.getpid(),
               "donor_pid": args.donor_pid, "trainer_pid": args.trainer_pid,
               "donor_campaign": str(args.donor_campaign), "variant": args.variant,
               "started_at": stamp()}
    write(handoff_path, handoff)
    while Path(f"/proc/{args.donor_pid}").exists():
        time.sleep(240)
    donor["original_variants"] = donor["variants"]
    donor["variants"] = [args.finishing_variant]
    donor["transferred_variants"] = {args.variant: str(args.transfer_campaign)}
    donor["controller_pid"] = os.getpid()
    write(donor_path, donor)
    handoff.update(status="ADOPTED", adopted_at=stamp())
    write(handoff_path, handoff)
    while Path(f"/proc/{args.trainer_pid}").exists():
        time.sleep(240)
    output_dir = Path(row["output_dir"])
    receipt = json.loads((output_dir / "training.json").read_text())
    if receipt["status"] != "BEST_OFFICIAL_MAP_TRAINING_COMPLETE":
        row.update(status="FAILED", exit_code=None, completed_at=stamp(),
                   reason="adopted_trainer_exited_without_complete_receipt")
        donor.update(status="FAILED", completed_at=stamp())
        handoff.update(status="FAILED", completed_at=stamp())
        write(donor_path, donor)
        write(handoff_path, handoff)
        return 1
    assert [item["epoch"] for item in receipt["history"]] == list(range(1, 51))
    row.update(status="COMPLETE", exit_code=0, completed_at=receipt["completed_at"],
               receipt=str(output_dir / "training.json"), adopted_trainer_pid=args.trainer_pid)
    evaluation = {"dataset": "RGBNT100", "variant": args.finishing_variant,
                  "mode": "evaluate", "output_dir": str(output_dir),
                  "status": "RUNNING", "started_at": stamp()}
    donor["jobs"].append(evaluation)
    write(donor_path, donor)
    wait_gpu(args.donor_gpu)
    run_args = argparse.Namespace(gpu=args.donor_gpu, campaign=args.donor_campaign)
    code = run(run_args, "RGBNT100", args.finishing_variant, "evaluate", output_dir)
    if code == 0:
        metrics = json.loads((output_dir / "official_metrics.json").read_text())
        code = 0 if metrics["status"] == "COMPLETE" else 1
    if code:
        evaluation.update(status="FAILED", exit_code=code, completed_at=stamp())
        donor.update(status="FAILED", completed_at=stamp())
        handoff.update(status="FAILED", completed_at=stamp())
        write(donor_path, donor)
        write(handoff_path, handoff)
        return code
    evaluation.update(status="COMPLETE", exit_code=0, completed_at=metrics["completed_at"],
                      receipt=str(output_dir / "official_metrics.json"))
    donor.update(status="COMPLETE", completed_at=stamp())
    write(donor_path, donor)
    handoff.update(status="COMPLETE", completed_at=stamp())
    write(handoff_path, handoff)
    return 0


def transfer(args):
    handoff_path = args.transfer_campaign / "handoff.json"
    assert handoff_path.exists()
    path = args.transfer_campaign / "campaign.json"
    assert not path.exists()
    status = {"schema": "trifusion-correspondence-roles-queue-v1", "status": "WAITING",
              "gpu": args.gpu, "controller_pid": os.getpid(), "variant": args.variant,
              "output_campaign_prefix": str(args.donor_campaign),
              "after": str(args.after_campaign), "started_at": stamp(), "jobs": []}
    write(path, status)
    while True:
        parent = json.loads((args.after_campaign / "campaign.json").read_text())
        handoff = json.loads(handoff_path.read_text())
        if parent["status"] == "FAILED" or handoff["status"] == "FAILED":
            status.update(status="FAILED", reason="parent_or_handoff_failed", completed_at=stamp())
            write(path, status)
            return 1
        if parent["status"] == "COMPLETE" and handoff["status"] in ("ADOPTED", "COMPLETE"):
            break
        if parent["status"] != "COMPLETE":
            assert Path(f"/proc/{args.after_pid}").exists()
        assert Path(f"/proc/{handoff['adopter_pid']}").exists()
        time.sleep(240)
    donor = json.loads((args.donor_campaign / "campaign.json").read_text())
    assert args.variant not in donor["variants"]
    status["status"] = "RUNNING"
    write(path, status)
    run_args = argparse.Namespace(gpu=args.gpu, campaign=args.transfer_campaign)
    for dataset in ("RGBNT201", "MSVR310", "RGBNT100"):
        for mode in ("m0", "train", "evaluate"):
            suffix = "m0" if mode == "m0" else "full"
            output = ROOT / f"trained-model/{args.donor_campaign.name}_{dataset}_m{args.variant}_seed42_{suffix}"
            if mode != "evaluate":
                assert not output.exists()
            wait_gpu(args.gpu)
            row = {"dataset": dataset, "variant": args.variant, "mode": mode,
                   "output_dir": str(output), "status": "RUNNING", "started_at": stamp()}
            status["jobs"].append(row)
            write(path, status)
            code = run(run_args, dataset, args.variant, mode, output)
            if code == 0:
                name = "official_metrics.json" if mode == "evaluate" else "training.json"
                receipt = json.loads((output / name).read_text())
                expected = {"m0": "M0_PASS", "train": "BEST_OFFICIAL_MAP_TRAINING_COMPLETE", "evaluate": "COMPLETE"}[mode]
                code = 0 if receipt["status"] == expected else 1
            if code:
                row.update(status="FAILED", exit_code=code, completed_at=stamp())
                status.update(status="FAILED", completed_at=stamp())
                write(path, status)
                return code
            row.update(status="COMPLETE", exit_code=0, completed_at=receipt["completed_at"],
                       receipt=str(output / name))
            write(path, status)
    status.update(status="COMPLETE", completed_at=stamp())
    write(path, status)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("adopt", "transfer"))
    parser.add_argument("--donor-campaign", type=Path, required=True)
    parser.add_argument("--transfer-campaign", type=Path, required=True)
    parser.add_argument("--variant", choices=("100", "010", "001"), required=True)
    parser.add_argument("--finishing-variant", choices=("110", "011", "101"))
    parser.add_argument("--donor-gpu", type=int)
    parser.add_argument("--donor-pid", type=int)
    parser.add_argument("--trainer-pid", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--after-pid", type=int)
    args = parser.parse_args()
    if args.mode == "adopt":
        return adopt(args)
    return transfer(args)


if __name__ == "__main__":
    raise SystemExit(main())
