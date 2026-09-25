#!/usr/bin/env python3
"""Compare the two completed V8 starts on all MSVR310 training identities."""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "logs/official_extra_seed42_RGBNT201_PLAIN_V27_20260925/campaign.json"
OUTPUT = ROOT / "logs/msvr_source_v8_probe_gpu3_20260925"
PYTHON = Path("/data/gaob/Re-ID/conda-envs/tri_reid/bin/python")


def stamp():
    return datetime.now().astimezone().isoformat()


def main():
    assert ROOT == Path("/data/gaob/Re-ID/Trifusion")
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    status_path = OUTPUT / "campaign.json"
    status = {"status": "WAITING", "created_at": stamp(), "gpu": 3,
              "after_campaign": str(PRIOR), "jobs": []}

    def save():
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    save()
    while True:
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        if prior["status"] == "COMPLETE":
            break
        assert prior["status"] == "RUNNING"
        time.sleep(240)
    status["status"] = "RUNNING"
    status["started_at"] = stamp()
    save()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    for method in ("PLAIN_V8", "SIGNAL_V8"):
        for seed in (42, 43, 44):
            tag = f"MSVR310_{method}_seed{seed}"
            receipt = (ROOT / f"trained-model/official_extra_seed{seed}_MSVR310_{method}_20260925"
                       / tag / "training.json")
            result = OUTPUT / f"{tag}.json"
            job = {"method": method, "seed": seed, "status": "RUNNING", "started_at": stamp()}
            status["jobs"].append(job)
            save()
            command = [str(PYTHON), "-B", "-u", str(ROOT / "tools/diagnose_official_source_top_rank.py"),
                       "--training-receipt", str(receipt),
                       "--signal-source", str(ROOT / "comparators/Signal-cd1b0a6"),
                       "--clip-weight", str(ROOT / "pertrained-model/ViT-B-16.pt"),
                       "--role-state", "final", "--output", str(result)]
            with (OUTPUT / f"{tag}.log").open("x", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            probe = json.loads(result.read_text(encoding="utf-8"))
            assert probe["method"] == method and probe["seed"] == seed
            job.update(status="COMPLETE", completed_at=stamp(), result=str(result))
            save()
    status["status"] = "COMPLETE"
    status["completed_at"] = stamp()
    save()


if __name__ == "__main__":
    main()
