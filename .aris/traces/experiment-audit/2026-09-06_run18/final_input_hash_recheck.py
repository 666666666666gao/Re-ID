import hashlib
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN_DIR = pathlib.Path(__file__).resolve().parent
MANIFEST_PATH = RUN_DIR / "input_manifest.json"
OUTPUT_PATH = RUN_DIR / "final_input_hash_recheck.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_manifest_entries(manifest: dict):
    entries = []
    seen = set()

    def visit(node, route=""):
        if isinstance(node, dict):
            path_value = node.get("path") or node.get("local_path") or node.get("snapshot_path")
            sha_value = node.get("sha256") or node.get("sha256_hex") or node.get("expected_sha256")
            if isinstance(path_value, str) and isinstance(sha_value, str) and len(sha_value) == 64:
                key = (path_value, sha_value)
                if key not in seen:
                    seen.add(key)
                    entries.append({"route": route, "path": path_value, "expected_sha256": sha_value})
            for k, v in node.items():
                visit(v, f"{route}/{k}" if route else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, f"{route}[{i}]")

    visit(manifest)
    return entries


def resolve_entry_path(path_text: str) -> pathlib.Path:
    p = pathlib.Path(path_text)
    if p.is_absolute():
        return p
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate
    return RUN_DIR / p


def main() -> int:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = collect_manifest_entries(manifest)
    checks = []
    for entry in entries:
        resolved = resolve_entry_path(entry["path"])
        actual = sha256_file(resolved) if resolved.exists() else None
        checks.append({
            "route": entry["route"],
            "path": entry["path"],
            "resolved_path": str(resolved),
            "expected_sha256": entry["expected_sha256"],
            "actual_sha256": actual,
            "exists": resolved.exists(),
            "match": actual == entry["expected_sha256"],
        })
    elapsed = time.perf_counter() - started
    result = {
        "schema": "msvr310-trifusion-m0-final-input-hash-recheck-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(pathlib.Path(__file__).resolve())],
        "working_directory": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "entry_count": len(checks),
        "all_exist": all(c["exists"] for c in checks),
        "all_match": all(c["match"] for c in checks),
        "elapsed_seconds": elapsed,
        "checks": checks,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "manifest_sha256": result["manifest_sha256"],
        "entry_count": result["entry_count"],
        "all_exist": result["all_exist"],
        "all_match": result["all_match"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output_path": str(OUTPUT_PATH),
    }, indent=2, sort_keys=True))
    return 0 if result["all_exist"] and result["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
