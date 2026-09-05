import hashlib,json,subprocess,sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
summary=Path("evidence/trifusion_v22_q1_seed42_5ae096b.json")
audit=Path("evidence/trifusion_v22_terminal_array_verification_20260905.json")
comparison=Path("evidence/trifusion_v22_complete_comparison_20260905.json")
report=Path("results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_2026-09-05.md")
assert not audit.exists() and not comparison.exists() and not report.exists()
subprocess.run([sys.executable,"tools/audit_v22_terminal_arrays.py","--summary",str(summary),
                "--metadata","evidence/trifusion_source_camera_metadata_20260905.json","--output",str(audit)],check=True)
subprocess.run([sys.executable,"tools/report_v22_complete_comparison.py","--summary",str(summary),
                "--array-audit",str(audit),"--output-json",str(comparison),"--output-md",str(report)],check=True)
d=json.loads(summary.read_bytes())
for fold in d["folds"]:
    for name,endpoint in fold["endpoints"].items():
        local=Path(f"evidence/trifusion_v22_fold_{fold['fold']}_{name}_receipt.json")
        assert json.loads(local.read_bytes())==endpoint
duplicate=Path("evidence/trifusion_v22_partial_q1_20260905_204357.json")
assert duplicate.read_bytes()==summary.read_bytes()
archive=Path(r"C:/Users/gb/.codex_tmp/trifusion_v22_terminal_progress_snapshot_20260905_204357.json")
assert archive.parent.resolve()==Path(r"C:/Users/gb/.codex_tmp").resolve() and not archive.exists()
duplicate.rename(archive)
print(json.dumps({"status":d["status"],"aggregate":d["aggregate"],"matched_gains_mAP":d["matched_gains_mAP"],
                  "fold_fused_gains_mAP":d["fold_fused_gains_mAP"],"scientific_checks":d["scientific_checks"],
                  "bootstrap":d["bootstrap"],"elapsed_seconds":d["elapsed_seconds"],
                  "files":[{"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in (summary,audit,comparison,report)],
                  "duplicate_progress_snapshot_archived":str(archive)},indent=2))
