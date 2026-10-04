"""Read-only fixed-best decomposition after all original task-ownership runs close."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnose_native_research_best as diagnosis
from tools import run_global_task_role as intervention

intervention.runner = diagnosis.entry.runner
diagnosis.entry = intervention

diagnosis.SCHEMA = 'trifusion-global-task-role-fixed-best-diagnosis-v1'
diagnosis.__file__ = str(Path(__file__).resolve())

if __name__ == '__main__':
    intervention.configure()
    raise SystemExit(diagnosis.main())
