"""Read-only fixed-best diagnosis after the six role-read interventions finish."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnose_native_research_best as diagnosis
from tools import run_role_input_detach as intervention

diagnosis.SCHEMA = 'trifusion-role-input-detach-fixed-best-diagnosis-v1'
diagnosis.__file__ = str(Path(__file__).resolve())

if __name__ == '__main__':
    intervention.configure()
    raise SystemExit(diagnosis.main())
