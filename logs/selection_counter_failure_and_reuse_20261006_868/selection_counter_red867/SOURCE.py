from pathlib import Path
import sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_signal_selection_reference as panel
panel.accepted_row(root/'logs/signal_selection_reference_v1_20261006_867','MSVR310','global_only')
