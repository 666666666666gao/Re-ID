from pathlib import Path
from datetime import datetime
import subprocess,json,os
folder=Path('/data/gaob/Re-ID/Trifusion/logs/selection_report_completion_20261006_871')
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/tools/report_signal_selection_reference.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/signal_selection_reference_v1_20261006_870', '--origin-campaigns', '/data/gaob/Re-ID/Trifusion/logs/signal_selection_reference_v1_20261006_870/endpoint_origins.json', '--batch-metadata-paths', '/data/gaob/Re-ID/Trifusion/logs/signal_selection_reference_v1_20261006_870/batch_metadata_paths.json', '--output-dir', '/data/gaob/Re-ID/Trifusion/results/signal_selection_reference_complete_20261006_871']
with (folder/"stdout.txt").open("x") as out,(folder/"stderr.txt").open("x") as err:
 p=subprocess.Popen(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"),stdout=out,stderr=err)
 (folder/"CHILD.json").write_text(json.dumps(dict(pid=p.pid,start_ticks=int(Path(f"/proc/{p.pid}/stat").read_text().split()[21]),started_at=datetime.now().astimezone().isoformat()))+"\n")
 code=p.wait()
(folder/"EXIT.json").write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+"\n")
raise SystemExit(code)
