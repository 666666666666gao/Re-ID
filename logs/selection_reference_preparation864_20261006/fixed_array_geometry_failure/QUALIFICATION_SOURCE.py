inputs={'/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/RGBNT201_semantic/original/diagnostic_arrays.pt': '068a0ac21ed7a120ba09f5e489015a0c8e731e7dcca7713af7baf2f86f4a2466', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/RGBNT201_semantic/original/distances.pt': 'd33bf76f4d16d5fa83b12249ed585d84441f5f135d802bd45713653985cf98fe', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/RGBNT201_native/original/diagnostic_arrays.pt': '00969b13f9a38bc27e08e58c72013828a05996564b7b16c2b24256aaa5358b22', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/RGBNT201_native/original/distances.pt': '5984711c4b234980d3472b23435d80310f376367a78233c67c4138d6bc06a800', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/MSVR310_semantic/original/diagnostic_arrays.pt': 'f4aefa3e27695817e82b28623eb7555debcc98fca31dad0fe172281f734512e0', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/MSVR310_semantic/original/distances.pt': '53363bc5437aff02ee51ba12745df5e9273c118df0c60a9e54f55d0df7ea444a', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/MSVR310_native/original/diagnostic_arrays.pt': 'bc5336e3bf354bfbfdcd9e9078b30d8c912b7c86cbc392ab8558d62dac694af5', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/MSVR310_native/original/distances.pt': '2c035cd5ee684b94eadd2ae1dabb57fd85187767f2e280a8c84159d44ff2e03d', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/RGBNT100_semantic/original/diagnostic_arrays.pt': '677659f39c47688ca0d7bebca29753790bc2fbdfbe87d68252fb466bba089490', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/RGBNT100_semantic/original/distances.pt': 'b6a8b7e8bde240f22d550bebe924d9653854667f27dc38b7bfcaa01b637f7022', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/RGBNT100_native/original/diagnostic_arrays.pt': '2dac7670f9040aa7ac2a8d7e53dd0638077b4b5a8a8d15f3a656b6c9239f29b0', '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862/RGBNT100_native/original/distances.pt': '3d640d8f1c04a8535f7469658d7d0dc1c0a17ce9762f9be23f9f363bbcc78c46'}
sources={'/data/gaob/Re-ID/Trifusion/tools/report_row_mass_fixed_best.py': '56d10437b598faceb07268e5362e560af91f8c5d5a9a9bb7e21fedcc422e3e8d', '/data/gaob/Re-ID/Trifusion/tools/run_official_three_dataset_roles.py': 'f30bb00a0b71429e9d11dd196cd1d543202a133c663ad66e175d2f82e28d448d', '/data/gaob/Re-ID/Trifusion/tools/train_rgbnt100_signal_oof.py': '4462b73139e034d450d955c5b4a994aaa967548e9cc503d97bb753a14ea03b22', '/data/gaob/Re-ID/Trifusion/tools/train_msvr310_signal_oof.py': 'c25579d931df34481fef321558f0b9a1f9179d72a25d6047dbbfd3682d4ae60f'}
launch='/data/gaob/Re-ID/Trifusion/logs/additive_geometry_fixed_arrays_launch_20261006_864'
output='/data/gaob/Re-ID/Trifusion/results/additive_geometry_fixed_arrays_20261006_864'
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
assert not Path(launch).exists() and not Path(output).exists()
assert json.loads((root/'logs/fixed_best_row_transport_resume_20261006_863/EXIT.json').read_text())['exit_code']==0
active=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit() and (p/'cmdline').is_file():
  cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if '/data/gaob/Re-ID/Trifusion/tools/' in cmd:active.append(p.name)
assert not active,active
assert all(sha(p)==d for p,d in inputs.items()) and all(sha(p)==d for p,d in sources.items())
for p in list(inputs):
 if p.endswith('distances.pt'):
  q=str(Path(p).parent/'own_global.json');inputs[q]=sha(q)
assert shutil.disk_usage(root).free>2*1024**3+256*1024**2
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),input_sha256=inputs,source_sha256=sources,free_bytes=shutil.disk_usage(root).free,no_nn=True)))
