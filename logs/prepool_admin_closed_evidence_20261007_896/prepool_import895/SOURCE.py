import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import ast,hashlib,importlib.metadata,json,sys
root=Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,str(root))
scope=json.loads((root/'refine-logs/prepool_dense_correspondence_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert len(scope)==428 and all(sha(root/name)==digest for name,digest in scope.items())
names=['modeling/trifusion/prepool_dense_correspondence.py','tools/run_prepool_dense_correspondence.py',
       'tools/queue_prepool_dense_correspondence.py','tools/report_prepool_dense_correspondence.py']
for name in names: ast.parse((root/name).read_text(),feature_version=(3,10))
from tools import run_prepool_dense_correspondence as entry
from tools import queue_prepool_dense_correspondence as queue
from tools import report_prepool_dense_correspondence as report
entry.configure()
entry.inner.configure()  # The actual inner.main repeats this configuration.
assert entry.inner.AuthorHeadEvidence is entry.PrepoolDenseHeads
assert entry.inner.runner._training_batch is entry.training_batch
assert entry.inner.train_loader is entry.train_loader
assert entry.inner.loss_values is entry.loss_values
assert entry.inner.runner.official_metrics is entry.official_metrics
assert entry.inner.foundation.build_core is entry.build_core
assert entry.inner.foundation.train_loader is entry.train_loader
assert entry.inner.foundation.loss_values is entry.loss_values
assert entry.inner.foundation.SCHEMA == entry.SCHEMA == queue.SCHEMA
assert queue.source_map()==scope
assert report.panel is queue
versions={name:importlib.metadata.version(name) for name in ('torch','numpy','mamba-ssm','timm')}
assert versions=={'torch':'2.5.1+cu121','numpy':'1.24.4','mamba-ssm':'2.2.6.post3','timm':'1.0.15'}
assert all(sha(root/name)==digest for name,digest in scope.items())
print(json.dumps({'status':'ACTUAL_IMPORT_CONFIGURE_CHAIN_PASS','at':datetime.now().astimezone().isoformat(),
    'versions':versions,'source_count':428,'new_code_ast':4,'production_model_constructions':0,
    'production_NN_forwards':0,'optimizer_updates':0,'cuda_visible_devices':'',
    'boundary':'Warm actual CPU imports and twice-configured pointer/schema chain only; no actual initializer/teacher/M0/full50, no GPU/temperature/power/25 query or package install.'}))
