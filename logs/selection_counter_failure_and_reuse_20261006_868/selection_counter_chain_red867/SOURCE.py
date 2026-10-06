"""CPU replay of the actual shared train/eval converter seam, no images or NN."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,hashlib,json

parser=argparse.ArgumentParser()
parser.add_argument('--entry',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
assert not args.output_dir.exists()
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
intake=base/'selection_counter_failure867_r2'
remote=json.loads((intake/'REMOTE.json').read_bytes())
assert json.loads((intake/'COMPLETE.json').read_bytes())['status']=='COMPLETE'
root=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
folder=intake/'received/trained-model/signal_selection_reference_v1_20261006_867_full_global_only_MSVR310'
input_sha={}
for name in ('training.json','training_steps.jsonl','training_batch_metadata.jsonl'):
    path=folder/name
    key='trained-model/signal_selection_reference_v1_20261006_867_full_global_only_MSVR310/'+name
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==remote['text_files'][key]['sha256'];input_sha[str(path)]=digest
training=json.loads((folder/'training.json').read_bytes())
steps=[json.loads(line) for line in (folder/'training_steps.jsonl').read_text().splitlines()]
metadata=[json.loads(line) for line in (folder/'training_batch_metadata.jsonl').read_text().splitlines()]
protocol_path=root/'logs/training_feature_scale_protocols_20261002/MSVR310.json'
protocol=json.loads(protocol_path.read_bytes())
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()==training['initializer']['protocol_sha256']
input_sha[str(protocol_path)]=training['initializer']['protocol_sha256']
loader_source=root/'tools/train_msvr310_signal_oof.py'
assert hashlib.sha256(loader_source.read_bytes()).hexdigest()==remote['source_sha256']['tools/train_msvr310_signal_oof.py']
loader_ast=ast.parse(loader_source.read_text(encoding='utf-8'))
loader_fn=next(n for n in loader_ast.body if isinstance(n,ast.FunctionDef) and n.name=='loader_for')
assert any(isinstance(n,ast.Call) and any(k.arg=='batch_size' and isinstance(k.value,ast.Constant) and k.value.value==64 for k in n.keywords) for n in ast.walk(loader_fn))
expected_eval=[]
for split in ('query','gallery'):
    records=protocol['records'][split]
    for start in range(0,len(records),64):
        batch=records[start:start+64]
        expected_eval.append(dict(labels=[r['identity'] for r in batch],cameras=[r['camera'] for r in batch],
            view_ids=[r['view'] for r in batch],rgb_basenames=[Path(r['paths'][0]).name for r in batch]))
assert len(expected_eval)==27
train_records={(r['label'],r['camera'],r['view'],Path(r['paths'][0]).name) for r in protocol['records']['train']}
phases=[];optimizer_metadata=[];position=0
for h in training['history']:
    epoch_steps=[s for s in steps if s['epoch']==h['epoch']]
    assert len(epoch_steps)==h['steps'] and [s['batch'] for s in epoch_steps]==list(range(h['steps']))
    for _ in epoch_steps:
        row=metadata[position];position+=1
        assert len(row['labels'])==training['initializer']['batch_size']==64
        assert all(signature in train_records for signature in zip(row['labels'],row['cameras'],row['view_ids'],row['rgb_basenames']))
        canonical=dict(row,global_step=len(optimizer_metadata)+1)
        optimizer_metadata.append(canonical);phases.append((True,row))
    for expected in expected_eval:
        row=metadata[position];position+=1
        assert {k:v for k,v in row.items() if k!='global_step'}==expected
        phases.append((False,row))
assert position==len(metadata)==2056 and len(optimizer_metadata)==len(steps)==706
assert [r['global_step'] for r in metadata]==list(range(1,2057))
eval_source=root/'tools/run_correspondence_roles.py'
assert hashlib.sha256(eval_source.read_bytes()).hexdigest()==remote['source_sha256']['tools/run_correspondence_roles.py']
eval_ast=ast.parse(eval_source.read_text(encoding='utf-8'))
eval_fn=next(n for n in eval_ast.body if isinstance(n,ast.FunctionDef) and n.name=='_eval_batch')
entry_ast=ast.parse(args.entry.read_text(encoding='utf-8'))
logger_fn=next(n for n in entry_ast.body if isinstance(n,ast.FunctionDef) and n.name=='training_batch')
class Sequence(list):
    def tolist(self):return list(self)
converter_calls=[]
def convert(raw):
    converter_calls.append(tuple(raw[1]))
    return dict(images=raw[0],camera_ids=list(raw[2])),raw[1]
args.output_dir.mkdir()
log_path=args.output_dir/'REPLAY_LOG.jsonl'
namespace=dict(json=json,batch_log_path=log_path,batch_step=0,original_training_batch=convert,
    last_built_model=SimpleNamespace(training=True))
seam=ast.Module(body=[logger_fn,eval_fn],type_ignores=[])
exec(compile(seam,'real_logger_and_eval_converter_seam','exec'),namespace)
namespace['_training_batch']=namespace['training_batch']
for training_phase,row in phases:
    namespace['last_built_model'].training=training_phase
    raw=('IMAGE_PLACEHOLDER',Sequence(row['labels']),Sequence(row['cameras']),Sequence(row['view_ids']),row['rgb_basenames'])
    if training_phase:
        batch,labels=namespace['training_batch'](raw)
        assert list(labels)==row['labels']
    else:
        batch=namespace['_eval_batch'](raw,'MSVR310')
    assert batch==dict(images='IMAGE_PLACEHOLDER',camera_ids=row['cameras'])
assert len(converter_calls)==2056
output=[json.loads(line) for line in log_path.read_text().splitlines()]
fact=dict(status='PASS' if output==optimizer_metadata else 'RED_LOGGER_INCLUDES_EVALUATION',
    actual_source_entry_sha256=hashlib.sha256(args.entry.read_bytes()).hexdigest(),
    input_sha256=input_sha,consumed_calls=2056,optimizer_calls=706,evaluation_calls=1350,logged_calls=len(output),
    all50_ordered_evaluation_blocks_verified=True,all_training_rows_in_protocol=True,
    returned_batch_values_unchanged=True,converter_calls_unchanged=True,
    boundary='ActualtwofunctionASTwithcapturedrawmetadata/modelphase. Originaltensor/deviceconversionstubbed; noimages/NN/officialevaluation/retraining orformalacceptance. Canonicaltrainrecordsderivedafterexact50protocolquery/gallerychecks, original2056fileunchanged.')
(args.output_dir/'FACTS.json').write_text(json.dumps(fact,indent=2)+'\n',encoding='utf-8')
(args.output_dir/'DERIVED_OPTIMIZER_BATCHES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in optimizer_metadata),encoding='utf-8')
(args.output_dir/'SOURCE.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(fact,indent=2))
assert output==optimizer_metadata,'Training logger must record706optimizerbatches, not2056train-plus-evalcalls'
