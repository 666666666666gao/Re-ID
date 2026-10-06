"""Actual acceptance-function replay using sealed texts and verified binary hashes."""
from pathlib import Path
import argparse,hashlib,json

parser=argparse.ArgumentParser()
parser.add_argument('--case',choices=('original','canonical'),required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args();assert not args.output_dir.exists()
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
intake=base/'selection_counter_failure867_r2'
remote=json.loads((intake/'REMOTE.json').read_bytes())
assert json.loads((intake/'COMPLETE.json').read_bytes())['status']=='COMPLETE'
fixture_root=intake/'received'
campaign=fixture_root/'logs/signal_selection_reference_v1_20261006_867'
folder=fixture_root/'trained-model/signal_selection_reference_v1_20261006_867_full_global_only_MSVR310'
for name in ('training.json','training_steps.jsonl','training_batch_metadata.jsonl','official_metrics.json'):
    p=folder/name;key=str(p.relative_to(fixture_root)).replace('\\','/')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==remote['text_files'][key]['sha256']
hashes={str(folder/name):info['sha256'] for name,info in remote['weights'].items()}
hashes[str(folder/'official_metrics.json')]=hashlib.sha256((folder/'official_metrics.json').read_bytes()).hexdigest()
source=Path('C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_signal_selection_reference.py')
namespace=dict(__name__='sealed_count_consumer_fixture',__file__=str(source))
exec(compile(source.read_bytes(),str(source),'exec'),namespace)
namespace['ROOT']=fixture_root
namespace['sha']=lambda p:hashes[str(Path(p))]
canonical=base/'selection_counter_chain_green868/DERIVED_OPTIMIZER_BATCHES.jsonl'
assert hashlib.sha256(canonical.read_bytes()).hexdigest()=='48ed83d5118036bd0ad322389a249be5efb028daed074c68c629e72270b37472'
args.output_dir.mkdir()
(args.output_dir/'SOURCE.py').write_bytes(Path(__file__).read_bytes())
(args.output_dir/'INPUT.json').write_text(json.dumps(dict(case=args.case,actual_queue_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    canonical_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest(),
    boundary='Actualaccepted_rowfunction/sealedjsons; physicalbinarySHAcallbacksfromread-onlyremoteintake, binariesnotloadedlocal. Notformalacceptance/NN/officialrerun oroldcampaignstatusrewrite.'),indent=2)+'\n')
if args.case=='original':
    namespace['accepted_row'](campaign,'MSVR310','global_only')
    raise AssertionError('Original2056recordpathmustremainRED')
row=namespace['accepted_row'](campaign,'MSVR310','global_only',batch_metadata_path=canonical)
assert row['formal_steps']==706 and row['best_epoch']==38 and row['metrics']==remote['first_strict']['metrics']
assert row['checkpoint_sha256']==remote['weights']['best_map.pth']['sha256']
(args.output_dir/'FACTS.json').write_text(json.dumps(dict(status='PASS_EXPLICIT_CANONICAL_CONSUMER',row=row,
    formal_acceptance=False,original_log_untouched=True,original_count_guard_unchanged=True),indent=2)+'\n')
print(json.dumps(dict(status='PASS_EXPLICIT_CANONICAL_CONSUMER',formal_steps=row['formal_steps'],best_epoch=row['best_epoch'],formal_acceptance=False)))
