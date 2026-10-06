from pathlib import Path
import ast,hashlib,json

path=Path('C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_selection_reference.py')
old=path.read_bytes()
assert hashlib.sha256(old).hexdigest()=='d40999a7beaea0e8262f61340f3973a585e1a7767c5a3bd119889b9c2c378dcc'
before=("    batch_step+=1\r\n"
        "    with batch_log_path.open('a') as stream:\r\n"
        "        stream.write(json.dumps(dict(global_step=batch_step,labels=raw[1].tolist(),cameras=raw[2].tolist(),\r\n"
        "                                     view_ids=raw[3].tolist(),rgb_basenames=list(raw[4])))+'\\n')\r\n").encode()
after=("    if last_built_model.training:\r\n"
       "        batch_step+=1\r\n"
       "        with batch_log_path.open('a') as stream:\r\n"
       "            stream.write(json.dumps(dict(global_step=batch_step,labels=raw[1].tolist(),cameras=raw[2].tolist(),\r\n"
       "                                         view_ids=raw[3].tolist(),rgb_basenames=list(raw[4])))+'\\n')\r\n").encode()
assert old.count(before)==1
new=old.replace(before,after)
a,b=ast.parse(old),ast.parse(new)
old_others=[n for n in a.body if not (isinstance(n,ast.FunctionDef) and n.name=='training_batch')]
new_others=[n for n in b.body if not (isinstance(n,ast.FunctionDef) and n.name=='training_batch')]
assert [ast.dump(n) for n in old_others]==[ast.dump(n) for n in new_others]
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_logger_fix868')
assert not packet.exists();packet.mkdir()
(packet/'BEFORE.py').write_bytes(old);(packet/'AFTER.py').write_bytes(new)
path.write_bytes(new)
facts=dict(status='LOCAL_LOGGER_PHASE_ONLY_FIX',before_sha256=hashlib.sha256(old).hexdigest(),
    after_sha256=hashlib.sha256(new).hexdigest(),other_module_ast_identical=True,
    original_converter_call_and_return_unchanged=True,line_endings_preserved=True,
    boundary='Onlyappendmetadatawhilemodel.training; converter stillrunsduringeval. No NN/recipe/loss/precision/metric/thresholdchange. Not deployed orformalvalidated. Oldfailedcampaignrecordsunchanged.')
(packet/'FACTS.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
(packet/'SOURCE.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(facts,indent=2))
