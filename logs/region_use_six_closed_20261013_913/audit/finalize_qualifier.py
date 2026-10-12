"""Record the minimal publication qualifier and preserve the failed display call."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
audit = json.loads((HERE / 'AUDIT.json').read_text(encoding='utf-8'))
original = BASE / 'SIX_RESULTS_REPORT_913.md'
qualified = BASE / 'SIX_RESULTS_REPORT_913_QUALIFIED.md'
master = BASE / 'MASTER_SECTION_913_DRAFT.md'
original_text = original.read_text(encoding='utf-8')
qualified_text = qualified.read_text(encoding='utf-8')
expected_original_hash = audit['reviewed_hashes'][str(original)]
assert sha(original) == expected_original_hash
assert qualified_text.startswith(original_text)
addition = qualified_text[len(original_text):]
assert addition.strip() == 'Publication qualifier: the cleanup/protected-binary and archive-retention statements above are supported by the supplied cleanup receipts and their recorded checks. This bounded result audit does not independently re-read all protected remote binaries or all historical local archives.'
qualifier_line = next(i for i, line in enumerate(qualified_text.splitlines(), 1) if line.startswith('Publication qualifier:'))
master_lines = master.read_text(encoding='utf-8').splitlines()
master_qualifier = master_lines[44]
assert '没有独立重读全部远端二进制或全部历史本地归档' in master_qualifier
qualifier = dict(status='ACCEPTED', reviewed_at=datetime.now().astimezone().isoformat(),
    original_report=str(original), original_sha256=sha(original), original_report_unchanged=True,
    qualified_report=str(qualified), qualified_sha256=sha(qualified), original_text_preserved_as_exact_prefix=True,
    appended_text=addition, evidence=str(qualified) + ':' + str(qualifier_line),
    master_qualifier_evidence=str(master) + ':45', master_qualifier_text=master_qualifier,
    master_draft_sha256=sha(master), master_review_scope='Only the added receipt/provenance qualification, not a wholesale audit of the unpublished master draft.',
    replayed_scientific_or_29060_checks=False, actual_checks=4)
(HERE / 'CHECKS_QUALIFIER.json').write_text(json.dumps(qualifier, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

stdin_source = r"""from pathlib import Path
import json
p=Path(r'C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\six_results913')
a=json.loads((p/'AUDIT.json').read_text(encoding='utf-8'))
print('verdict',a['verdict'],'blockers',a['blockers'],'checks',a['deterministic_checks']['total_primary_checks'])
print('writer_failed_exit',a['failed_artifact_writer_invocations'][0]['receipt']['exit_code'])
print('writer_final_exit',json.loads((p/'WRITE_AUDIT_EXECUTION_02.json').read_text(encoding='utf-8-sig'))['exit_code'])
print('audit_files',[(name,(p/name).stat().st_size) for name in ('AUDIT.json','AUDIT.md','CHECKS.json','CHECKS_CHAIN.json','CHECKS_FINAL.json')])
print('\n'.join((p/'AUDIT.md').read_text(encoding='utf-8').splitlines()[:19]))
"""
(HERE / 'READBACK_FAILED_01.py').write_text(stdin_source, encoding='utf-8')
observed_output = """verdict WARN blockers [] checks 29060
writer_failed_exit 1
writer_final_exit 0
audit_files [('AUDIT.json', 46390), ('AUDIT.md', 14348), ('CHECKS.json', 33623), ('CHECKS_CHAIN.json', 4621), ('CHECKS_FINAL.json', 22300)]
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
UnicodeEncodeError: 'gbk' codec can't encode character '\\u2212' in position 1921: illegal multibyte sequence
"""
(HERE / 'READBACK_FAILED_01.log').write_text(observed_output, encoding='utf-8')
readback_failure = dict(tool_chunk_id='9e18c7', actual_exit_code=1,
    command=[r'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe', '-'],
    stdin_source=str(HERE / 'READBACK_FAILED_01.py'), stdin_sha256=sha(HERE / 'READBACK_FAILED_01.py'),
    output_log=str(HERE / 'READBACK_FAILED_01.log'), output_sha256=sha(HERE / 'READBACK_FAILED_01.log'),
    capture_note='Retrospectively preserved from the actual tool call input/output; this script snapshot was not rerun.',
    error='UnicodeEncodeError during Markdown display through GBK stdout, after successful JSON fields/file-size reads.',
    impact='Display-only failure. Audit artifact generation and both deterministic verifiers had already succeeded. No source/result mutation.')
(HERE / 'READBACK_FAILED_01.json').write_text(json.dumps(readback_failure, indent=2) + '\n', encoding='utf-8')

audit['publication_qualifier_review'] = qualifier
audit['publication_conditions_satisfied'] = audit.pop('required_publication_qualifiers')
audit['required_publication_qualifiers'] = []
audit['failed_readback_invocations'] = [readback_failure]
audit['successful_artifact_writer_invocation'] = json.loads((HERE / 'WRITE_AUDIT_EXECUTION_02.json').read_text(encoding='utf-8-sig'))
audit['reviewed_hashes'][str(qualified)] = sha(qualified)
audit['reviewed_hashes'][str(master)] = sha(master)
audit['amended_at'] = datetime.now().astimezone().isoformat()
audit['artifacts'] += [str(HERE / name) for name in ('CHECKS_QUALIFIER.json', 'READBACK_FAILED_01.json', 'READBACK_FAILED_01.log', 'READBACK_FAILED_01.py', 'write_audit_FAILED_01.py', 'WRITE_AUDIT_EXECUTION.json', 'WRITE_AUDIT_EXECUTION_02.json')]
(HERE / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
md = (HERE / 'AUDIT.md').read_text(encoding='utf-8')
md = md.replace('**Required publication qualifications; no source/result edits required**', '**Publication qualifications — now satisfied; no source/result edits required**')
md += ('\n**Final qualifier acceptance**\n\n'
       'The newly supplied `SIX_RESULTS_REPORT_913_QUALIFIED.md` retains the complete original report as an exact prefix and appends only the requested receipt/provenance qualifier. Its SHA256 is `' + sha(qualified) + '`; qualifier evidence is `' + str(qualified) + ':' + str(qualifier_line) + '`. The original report SHA256 is unchanged. The corresponding master draft qualifier at `MASTER_SECTION_913_DRAFT.md:45` is accepted only for that paragraph; no wholesale master-draft review is claimed. The two publication conditions above are satisfied, with no remaining corrections or blockers. These four small checks are recorded in `CHECKS_QUALIFIER.json`; the 29,060 checks were not replayed.\n\n'
       'One later read-only Markdown display invocation returned EXIT1 (tool chunk `9e18c7`) because GBK stdout could not encode U+2212. It occurred after JSON/size reads, changed no audit or scientific files, and does not alter the successful verifier results. Its exact observed input/output and failure are preserved in `READBACK_FAILED_01.py`, `READBACK_FAILED_01.log`, and `READBACK_FAILED_01.json` as an explicitly retrospective tool receipt. The earlier private writer SyntaxError remains preserved separately.\n')
(HERE / 'AUDIT.md').write_text(md, encoding='utf-8')
print(json.dumps(dict(verdict=audit['verdict'], blockers=0, remaining_publication_qualifiers=0,
    saved_evidence_checks=29060, qualifier_checks=4, qualified_report_sha256=sha(qualified),
    preserved_failed_writer_exit=1, preserved_failed_display_exit=1)))
