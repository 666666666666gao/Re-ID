"""Read-only artifact inspection. Never imports artifact modules or tensor libraries."""
import hashlib
import json
import sys
from pathlib import Path
import ast
import difflib

sys.stdout.reconfigure(encoding='utf-8')

TRACE = Path(__file__).resolve().parent
MANIFEST = TRACE / 'input_manifest.json'
EXPECTED_MANIFEST = 'b83e0121166effe4e6e84940ade7ab593610846f93634f9bca1b195a9c837e69'
entries = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))['entries']

def digest(b):
    return hashlib.sha256(b).hexdigest()

def command_record():
    with (TRACE / 'audit_commands.txt').open('a', encoding='utf-8') as stream:
        stream.write('& ' + repr(sys.executable) + ' ' + repr(str(Path(__file__).resolve())) + ' ' + ' '.join(sys.argv[1:]) + '\n')

def emit(name, text):
    (TRACE / name).write_text(text, encoding='utf-8')
    print(text)

command_record()
action = sys.argv[1]
if action == 'inventory':
    rows = []
    for i, e in enumerate(entries):
        b = Path(e['path']).read_bytes()
        txt = b.decode('utf-8-sig')
        rows.append({'id': i, 'source': e['source_relative_path'], 'bytes': len(b),
                     'lines': len(txt.splitlines()), 'sha256': digest(b),
                     'hash_match': digest(b) == e['sha256'], 'size_match': len(b) == e['bytes']})
    result = {'manifest_sha256': digest(MANIFEST.read_bytes()),
              'manifest_hash_match': digest(MANIFEST.read_bytes()) == EXPECTED_MANIFEST,
              'files': len(rows), 'total_bytes': sum(r['bytes'] for r in rows),
              'all_hashes_match': all(r['hash_match'] and r['size_match'] for r in rows),
              'entries': rows}
    (TRACE / 'audit_inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    emit('audit_inventory.stdout.txt', '\n'.join([
        f"manifest_sha256={result['manifest_sha256']} expected_match={result['manifest_hash_match']}",
        f"files={result['files']} bytes={result['total_bytes']} all_hashes_match={result['all_hashes_match']}"
    ] + [f"{r['id']:03} {r['bytes']:8} {r['lines']:6} {r['source']}" for r in rows]))
elif action == 'read':
    texts = []
    for spec in sys.argv[2:]:
        parts = spec.split(':')
        index = int(parts[0])
        e = entries[index]
        lines = Path(e['path']).read_text(encoding='utf-8-sig').splitlines()
        start = int(parts[1]) if len(parts) > 1 else 1
        end = int(parts[2]) if len(parts) > 2 else len(lines)
        texts.append(f"FILE {index}: {e['source_relative_path']}\n" +
                     '\n'.join(f'{n}: {lines[n-1]}' for n in range(start, min(end,len(lines))+1)))
    emit('audit_read_' + '_'.join(s.replace(':','-') for s in sys.argv[2:]) + '.stdout.txt', '\n\n'.join(texts))
elif action == 'schema':
    texts = []
    for spec in sys.argv[2:]:
        index = int(spec)
        e = entries[index]
        data = json.loads(Path(e['path']).read_text(encoding='utf-8-sig'))
        def shape(obj):
            if isinstance(obj, dict):
                return {k: shape(v) for k,v in obj.items()}
            if isinstance(obj, list):
                return {'_list_len': len(obj), '_first_shape': shape(obj[0]) if obj else None}
            return obj
        texts.append(f"FILE {index}: {e['source_relative_path']}\n" + json.dumps(shape(data), indent=2, ensure_ascii=False))
    emit('audit_schema_' + '_'.join(sys.argv[2:]) + '.stdout.txt', '\n\n'.join(texts))
elif action == 'diff':
    a,b = (entries[int(x)] for x in sys.argv[2:])
    old,new = (Path(x['path']).read_text(encoding='utf-8-sig') for x in (a,b))
    emit('audit_diff_' + '_'.join(sys.argv[2:]) + '.stdout.txt', ''.join(difflib.unified_diff(
        old.splitlines(keepends=True),new.splitlines(keepends=True),
        fromfile=a['source_relative_path'],tofile=b['source_relative_path'])))
elif action == 'compact':
    texts = []
    for spec in sys.argv[2:]:
        parts = spec.split(':')
        index = int(parts[0])
        e = entries[index]
        source = Path(e['path']).read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        doc_lines = set()
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if isinstance(body,list) and body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
                doc_lines.update(range(body[0].lineno,body[0].end_lineno+1))
        lines = source.splitlines()
        start = int(parts[1]) if len(parts)>1 else 1
        end = int(parts[2]) if len(parts)>2 else len(lines)
        texts.append(f"FILE {index}: {e['source_relative_path']} (comments/blank lines/docstrings omitted only)\n" + '\n'.join(
            f'{n}: {lines[n-1]}' for n in range(start,min(end,len(lines))+1)
            if lines[n-1].strip() and not lines[n-1].lstrip().startswith('#') and n not in doc_lines))
    emit('audit_compact_' + '_'.join(s.replace(':','-') for s in sys.argv[2:]) + '.stdout.txt', '\n\n'.join(texts))
